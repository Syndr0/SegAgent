from __future__ import annotations

import operator
import sqlite3
import uuid
from typing import Annotated, Iterator, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .config import Settings
from .observability import Tracing
from .planner import Planner
from .schemas import (
    ChatRole,
    Intent,
    PlannerAction,
    PlannerDecision,
    RunEvent,
    SegmentRequest,
    TaskIntent,
    ToolObservation,
)
from .dialogue import (
    apply_reply,
    forced_structures,
    guard_decision,
    pre_plan_gate,
    rejection_notice,
)
from .guards import request_signature
from .intent import IntentClassifier
from .prompts import build_prompts, target_type_for
from .storage import ResearchStore
from .tools import (
    ContourQCTool,
    EvidenceCritic,
    MaskInspectionTool,
    ProtocolLookupTool,
    SegmentationTool,
)


class AgentState(TypedDict, total=False):
    case_id: str
    run_id: str
    conversation_id: str
    question: str
    # Prior turns, refreshed from the store at the start of each turn so the
    # planner can resolve references like "the mask you just made".
    history: list[dict]
    intent: dict
    step: int
    observations: Annotated[list[dict], merge_turn_scoped]
    decisions: Annotated[list[dict], merge_turn_scoped]
    events: Annotated[list[dict], operator.add]
    current_decision: dict
    final_answer: str | None
    critic_findings: list[dict]
    seen_signatures: Annotated[list[str], merge_turn_scoped]
    # How many times the planner re-requested work already done this turn.
    duplicate_skips: int
    # Deterministic gates that already fired, so a gate cannot re-ask the same
    # question after the user replies and the step counter resets.
    gates_fired: Annotated[list[str], merge_turn_scoped]
    # What the user answered to those gated questions (confirmed site, tumor
    # description, whether a tumor contour was rejected).
    answers: dict
    # Whether the case has any contour to work from, for the report gate.
    has_contours: bool


TURN_RESET = "__turn_reset__"


def _is_reset(item) -> bool:
    if isinstance(item, str):
        return item == TURN_RESET
    return isinstance(item, dict) and item.get(TURN_RESET) is True


def merge_turn_scoped(left: list, right: list) -> list:
    """Append updates, but a leading reset marker starts the list fresh.

    The graph thread spans the whole conversation so state survives between
    questions, which is what makes the dialogue work. These particular channels
    must still be per-turn: otherwise a new question is planned against the
    previous question's evidence, a duplicate-request guard refuses work the
    user deliberately asked for again, and a gate that fired once never asks
    again for a later question.
    """
    if right and _is_reset(right[0]):
        return list(right[1:])
    return [*left, *right]


def new_turn() -> dict:
    """State update that begins a fresh turn on an existing conversation thread."""
    return {
        "observations": [{TURN_RESET: True}],
        "seen_signatures": [TURN_RESET],
        "gates_fired": [TURN_RESET],
        "decisions": [{TURN_RESET: True}],
        "duplicate_skips": 0,
    }


def event(event_type: str, **payload) -> dict:
    return {"type": event_type, "payload": payload}


class SegAgentGraph:
    def __init__(
        self,
        settings: Settings,
        planner: Planner,
        protocol_tool: ProtocolLookupTool,
        segmentation_tool: SegmentationTool,
        critic: EvidenceCritic,
        tracing: Tracing | None = None,
        inspection_tool: MaskInspectionTool | None = None,
        qc_tool: ContourQCTool | None = None,
    ):
        self.qc_tool = qc_tool
        self.settings = settings
        self.planner = planner
        self.protocol_tool = protocol_tool
        self.segmentation_tool = segmentation_tool
        self.inspection_tool = inspection_tool
        self.critic = critic
        # Organ vocabulary, used to recognize when a tumor description names a
        # neighbouring structure and is therefore spatially anchored.
        self._organ_names = tuple(
            organ
            for protocol in getattr(protocol_tool.knowledge, "protocols", [])
            for organ in protocol.get("oars", [])
        )
        self.tracing = tracing or Tracing()
        self._checkpoint_connection = sqlite3.connect(
            str(settings.checkpoint_db), check_same_thread=False
        )
        self.checkpointer = SqliteSaver(self._checkpoint_connection)
        self.graph = self._build().compile(checkpointer=self.checkpointer)

    @staticmethod
    def _observations(state: AgentState) -> list[ToolObservation]:
        return [ToolObservation.model_validate(item) for item in state.get("observations", [])]

    def _build(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("plan", self._plan)
        graph.add_node("lookup_protocol", self._lookup_protocol)
        graph.add_node("segment", self._segment)
        graph.add_node("inspect_mask", self._inspect_mask)
        graph.add_node("run_qc", self._run_qc)
        graph.add_node("answer_question", self._answer_question)
        graph.add_node("ask_user", self._ask_user)
        graph.add_node("finish", self._finish)
        graph.add_edge(START, "plan")
        graph.add_conditional_edges(
            "plan",
            self._route,
            {
                PlannerAction.LOOKUP_PROTOCOL.value: "lookup_protocol",
                PlannerAction.SEGMENT.value: "segment",
                PlannerAction.INSPECT_MASK.value: "inspect_mask",
                PlannerAction.RUN_QC.value: "run_qc",
                PlannerAction.ANSWER_QUESTION.value: "answer_question",
                PlannerAction.ASK_USER.value: "ask_user",
                PlannerAction.FINAL.value: "finish",
            },
        )
        graph.add_edge("lookup_protocol", "plan")
        graph.add_edge("segment", "plan")
        graph.add_edge("inspect_mask", "plan")
        graph.add_edge("run_qc", "plan")
        # Answering a descriptive question completes the turn: it is a reply,
        # not a step toward some other action.
        graph.add_edge("answer_question", END)
        # Asking is no longer terminal: the reply flows back into planning, so a
        # clarification can continue the same turn (tumor and OAR dialogues).
        graph.add_edge("ask_user", "plan")
        graph.add_edge("finish", END)
        return graph

    def _inspect_mask(self, state: AgentState) -> dict:
        """Measure contours that already exist, instead of segmenting again."""
        decision = PlannerDecision.model_validate(state["current_decision"])
        if self.inspection_tool is None:
            observation = ToolObservation(
                observation_id=f"observation_{uuid.uuid4().hex[:16]}",
                tool="inspect_mask",
                summary="Mask inspection is not available in this deployment.",
                data={"error": "no inspection tool"},
            )
            return {
                "observations": [observation.model_dump(mode="json")],
                "events": [event("observation", observation=observation.model_dump(mode="json"))],
            }
        with self.tracing.span(
            "execute_tool",
            **{
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "inspect_mask",
                "segagent.case_id": state["case_id"],
            },
        ):
            _, observation = self.inspection_tool.run(state["case_id"], decision.structures)
        return {
            "observations": [observation.model_dump(mode="json")],
            "events": [
                event("tool_started", tool="inspect_mask", structures=decision.structures),
                event("observation", observation=observation.model_dump(mode="json")),
            ],
        }

    def _answer_question(self, state: AgentState) -> dict:
        """Answer a descriptive question about the image without running tools."""
        with self.tracing.span(
            "invoke_agent",
            **{
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": "segagent-vqa",
                "segagent.case_id": state["case_id"],
            },
        ):
            try:
                answer = self.planner.answer(
                    state["case_id"], state["question"], state.get("history", [])
                )
            except Exception as exc:
                answer = f"I could not read the image to answer that: {exc}"
        answer = (answer or "").strip() or "I could not describe this image."
        findings = self.critic.review(answer, [])
        blocking = [item for item in findings if item.severity == "error"]
        if blocking:
            answer = (
                "I cannot give that description because it stated values that "
                "were not measured.\n\n"
                + "\n".join(f"- {item.message}" for item in blocking)
            )
        return {
            "final_answer": answer,
            "gates_fired": ["vqa"],
            "critic_findings": [item.model_dump(mode="json") for item in findings],
            "events": [event("answer", text=answer), event("run_completed")],
        }

    def _run_qc(self, state: AgentState) -> dict:
        """Compare uploaded contours with an independent model contour."""
        decision = PlannerDecision.model_validate(state["current_decision"])
        if self.qc_tool is None:
            observation = ToolObservation(
                observation_id=f"observation_{uuid.uuid4().hex[:16]}",
                tool="run_qc",
                summary="Contour QC is not available in this deployment.",
                data={"error": "no qc tool"},
            )
            return {
                "observations": [observation.model_dump(mode="json")],
                "events": [event("observation", observation=observation.model_dump(mode="json"))],
            }
        intent_data = state.get("intent") or {}
        site = decision.site_query or intent_data.get("site") or None
        with self.tracing.span(
            "execute_tool",
            **{
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "run_qc",
                "segagent.case_id": state["case_id"],
            },
        ):
            _, observation = self.qc_tool.run(state["case_id"], site)
        return {
            "observations": [observation.model_dump(mode="json")],
            "events": [
                event("tool_started", tool="run_qc", query=site),
                event("observation", observation=observation.model_dump(mode="json")),
            ],
        }

    def _ask_user(self, state: AgentState) -> dict:
        """Ask the user something and suspend until they reply.

        The interrupt payload is surfaced by the service as a question_pending
        event; resuming supplies the reply, which becomes the current question
        so the planner continues with the new information.
        """
        decision = PlannerDecision.model_validate(state["current_decision"])
        question = decision.user_message or "Could you clarify what you would like me to do?"
        gate = decision.gate or ""
        reply = interrupt(
            {
                "question": question,
                "rationale_summary": decision.rationale_summary,
                "gate": gate,
            }
        )
        text = reply.get("text") if isinstance(reply, dict) else reply
        text = " ".join(str(text or "").strip().split())
        update = {
            "question": text or state.get("question", ""),
            # Reset the step budget: the dialogue advanced because a human
            # answered, so the loop guard should not count earlier steps.
            "step": 0,
            "gates_fired": [gate] if gate else [],
            "events": [event("reply_recorded", text=text, question=question)],
        }
        if gate:
            update.update(apply_reply(state, gate, text, self._organ_names))
        return update

    @staticmethod
    def _summarize(observations: list[ToolObservation]) -> str:
        """Fall-back answer assembled only from what the tools actually reported."""
        useful = [
            item.summary
            for item in observations
            if item.summary and not item.data.get("skipped")
        ]
        if not useful:
            return (
                "I could not gather any new evidence for that. Could you rephrase "
                "what you would like me to contour or measure?"
            )
        return "Here is what I have measured so far:\n\n" + "\n\n".join(useful)

    def _plan(self, state: AgentState) -> dict:
        step = int(state.get("step", 0)) + 1
        observations = self._observations(state)
        with self.tracing.span(
            "invoke_agent",
            **{
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": "segagent-planner",
                "gen_ai.conversation.id": state["run_id"],
                "segagent.case_id": state["case_id"],
                "segagent.step": step,
            },
        ):
            stop = rejection_notice(state)
            gate = pre_plan_gate(state, self._organ_names, bool(state.get("has_contours")))
            # A planner that keeps re-requesting work already done will otherwise
            # burn the whole step budget and end with no answer, so answer from
            # the evidence already gathered instead.
            if int(state.get("duplicate_skips", 0)) >= 2:
                decision = PlannerDecision(
                    action=PlannerAction.FINAL,
                    rationale_summary="The requested work was already done; answering from the existing evidence.",
                    confidence=0.7,
                    final_answer=self._summarize(observations),
                )
            elif stop is not None:
                decision = PlannerDecision(
                    action=PlannerAction.FINAL,
                    rationale_summary="The user rejected the tumor contour; stopping instead of repeating it.",
                    confidence=1.0,
                    final_answer=stop,
                )
            elif gate is not None:
                decision = gate
            elif step > self.settings.max_steps:
                decision = PlannerDecision(
                    action=PlannerAction.FINAL,
                    rationale_summary="The configured step budget was reached.",
                    confidence=0.5,
                    final_answer=self._summarize(observations),
                )
            else:
                decision = guard_decision(
                    state,
                    self.planner.decide(
                        state["case_id"],
                        state["question"],
                        step,
                        observations,
                        history=state.get("history", []),
                    ),
                )
        events = []
        # The model's own reasoning is surfaced as its own step so the UI can
        # fold it; it is never treated as evidence.
        if decision.thinking:
            events.append(event("thinking", step=step, thinking=decision.thinking))
        events.append(
            event(
                "planner_decision",
                step=step,
                action=decision.action.value,
                rationale_summary=decision.rationale_summary,
                structures=decision.structures,
                site_query=decision.site_query,
            )
        )
        return {
            "step": step,
            "current_decision": decision.model_dump(mode="json"),
            "decisions": [decision.model_dump(mode="json")],
            "events": events,
        }

    @staticmethod
    def _route(state: AgentState) -> str:
        return PlannerDecision.model_validate(state["current_decision"]).action.value

    def _lookup_protocol(self, state: AgentState) -> dict:
        decision = PlannerDecision.model_validate(state["current_decision"])
        query = decision.site_query or state["question"]
        with self.tracing.span(
            "retrieval",
            **{
                "gen_ai.operation.name": "retrieval",
                "segagent.case_id": state["case_id"],
            },
        ):
            _, observation = self.protocol_tool.run(query)
        return {
            "observations": [observation.model_dump(mode="json")],
            "events": [
                event("tool_started", tool="lookup_protocol", query=query),
                event("observation", observation=observation.model_dump(mode="json")),
            ],
        }

    @staticmethod
    def _protocol_oars(state: AgentState) -> list[str]:
        """The OAR list from this turn's protocol lookup, if one was matched."""
        for item in reversed(state.get("observations") or []):
            if item.get("tool") != "lookup_protocol":
                continue
            data = item.get("data") or {}
            if data.get("matched"):
                return [str(name) for name in data.get("oars", []) or []]
        return []

    def _segment(self, state: AgentState) -> dict:
        decision = PlannerDecision.model_validate(state["current_decision"])
        # The workflow, not the model, decides what a confirmed OAR request or a
        # tumor-refinement stage contours; the model picking a subset is exactly
        # how a full protocol turns into two organs.
        override = forced_structures(
            state, self._protocol_oars(state), state.get("observations") or []
        )
        stage = ""
        if override is not None:
            structures, stage = override
            decision = decision.model_copy(update={"structures": structures})
        if not decision.structures:
            observation = ToolObservation(
                observation_id=f"observation_{uuid.uuid4().hex[:16]}",
                tool="segment",
                summary="Segmentation was not run because the typed structure list was empty.",
                data={"error": "empty structures"},
            )
            return {
                "observations": [observation.model_dump(mode="json")],
                "events": [event("observation", observation=observation.model_dump(mode="json"))],
            }
        task_intent = None
        intent_data = state.get("intent")
        if intent_data:
            try:
                task_intent = TaskIntent.model_validate(intent_data)
            except Exception:
                task_intent = None
        prompts = build_prompts(task_intent, decision.structures) or list(decision.structures)
        signature = request_signature(
            state["case_id"],
            "segment",
            prompts,
            getattr(self.segmentation_tool.backend, "version", ""),
        )
        if signature in state.get("seen_signatures", []):
            observation = ToolObservation(
                observation_id=f"observation_{uuid.uuid4().hex[:16]}",
                tool="segment",
                summary=(
                    "This exact segmentation was already produced and was not repeated. "
                    "Refine the requested structures or finalize with the existing evidence."
                ),
                data={"skipped": "duplicate_request", "signature": signature},
            )
            return {
                "observations": [observation.model_dump(mode="json")],
                "duplicate_skips": int(state.get("duplicate_skips", 0)) + 1,
                "events": [event("observation", observation=observation.model_dump(mode="json"))],
            }
        target_type = target_type_for(task_intent)
        request = SegmentRequest(
            case_id=state["case_id"],
            structures=prompts,
            purpose=state["question"],
            target_type=target_type,
        )
        with self.tracing.span(
            "execute_tool",
            **{
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "segment",
                "segagent.case_id": state["case_id"],
                "segagent.structure_count": len(request.structures),
            },
        ):
            result, observation = self.segmentation_tool.run(request)
        # Masks stream straight to the viewer as artifacts; there is no blocking
        # review. The target_type rides on each artifact so the UI can flag
        # tumor targets (GTV) as unvalidated and requiring clinician review.
        artifact_events = [
            event("artifact", artifact=item.mask.model_dump(mode="json"))
            for item in result.measurements
            if item.mask is not None
        ]
        # Record which stage of a dictated workflow just completed, so the next
        # planning step advances instead of repeating it.
        answers = dict(state.get("answers") or {})
        produced = any(item.mask is not None for item in result.measurements)
        if stage and produced:
            if stage.startswith("blind"):
                answers["gtv_first_mask"] = True
            elif stage.startswith("contouring neighbouring"):
                answers["gtv_neighbours"] = True
            elif stage.startswith("re-contouring"):
                answers["gtv_refined"] = True
            elif stage.startswith("contouring the confirmed"):
                answers["oar_contoured"] = True
        return {
            "observations": [observation.model_dump(mode="json")],
            "seen_signatures": [signature],
            "answers": answers,
            "events": [
                event("tool_started", tool="segment", structures=request.structures, stage=stage or None),
                event("observation", observation=observation.model_dump(mode="json")),
                *artifact_events,
            ],
        }

    def _finish(self, state: AgentState) -> dict:
        decision = PlannerDecision.model_validate(state["current_decision"])
        answer = decision.final_answer or "No final answer was produced."
        observations = self._observations(state)
        findings = self.critic.review(answer, observations)
        blocking = [item for item in findings if item.severity == "error"]
        if blocking:
            answer = (
                "I cannot return the proposed answer because its claims were not fully "
                "grounded in the recorded tool evidence.\n\n"
                + "\n".join(f"- {item.message}" for item in blocking)
            )
        elif findings:
            answer += "\n\nEvidence limitations:\n" + "\n".join(
                f"- {item.message}" for item in findings
            )
        return {
            "final_answer": answer,
            "critic_findings": [item.model_dump(mode="json") for item in findings],
            "events": [
                event("answer", text=answer, critic_findings=[item.model_dump(mode="json") for item in findings]),
                event("run_completed"),
            ],
        }


class WorkflowService:
    """Persists graph updates as typed events and streams them to the client."""

    def __init__(
        self,
        store: ResearchStore,
        graph: SegAgentGraph,
        intent_classifier: "IntentClassifier | None" = None,
    ):
        self.store = store
        self.graph = graph
        self.intent_classifier = intent_classifier

    def send(self, case_id: str, text: str) -> tuple[str, Iterator[RunEvent]]:
        """Handle one user message.

        If the agent is waiting on a question for this case, the message is the
        reply and continues that turn; otherwise it opens a new turn. The client
        therefore only ever needs one "send" action.
        """
        text = " ".join(text.strip().split())
        if not text or len(text) > 4000:
            raise ValueError("message must contain 1-4000 characters")
        conversation = self.store.conversation_for_case(case_id)
        if conversation.pending_run_id:
            try:
                return self.reply(conversation.pending_run_id, text)
            except (ValueError, FileNotFoundError):
                # The pending turn is gone or no longer waiting; start a new one.
                conversation.pending_run_id = None
                conversation.pending_question = None
                self.store.save_conversation(conversation)

        run = self.store.create_run(case_id, text, conversation.conversation_id)
        self.store.append_message(
            conversation.conversation_id, case_id, ChatRole.USER, text, run.run_id
        )
        history = [
            item.model_dump(mode="json")
            for item in self.store.messages(conversation.conversation_id)
        ]

        intent_payload: dict = {}
        # Uploaded contours drive QC; any contour or generated mask is enough to
        # report on.
        has_uploaded = False
        has_any_mask = False
        try:
            case = self.store.get_case(case_id)
            has_uploaded = bool(case.contours)
            has_any_mask = any(
                item.kind in {"contour", "mask"} for item in case.artifacts
            )
        except Exception:
            pass
        if self.intent_classifier is not None:
            try:
                task_intent = self.intent_classifier.classify(
                    text, has_contours=has_uploaded
                )
                intent_payload = {"intent": task_intent.model_dump(mode="json")}
            except Exception as exc:
                # A classifier failure silently disables every intent gate, so
                # surface it in the trace instead of degrading without a trace.
                intent_payload = {
                    "intent_error": f"{type(exc).__name__}: {exc}",
                }

        def stream() -> Iterator[RunEvent]:
            run.status = "running"
            self.store.save_run(run)
            conversation.turn_count += 1
            self.store.save_conversation(conversation)
            yield self._persist(
                run.run_id,
                case_id,
                event("run_started", question=text, **intent_payload),
            )
            initial: AgentState = {
                "case_id": case_id,
                "run_id": run.run_id,
                "conversation_id": conversation.conversation_id,
                "question": text,
                "history": history,
                "intent": intent_payload.get("intent", {}),
                "has_contours": has_any_mask,
                "step": 0,
                "events": [],
                **new_turn(),
            }
            yield from self._execute(run.run_id, conversation.conversation_id, initial)

        return run.run_id, stream()

    def reply(self, run_id: str, text: str) -> tuple[str, Iterator[RunEvent]]:
        """Answer the agent's pending question and resume the same turn."""
        text = " ".join(text.strip().split())
        if not text or len(text) > 4000:
            raise ValueError("reply must contain 1-4000 characters")
        run = self.store.get_run(run_id)
        if run.status != "waiting_reply":
            raise ValueError("this turn is not waiting for a reply")
        conversation_id = run.conversation_id or ""
        if conversation_id:
            self.store.append_message(
                conversation_id, run.case_id, ChatRole.USER, text, run_id
            )

        def stream() -> Iterator[RunEvent]:
            if conversation_id:
                conversation = self.store.get_conversation(conversation_id)
                conversation.pending_question = None
                conversation.pending_run_id = None
                self.store.save_conversation(conversation)
            run.status = "running"
            self.store.save_run(run)
            yield from self._execute(
                run_id, conversation_id, Command(resume={"text": text})
            )

        return run_id, stream()

    def _execute(
        self,
        run_id: str,
        conversation_id: str,
        value: AgentState | Command,
    ) -> Iterator[RunEvent]:
        run = self.store.get_run(run_id)
        # The thread is the conversation, not the turn, so evidence and dialogue
        # state accumulate across every question asked about one patient.
        config = {"configurable": {"thread_id": conversation_id or run_id}}
        try:
            for update in self.graph.graph.stream(value, config=config, stream_mode="updates"):
                if "__interrupt__" in update:
                    yield from self._suspend(run_id, run.case_id, conversation_id, update)
                    return
                for node_update in update.values():
                    if not isinstance(node_update, dict):
                        continue
                    for raw in node_update.get("events", []):
                        persisted = self._persist(run_id, run.case_id, raw)
                        yield persisted
                        if persisted.type == "answer" and conversation_id:
                            self.store.append_message(
                                conversation_id,
                                run.case_id,
                                ChatRole.AGENT,
                                str(persisted.payload.get("text", "")),
                                run_id,
                            )
                        if persisted.type == "run_completed":
                            current = self.store.get_run(run_id)
                            current.status = "completed"
                            self.store.save_run(current)
        except Exception as exc:
            current = self.store.get_run(run_id)
            current.status = "failed"
            self.store.save_run(current)
            yield self._persist(
                run_id,
                current.case_id,
                event("error", message=str(exc), error_type=type(exc).__name__),
            )

    def _suspend(
        self, run_id: str, case_id: str, conversation_id: str, update: dict
    ) -> Iterator[RunEvent]:
        """Record the agent's pending question and pause the turn."""
        payload: dict = {}
        for item in update["__interrupt__"]:
            value = getattr(item, "value", item)
            if isinstance(value, dict):
                payload = value
                break
        question = str(payload.get("question") or "Could you clarify?")
        current = self.store.get_run(run_id)
        current.status = "waiting_reply"
        self.store.save_run(current)
        if conversation_id:
            conversation = self.store.get_conversation(conversation_id)
            conversation.pending_question = question
            conversation.pending_run_id = run_id
            self.store.save_conversation(conversation)
            self.store.append_message(
                conversation_id, case_id, ChatRole.AGENT, question, run_id,
                metadata={"pending": True},
            )
        yield self._persist(run_id, case_id, event("question_pending", **payload))

    def _persist(self, run_id: str, case_id: str, raw: dict) -> RunEvent:
        run = self.store.get_run(run_id)
        item = RunEvent(
            event_id=f"event_{uuid.uuid4().hex[:16]}",
            run_id=run_id,
            case_id=case_id,
            sequence=run.event_count + 1,
            type=raw["type"],
            payload=raw.get("payload", {}),
        )
        self.store.append_event(item)
        return item
