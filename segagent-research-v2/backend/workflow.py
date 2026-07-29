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
from .dialogue import apply_reply, guard_decision, pre_plan_gate, rejection_notice
from .guards import request_signature
from .intent import IntentClassifier
from .prompts import build_prompts, target_type_for
from .storage import ResearchStore
from .tools import EvidenceCritic, MaskInspectionTool, ProtocolLookupTool, SegmentationTool


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
    observations: Annotated[list[dict], operator.add]
    decisions: Annotated[list[dict], operator.add]
    events: Annotated[list[dict], operator.add]
    current_decision: dict
    final_answer: str | None
    critic_findings: list[dict]
    seen_signatures: Annotated[list[str], operator.add]
    # Deterministic gates that already fired, so a gate cannot re-ask the same
    # question after the user replies and the step counter resets.
    gates_fired: Annotated[list[str], operator.add]
    # What the user answered to those gated questions (confirmed site, tumor
    # description, whether a tumor contour was rejected).
    answers: dict


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
    ):
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
                PlannerAction.ASK_USER.value: "ask_user",
                PlannerAction.FINAL.value: "finish",
            },
        )
        graph.add_edge("lookup_protocol", "plan")
        graph.add_edge("segment", "plan")
        graph.add_edge("inspect_mask", "plan")
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
            gate = pre_plan_gate(state, self._organ_names)
            if stop is not None:
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
                    final_answer=(
                        "The agent reached its step limit. Evidence collected so far:\n\n"
                        + "\n\n".join(item.summary for item in observations)
                    ),
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

    def _segment(self, state: AgentState) -> dict:
        decision = PlannerDecision.model_validate(state["current_decision"])
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
        return {
            "observations": [observation.model_dump(mode="json")],
            "seen_signatures": [signature],
            "events": [
                event("tool_started", tool="segment", structures=request.structures),
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
        if self.intent_classifier is not None:
            try:
                case = self.store.get_case(case_id)
                task_intent = self.intent_classifier.classify(
                    text, has_contours=bool(case.contours)
                )
                intent_payload = {"intent": task_intent.model_dump(mode="json")}
            except Exception:
                intent_payload = {}

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
                "step": 0,
                "observations": [],
                "decisions": [],
                "events": [],
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
