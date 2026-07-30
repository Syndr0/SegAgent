from __future__ import annotations

import sqlite3
import uuid
from inspect import signature
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
    MessageArtifactContext,
    PlannerAnswer,
    PlannerAction,
    PlannerDecision,
    RunEvent,
    SegmentRequest,
    TargetType,
    TaskIntent,
    ToolObservation,
)
from .dialogue import (
    STAGE_LABELS,
    STAGE_MARKERS,
    STAGE_TUMOR_NEIGHBOURS,
    STAGE_TUMOR_INITIAL,
    apply_reply,
    forced_structures,
    guard_decision,
    pre_plan_gate,
    required_workflow_decision,
    rejection_notice,
)
from .guards import request_signature
from .intent import IntentClassifier
from .prompts import build_prompts, clean_prompt, target_type_for
from .qc import ReportEvidenceBuilder, assess_report_readiness
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
    events: Annotated[list[dict], merge_turn_scoped]
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
    # Case-owned artifact IDs explicitly attached by the viewer to this turn.
    message_context: dict
    # Typed suspended interaction; reset at every new turn.
    pending_interaction: dict
    report_scope: str
    report_readiness: dict
    artifact_inventory: list[dict]


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
        "events": [{TURN_RESET: True}],
        "duplicate_skips": 0,
        "answers": {},
        "pending_interaction": {},
        "current_decision": {},
        "final_answer": None,
        "critic_findings": [],
        "report_scope": "comprehensive",
        "report_readiness": {},
    }


def event(event_type: str, **payload) -> dict:
    return {"type": event_type, "payload": payload}


def emit_live(raw: dict) -> None:
    """Write a custom LangGraph stream item while a node is still running."""

    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()(raw)
    except (ImportError, LookupError, RuntimeError):
        # Direct unit calls have no graph stream writer; terminal events still
        # remain in the node update for deterministic testing/replay.
        return


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
        self._site_aliases = {
            str(alias).casefold(): str(protocol.get("site", alias))
            for protocol in getattr(protocol_tool.knowledge, "protocols", [])
            for alias in [protocol.get("site", ""), *(protocol.get("aliases", []) or [])]
            if alias
        }
        self.report_builder = ReportEvidenceBuilder(segmentation_tool.store)
        self.tracing = tracing or Tracing()
        self._checkpoint_connection = sqlite3.connect(
            str(settings.checkpoint_db), check_same_thread=False
        )
        self.checkpointer = SqliteSaver(self._checkpoint_connection)
        self.graph = self._build().compile(checkpointer=self.checkpointer)

    @staticmethod
    def _observations(state: AgentState) -> list[ToolObservation]:
        return [ToolObservation.model_validate(item) for item in state.get("observations", [])]

    def _planner_context(self, state: AgentState) -> dict:
        return {
            "intent": state.get("intent") or {},
            "gates_fired": list(state.get("gates_fired") or []),
            "answers": dict(state.get("answers") or {}),
            "pending_interaction": dict(state.get("pending_interaction") or {}),
            "message_context": dict(state.get("message_context") or {}),
            "report_scope": state.get("report_scope") or "comprehensive",
            "report_readiness": dict(state.get("report_readiness") or {}),
            "artifact_inventory": self.segmentation_tool.store.artifact_inventory(
                state["case_id"]
            ),
        }

    def _call_planner_decide(
        self,
        state: AgentState,
        step: int,
        observations: list[ToolObservation],
        on_delta,
    ) -> PlannerDecision:
        method = self.planner.decide
        params = signature(method).parameters
        kwargs = {"history": state.get("history", [])}
        if "context" in params:
            kwargs["context"] = self._planner_context(state)
        if "on_thinking_delta" in params:
            kwargs["on_thinking_delta"] = on_delta
        return method(
            state["case_id"], state["question"], step, observations, **kwargs
        )

    def _call_planner_answer(self, state: AgentState, on_delta) -> PlannerAnswer:
        method = self.planner.answer
        params = signature(method).parameters
        kwargs = {"history": state.get("history", [])}
        if "context" in params:
            kwargs["context"] = self._planner_context(state)
        if "on_thinking_delta" in params:
            kwargs["on_thinking_delta"] = on_delta
        result = method(state["case_id"], state["question"], **kwargs)
        if isinstance(result, PlannerAnswer):
            return result
        if isinstance(result, dict):
            return PlannerAnswer.model_validate(result)
        return PlannerAnswer(text=str(result or ""))

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
        emit_live(
            event(
                "phase_started",
                phase="inspect_mask",
                label="Inspecting existing case masks",
                structures=decision.structures,
            )
        )
        emit_live(event("tool_started", tool="inspect_mask", structures=decision.structures))
        if self.inspection_tool is None:
            observation = ToolObservation(
                observation_id=f"observation_{uuid.uuid4().hex[:16]}",
                tool="inspect_mask",
                summary="Mask inspection is not available in this deployment.",
                data={"error": "no inspection tool"},
            )
            return {
                "observations": [observation.model_dump(mode="json")],
                "events": [
                    event(
                        "phase_failed",
                        phase="inspect_mask",
                        label="Mask inspection unavailable",
                    ),
                    event("observation", observation=observation.model_dump(mode="json")),
                ],
            }
        try:
            with self.tracing.span(
                "execute_tool",
                **{
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": "inspect_mask",
                    "segagent.case_id": state["case_id"],
                },
            ):
                _, observation = self.inspection_tool.run(
                    state["case_id"], decision.structures
                )
        except Exception as exc:
            emit_live(
                event(
                    "phase_failed",
                    phase="inspect_mask",
                    label="Mask inspection failed",
                    message=str(exc),
                )
            )
            raise
        return {
            "observations": [observation.model_dump(mode="json")],
            "gates_fired": [decision.gate] if decision.gate else [],
            "events": [
                event("observation", observation=observation.model_dump(mode="json")),
                event(
                    "phase_completed",
                    phase="inspect_mask",
                    label="Existing mask measurements ready",
                ),
            ],
        }

    def _answer_question(self, state: AgentState) -> dict:
        """Answer a descriptive question about the image without running tools."""
        emit_live(
            event(
                "phase_started",
                phase="answer_question",
                label="Reading image and referenced mask views",
            )
        )

        def on_delta(delta: str) -> None:
            emit_live(
                event(
                    "thinking_delta",
                    delta=delta,
                    channel="vqa",
                )
            )

        answer_failed = False
        with self.tracing.span(
            "invoke_agent",
            **{
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": "segagent-vqa",
                "segagent.case_id": state["case_id"],
            },
        ):
            try:
                response = self._call_planner_answer(state, on_delta)
            except Exception as exc:
                answer_failed = True
                emit_live(
                    event(
                        "phase_failed",
                        phase="answer_question",
                        label="Visual answer failed",
                        message=str(exc),
                    )
                )
                response = PlannerAnswer(
                    text=f"I could not read the image to answer that: {exc}"
                )
        answer = response.text
        answer = (answer or "").strip() or "I could not describe this image."
        findings = self.critic.review(answer, [])
        blocking = [item for item in findings if item.severity == "error"]
        if blocking:
            answer = (
                "I cannot give that description because it stated values that "
                "were not measured.\n\n"
                + "\n".join(f"- {item.message}" for item in blocking)
            )
        events = []
        if response.thinking:
            events.append(
                event(
                    "thinking",
                    thinking=response.thinking,
                    channel="vqa",
                    complete=True,
                )
            )
        if not answer_failed:
            events.append(
                event(
                    "phase_completed",
                    phase="answer_question",
                    label="Visual answer ready",
                )
            )
        events.extend([event("answer", text=answer), event("run_completed")])
        return {
            "final_answer": answer,
            "gates_fired": [
                (
                    (state.get("current_decision") or {}).get("gate")
                    or "vqa"
                )
            ],
            "critic_findings": [item.model_dump(mode="json") for item in findings],
            "events": events,
        }

    def _run_qc(self, state: AgentState) -> dict:
        """Compare uploaded contours with an independent model contour."""
        decision = PlannerDecision.model_validate(state["current_decision"])
        emit_live(
            event(
                "phase_started",
                phase="run_qc",
                label="Comparing uploaded organ contours and describing targets",
            )
        )
        if self.qc_tool is None:
            observation = ToolObservation(
                observation_id=f"observation_{uuid.uuid4().hex[:16]}",
                tool="run_qc",
                summary="Contour QC is not available in this deployment.",
                data={"error": "no qc tool"},
            )
            return {
                "observations": [observation.model_dump(mode="json")],
                "events": [
                    event(
                        "phase_failed",
                        phase="run_qc",
                        label="Contour QC unavailable",
                    ),
                    event("observation", observation=observation.model_dump(mode="json")),
                ],
            }
        intent_data = state.get("intent") or {}
        site = decision.site_query or intent_data.get("site") or None
        emit_live(event("tool_started", tool="run_qc", query=site))
        try:
            with self.tracing.span(
                "execute_tool",
                **{
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": "run_qc",
                    "segagent.case_id": state["case_id"],
                },
            ):
                report, observation = self.qc_tool.run(state["case_id"], site)
        except Exception as exc:
            emit_live(
                event(
                    "phase_failed",
                    phase="run_qc",
                    label="Contour QC failed",
                    message=str(exc),
                )
            )
            raise
        artifacts = []
        seen_artifacts: set[str] = set()
        for comparison in report.comparisons:
            refs = [
                comparison.model_mask,
                *(comparison.difference_overlays or []),
            ]
            for ref in refs:
                if ref is not None and ref.artifact_id not in seen_artifacts:
                    seen_artifacts.add(ref.artifact_id)
                    artifacts.append(event("artifact", artifact=ref.model_dump(mode="json")))
        return {
            "observations": [observation.model_dump(mode="json")],
            "events": [
                event("observation", observation=observation.model_dump(mode="json")),
                *artifacts,
                event(
                    "phase_completed",
                    phase="run_qc",
                    label="Contour QC evidence ready",
                    comparison_count=len(report.comparisons),
                    target_count=len(report.targets),
                ),
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
        proposed_site = (
            self._site_aliases.get((decision.site_query or "").casefold())
            if decision.site_query
            else (state.get("answers") or {}).get("proposed_site")
        )
        pending = {
            "gate": gate,
            "question": question,
            "stage": (
                decision.workflow_stage.value if decision.workflow_stage else None
            ),
            "proposed_site": proposed_site,
        }
        reply = interrupt(
            {
                "question": question,
                "rationale_summary": decision.rationale_summary,
                "gate": gate,
                "pending_interaction": pending,
            }
        )
        text = reply.get("text") if isinstance(reply, dict) else reply
        reply_context = (
            reply.get("message_context")
            if isinstance(reply, dict)
            else state.get("message_context", {})
        )
        reply_intent = reply.get("intent") if isinstance(reply, dict) else None
        reply_has_contours = (
            reply.get("has_contours") if isinstance(reply, dict) else None
        )
        text = " ".join(str(text or "").strip().split())
        update = {
            "question": text or state.get("question", ""),
            # Reset the step budget: the dialogue advanced because a human
            # answered, so the loop guard should not count earlier steps.
            "step": 0,
            "gates_fired": [gate] if gate else [],
            "pending_interaction": {},
            "message_context": reply_context or {},
            "history": [
                *(state.get("history") or []),
                {"role": ChatRole.AGENT.value, "text": question},
                {"role": ChatRole.USER.value, "text": text},
            ],
            "events": [event("reply_recorded", text=text, question=question)],
        }
        if reply_has_contours is not None:
            update["has_contours"] = bool(reply_has_contours)
        if gate in {"clarify_intent", "missing_input"} and reply_intent:
            update["intent"] = reply_intent
        elif gate == "mask_reference" and reply_intent:
            # A short reply such as "left eye" classifies as an organ request
            # in isolation. Here it answers "which existing mask?", so retain
            # the mask-question speech act and carry over only its resolved
            # canonical target labels.
            reply_targets = list(reply_intent.get("targets") or [])
            if reply_targets:
                mask_intent = dict(state.get("intent") or {})
                mask_intent["intent"] = Intent.MASK_QUESTION.value
                mask_intent["targets"] = reply_targets
                update["intent"] = mask_intent
        if gate:
            reply_state = dict(state)
            if gate == "oar_site" and pending.get("proposed_site"):
                reply_answers = dict(state.get("answers") or {})
                reply_answers["proposed_site"] = pending["proposed_site"]
                reply_state["answers"] = reply_answers
            update.update(
                apply_reply(
                    reply_state,
                    gate,
                    text,
                    self._organ_names,
                    self._site_aliases,
                )
            )
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

    def _report_evidence(
        self, state: AgentState, observations: list[ToolObservation]
    ) -> tuple[dict, ToolObservation | None]:
        if ((state.get("intent") or {}).get("intent")) != Intent.REPORT.value:
            return {}, None
        scope = str(state.get("report_scope") or "comprehensive")
        case = self.segmentation_tool.store.get_case(state["case_id"])
        readiness = assess_report_readiness(case, scope).model_dump(mode="json")
        if not readiness.get("ready") or any(
            item.tool == "build_report_evidence" for item in observations
        ):
            return readiness, None
        bundle = self.report_builder.build(state["case_id"], scope)
        evidence_ids = list(
            dict.fromkeys(
                artifact_id
                for metric in bundle.metric_evidence
                for artifact_id in metric.artifact_ids
            )
        )
        observation = ToolObservation(
            observation_id=f"observation_{uuid.uuid4().hex[:16]}",
            tool="build_report_evidence",
            summary=(
                f"Measured report evidence for {len(bundle.structures)} structures "
                f"under the explicit {scope} scope. {bundle.readiness.reason}"
            ),
            data={"report_evidence": bundle.model_dump(mode="json")},
            evidence_ids=evidence_ids,
        )
        return readiness, observation

    def _plan(self, state: AgentState) -> dict:
        step = int(state.get("step", 0)) + 1
        observations = self._observations(state)
        report_readiness, report_observation = self._report_evidence(state, observations)
        added_observations: list[dict] = []
        if report_observation is not None:
            observations.append(report_observation)
            added_observations.append(report_observation.model_dump(mode="json"))
        planning_state = dict(state)
        planning_state["artifact_inventory"] = (
            self.segmentation_tool.store.artifact_inventory(state["case_id"])
        )
        if report_readiness:
            planning_state["report_readiness"] = report_readiness
        emit_live(
            event(
                "phase_started",
                phase="planning",
                label="Planning the next evidence-grounded action",
                step=step,
            )
        )

        def on_delta(delta: str) -> None:
            emit_live(
                event(
                    "thinking_delta",
                    delta=delta,
                    step=step,
                    channel="planner",
                )
            )

        try:
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
                stop = rejection_notice(planning_state)
                gate = pre_plan_gate(
                    planning_state,
                    self._organ_names,
                    bool(state.get("has_contours")),
                )
                mandatory = required_workflow_decision(
                    planning_state,
                    self._protocol_oars(planning_state),
                    [item.model_dump(mode="json") for item in observations],
                    self._organ_names,
                )
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
                elif mandatory is not None:
                    decision = mandatory
                elif step > self.settings.max_steps:
                    decision = PlannerDecision(
                        action=PlannerAction.FINAL,
                        rationale_summary="The configured step budget was reached.",
                        confidence=0.5,
                        final_answer=self._summarize(observations),
                    )
                else:
                    decision = guard_decision(
                        planning_state,
                        self._call_planner_decide(
                            planning_state, step, observations, on_delta
                        ),
                    )
        except Exception as exc:
            emit_live(
                event(
                    "phase_failed",
                    phase="planning",
                    label="Planning failed",
                    step=step,
                    message=str(exc),
                )
            )
            raise
        events = []
        # The model's own reasoning is surfaced as its own step so the UI can
        # fold it; it is never treated as evidence.
        if decision.thinking:
            events.append(
                event(
                    "thinking",
                    step=step,
                    thinking=decision.thinking,
                    channel="planner",
                    complete=True,
                )
            )
        events.append(
            event(
                "reasoning_summary",
                text=decision.rationale_summary,
                step=step,
                action=decision.action.value,
            )
        )
        events.append(
            event(
                "planner_decision",
                step=step,
                action=decision.action.value,
                rationale_summary=decision.rationale_summary,
                structures=decision.structures,
                site_query=decision.site_query,
                workflow_stage=(
                    decision.workflow_stage.value if decision.workflow_stage else None
                ),
            )
        )
        events.append(
            event(
                "phase_completed",
                phase="planning",
                label="Next action selected",
                step=step,
                action=decision.action.value,
            )
        )
        if report_observation is not None:
            events.insert(
                0,
                event(
                    "observation",
                    observation=report_observation.model_dump(mode="json"),
                ),
            )
        return {
            "step": step,
            "current_decision": decision.model_dump(mode="json"),
            "decisions": [decision.model_dump(mode="json")],
            "observations": added_observations,
            "report_readiness": report_readiness or state.get("report_readiness", {}),
            "events": events,
        }

    @staticmethod
    def _route(state: AgentState) -> str:
        return PlannerDecision.model_validate(state["current_decision"]).action.value

    def _lookup_protocol(self, state: AgentState) -> dict:
        decision = PlannerDecision.model_validate(state["current_decision"])
        query = decision.site_query or state["question"]
        emit_live(
            event(
                "phase_started",
                phase="lookup_protocol",
                label="Retrieving the confirmed-site OAR protocol",
                query=query,
            )
        )
        emit_live(event("tool_started", tool="lookup_protocol", query=query))
        try:
            with self.tracing.span(
                "retrieval",
                **{
                    "gen_ai.operation.name": "retrieval",
                    "segagent.case_id": state["case_id"],
                },
            ):
                _, observation = self.protocol_tool.run(query)
        except Exception as exc:
            emit_live(
                event(
                    "phase_failed",
                    phase="lookup_protocol",
                    label="Protocol retrieval failed",
                    message=str(exc),
                )
            )
            raise
        return {
            "observations": [observation.model_dump(mode="json")],
            "events": [
                event("observation", observation=observation.model_dump(mode="json")),
                event(
                    "phase_completed",
                    phase="lookup_protocol",
                    label="Protocol retrieval completed",
                    matched=bool(observation.data.get("matched")),
                ),
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
        stage = (
            decision.workflow_stage.value if decision.workflow_stage is not None else ""
        )
        if not stage:
            override = forced_structures(
                state, self._protocol_oars(state), state.get("observations") or []
            )
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
        if stage in {
            STAGE_TUMOR_BLIND,
            STAGE_TUMOR_NEIGHBOURS,
            STAGE_TUMOR_REFINE,
            STAGE_OAR_FULL,
        }:
            # These strings were selected by the deterministic FSM. In
            # particular, an enriched tumor prompt must not be combined again
            # with the original description by generic GTV prompt building.
            prompts = [clean_prompt(item) for item in decision.structures if clean_prompt(item)]
        elif stage == STAGE_TUMOR_INITIAL:
            prompts = [clean_prompt(item) for item in decision.structures if clean_prompt(item)]
        else:
            prompts = build_prompts(task_intent, decision.structures) or list(
                decision.structures
            )
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
        # Neighbouring organs contoured to locate a target are organs, not the
        # target: tagging them GTV would flag them all as unvalidated tumors.
        if stage == STAGE_TUMOR_NEIGHBOURS:
            target_type = TargetType.ORGAN
        request = SegmentRequest(
            case_id=state["case_id"],
            structures=prompts,
            purpose=state["question"],
            target_type=target_type,
        )
        emit_live(
            event(
                "phase_started",
                phase="segment",
                label=STAGE_LABELS.get(stage, "Running batched segmentation"),
                structures=request.structures,
                structure_count=len(request.structures),
                workflow_stage=stage or None,
            )
        )
        emit_live(
            event(
                "tool_started",
                tool="segment",
                structures=request.structures,
                stage=STAGE_LABELS.get(stage) if stage else None,
            )
        )
        try:
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
        except Exception as exc:
            emit_live(
                event(
                    "phase_failed",
                    phase="segment",
                    label="Segmentation failed",
                    message=str(exc),
                    workflow_stage=stage or None,
                )
            )
            raise
        # Masks stream straight to the viewer as artifacts; there is no blocking
        # review. The target_type rides on each artifact so the UI can flag
        # tumor targets (GTV) as unvalidated and requiring clinician review.
        artifact_events = [
            event("artifact", artifact=item.mask.model_dump(mode="json"))
            for item in result.measurements
            if item.mask is not None
        ]
        # Record what just completed, so the next planning step advances the
        # workflow instead of repeating this stage.
        answers = dict(state.get("answers") or {})
        produced = any(item.mask is not None for item in result.measurements)
        if produced:
            marker = STAGE_MARKERS.get(stage)
            if marker:
                answers[marker] = True
            # Any tumor contour, however it was prompted, becomes something the
            # user is asked to confirm before the run builds on it.
            if target_type == TargetType.GTV and stage in {
                STAGE_TUMOR_BLIND,
                STAGE_TUMOR_INITIAL,
                "",
            }:
                answers["gtv_first_mask"] = True
        return {
            "observations": [observation.model_dump(mode="json")],
            "seen_signatures": [signature],
            "answers": answers,
            "events": [
                event("observation", observation=observation.model_dump(mode="json")),
                *artifact_events,
                event(
                    "phase_completed",
                    phase="segment",
                    label=STAGE_LABELS.get(stage, "Batched segmentation completed"),
                    structure_count=len(request.structures),
                    produced_mask_count=sum(
                        item.mask is not None for item in result.measurements
                    ),
                    workflow_stage=stage or None,
                ),
            ],
        }

    def _finish(self, state: AgentState) -> dict:
        decision = PlannerDecision.model_validate(state["current_decision"])
        emit_live(
            event(
                "phase_started",
                phase="finalize",
                label="Checking the final answer against recorded evidence",
            )
        )
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
                event(
                    "phase_completed",
                    phase="finalize",
                    label="Evidence-grounded answer ready",
                ),
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

    def _validate_message_context(
        self,
        case_id: str,
        context: MessageArtifactContext | dict | None,
    ) -> MessageArtifactContext:
        parsed = MessageArtifactContext.model_validate(context or {})
        case = self.store.get_case(case_id)
        by_id = {item.artifact_id: item for item in case.artifacts}
        all_ids = [
            *parsed.referenced_artifact_ids,
            *parsed.visible_mask_ids,
            *([parsed.selected_mask_id] if parsed.selected_mask_id else []),
        ]
        missing = [artifact_id for artifact_id in all_ids if artifact_id not in by_id]
        if missing:
            raise ValueError(
                f"artifact does not belong to this case: {missing[0]}"
            )
        for artifact_id in [
            *parsed.visible_mask_ids,
            *([parsed.selected_mask_id] if parsed.selected_mask_id else []),
        ]:
            if by_id[artifact_id].kind not in {"mask", "contour"}:
                raise ValueError(f"artifact is not a mask or contour: {artifact_id}")
        return parsed

    def send(
        self,
        case_id: str,
        text: str,
        message_context: MessageArtifactContext | dict | None = None,
    ) -> tuple[str, Iterator[RunEvent]]:
        """Handle one user message.

        If the agent is waiting on a question for this case, the message is the
        reply and continues that turn; otherwise it opens a new turn. The client
        therefore only ever needs one "send" action.
        """
        text = " ".join(text.strip().split())
        if not text or len(text) > 4000:
            raise ValueError("message must contain 1-4000 characters")
        context = self._validate_message_context(case_id, message_context)
        conversation = self.store.conversation_for_case(case_id)
        if conversation.pending_run_id:
            try:
                return self.reply(conversation.pending_run_id, text, context)
            except (ValueError, FileNotFoundError):
                # The pending turn is gone or no longer waiting; start a new one.
                conversation.pending_run_id = None
                conversation.pending_question = None
                conversation.pending_gate = None
                conversation.pending_context = {}
                self.store.save_conversation(conversation)

        run = self.store.create_run(case_id, text, conversation.conversation_id)
        self.store.append_message(
            conversation.conversation_id,
            case_id,
            ChatRole.USER,
            text,
            run.run_id,
            metadata={"artifact_context": context.model_dump(mode="json")},
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
                event(
                    "run_started",
                    question=text,
                    message_context=context.model_dump(mode="json"),
                    artifact_inventory=self.store.artifact_inventory(case_id),
                    **intent_payload,
                ),
            )
            initial: AgentState = {
                "case_id": case_id,
                "run_id": run.run_id,
                "conversation_id": conversation.conversation_id,
                "question": text,
                "history": history,
                "intent": intent_payload.get("intent", {}),
                "has_contours": has_any_mask,
                "message_context": context.model_dump(mode="json"),
                "artifact_inventory": self.store.artifact_inventory(case_id),
                "step": 0,
                "events": [],
                **new_turn(),
            }
            yield from self._execute(run.run_id, conversation.conversation_id, initial)

        return run.run_id, stream()

    def reply(
        self,
        run_id: str,
        text: str,
        message_context: MessageArtifactContext | dict | None = None,
    ) -> tuple[str, Iterator[RunEvent]]:
        """Answer the agent's pending question and resume the same turn."""
        text = " ".join(text.strip().split())
        if not text or len(text) > 4000:
            raise ValueError("reply must contain 1-4000 characters")
        run = self.store.get_run(run_id)
        if run.status != "waiting_reply":
            raise ValueError("this turn is not waiting for a reply")
        context = self._validate_message_context(run.case_id, message_context)
        conversation_id = run.conversation_id or ""
        if conversation_id:
            self.store.append_message(
                conversation_id,
                run.case_id,
                ChatRole.USER,
                text,
                run_id,
                metadata={"artifact_context": context.model_dump(mode="json")},
            )

        def stream() -> Iterator[RunEvent]:
            if conversation_id:
                conversation = self.store.get_conversation(conversation_id)
                conversation.pending_question = None
                conversation.pending_run_id = None
                conversation.pending_gate = None
                conversation.pending_context = {}
                self.store.save_conversation(conversation)
            run.status = "running"
            self.store.save_run(run)
            resume_payload: dict = {
                "text": text,
                "message_context": context.model_dump(mode="json"),
            }
            try:
                case = self.store.get_case(run.case_id)
                has_uploaded = bool(case.contours)
                resume_payload["has_contours"] = any(
                    item.kind in {"mask", "contour"} for item in case.artifacts
                )
                if self.intent_classifier is not None:
                    resume_payload["intent"] = self.intent_classifier.classify(
                        text, has_contours=has_uploaded
                    ).model_dump(mode="json")
            except Exception:
                pass
            yield from self._execute(
                run_id,
                conversation_id,
                Command(resume=resume_payload),
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
            for packet in self.graph.graph.stream(
                value,
                config=config,
                stream_mode=["updates", "custom"],
            ):
                if (
                    isinstance(packet, tuple)
                    and len(packet) == 2
                    and packet[0] in {"updates", "custom"}
                ):
                    mode, update = packet
                else:
                    # Compatibility with a graph runtime that returns the
                    # update directly for a single effective stream mode.
                    mode, update = "updates", packet
                if mode == "custom":
                    if isinstance(update, dict) and update.get("type"):
                        yield self._persist(run_id, run.case_id, update)
                    continue
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
            conversation.pending_gate = str(payload.get("gate") or "") or None
            conversation.pending_context = dict(
                payload.get("pending_interaction") or {}
            )
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
