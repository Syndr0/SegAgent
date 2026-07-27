from __future__ import annotations

import operator
import sqlite3
import uuid
from typing import Annotated, Iterator, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from .config import Settings
from .observability import Tracing
from .planner import Planner
from .schemas import (
    Intent,
    PlannerAction,
    PlannerDecision,
    RunEvent,
    SegmentRequest,
    TaskIntent,
    ToolObservation,
)
from .guards import request_signature
from .intent import IntentClassifier
from .prompts import build_prompts, target_type_for
from .storage import ResearchStore
from .tools import EvidenceCritic, ProtocolLookupTool, SegmentationTool


class AgentState(TypedDict, total=False):
    case_id: str
    run_id: str
    question: str
    intent: dict
    step: int
    observations: Annotated[list[dict], operator.add]
    decisions: Annotated[list[dict], operator.add]
    events: Annotated[list[dict], operator.add]
    current_decision: dict
    final_answer: str | None
    critic_findings: list[dict]
    seen_signatures: Annotated[list[str], operator.add]


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
    ):
        self.settings = settings
        self.planner = planner
        self.protocol_tool = protocol_tool
        self.segmentation_tool = segmentation_tool
        self.critic = critic
        self.tracing = tracing or Tracing()
        self._checkpoint_connection = sqlite3.connect(
            str(settings.checkpoint_db), check_same_thread=False
        )
        self.checkpointer = SqliteSaver(self._checkpoint_connection)
        self.graph = self._build().compile(checkpointer=self.checkpointer)

    @staticmethod
    def _observations(state: AgentState) -> list[ToolObservation]:
        return [ToolObservation.model_validate(item) for item in state.get("observations", [])]

    @staticmethod
    def _oar_site_gate(state: AgentState) -> PlannerDecision | None:
        """Deterministic, pre-LLM gate for organs-at-risk requests.

        An OAR request that names neither a treatment site nor explicit
        structures must ask the user which site to contour, rather than letting
        the planner guess an OAR list. This runs before the planner model.
        """
        intent_data = state.get("intent") or {}
        if intent_data.get("intent") != Intent.OAR.value:
            return None
        if (intent_data.get("site") or "").strip():
            return None
        if intent_data.get("targets"):
            return None
        return PlannerDecision(
            action=PlannerAction.ASK_USER,
            rationale_summary="OAR request has no treatment site; asking before segmenting.",
            confidence=1.0,
            user_message=(
                "Which treatment site's organs at risk should I contour? "
                "For example: head and neck, thorax, abdomen, or pelvis."
            ),
        )

    def _build(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("plan", self._plan)
        graph.add_node("lookup_protocol", self._lookup_protocol)
        graph.add_node("segment", self._segment)
        graph.add_node("finish", self._finish)
        graph.add_edge(START, "plan")
        graph.add_conditional_edges(
            "plan",
            self._route,
            {
                PlannerAction.LOOKUP_PROTOCOL.value: "lookup_protocol",
                PlannerAction.SEGMENT.value: "segment",
                PlannerAction.ASK_USER.value: "finish",
                PlannerAction.FINAL.value: "finish",
            },
        )
        graph.add_edge("lookup_protocol", "plan")
        graph.add_edge("segment", "plan")
        graph.add_edge("finish", END)
        return graph

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
            gate = self._oar_site_gate(state) if step == 1 else None
            if gate is not None:
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
                decision = self.planner.decide(
                    state["case_id"], state["question"], step, observations
                )
        return {
            "step": step,
            "current_decision": decision.model_dump(mode="json"),
            "decisions": [decision.model_dump(mode="json")],
            "events": [
                event(
                    "planner_decision",
                    step=step,
                    action=decision.action.value,
                    rationale_summary=decision.rationale_summary,
                    structures=decision.structures,
                    site_query=decision.site_query,
                )
            ],
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
        if decision.action == PlannerAction.ASK_USER:
            answer = decision.user_message or "Please provide more information."
        else:
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

    def start(self, case_id: str, question: str) -> tuple[str, Iterator[RunEvent]]:
        question = " ".join(question.strip().split())
        if not question or len(question) > 4000:
            raise ValueError("question must contain 1-4000 characters")
        run = self.store.create_run(case_id, question)

        intent_payload: dict = {}
        if self.intent_classifier is not None:
            try:
                task_intent = self.intent_classifier.classify(question)
                intent_payload = {"intent": task_intent.model_dump(mode="json")}
            except Exception:
                intent_payload = {}

        def stream() -> Iterator[RunEvent]:
            run.status = "running"
            self.store.save_run(run)
            yield self._persist(
                run.run_id,
                case_id,
                event("run_started", question=question, **intent_payload),
            )
            initial: AgentState = {
                "case_id": case_id,
                "run_id": run.run_id,
                "question": question,
                "intent": intent_payload.get("intent", {}),
                "step": 0,
                "observations": [],
                "decisions": [],
                "events": [],
            }
            yield from self._execute(run.run_id, initial)

        return run.run_id, stream()

    def _execute(self, run_id: str, initial: AgentState) -> Iterator[RunEvent]:
        run = self.store.get_run(run_id)
        config = {"configurable": {"thread_id": run_id}}
        try:
            for update in self.graph.graph.stream(initial, config=config, stream_mode="updates"):
                for node_update in update.values():
                    if not isinstance(node_update, dict):
                        continue
                    for raw in node_update.get("events", []):
                        persisted = self._persist(run_id, run.case_id, raw)
                        yield persisted
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
