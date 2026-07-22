from __future__ import annotations

import operator
import sqlite3
import uuid
from typing import Annotated, Any, Iterator, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .config import Settings
from .observability import Tracing
from .planner import Planner
from .schemas import (
    ApprovalDecision,
    ApprovalKind,
    PlannerAction,
    PlannerDecision,
    RunEvent,
    SegmentRequest,
    ToolObservation,
)
from .storage import ResearchStore
from .tools import ContourQCTool, EvidenceCritic, ProtocolLookupTool, SegmentationTool


class AgentState(TypedDict, total=False):
    case_id: str
    run_id: str
    question: str
    step: int
    observations: Annotated[list[dict], operator.add]
    decisions: Annotated[list[dict], operator.add]
    events: Annotated[list[dict], operator.add]
    current_decision: dict
    pending_review: dict | None
    approval: dict | None
    final_answer: str | None
    critic_findings: list[dict]


def event(event_type: str, **payload) -> dict:
    return {"type": event_type, "payload": payload}


class SegAgentGraph:
    def __init__(
        self,
        settings: Settings,
        planner: Planner,
        protocol_tool: ProtocolLookupTool,
        segmentation_tool: SegmentationTool,
        qc_tool: ContourQCTool,
        critic: EvidenceCritic,
        tracing: Tracing | None = None,
    ):
        self.settings = settings
        self.planner = planner
        self.protocol_tool = protocol_tool
        self.segmentation_tool = segmentation_tool
        self.qc_tool = qc_tool
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

    def _build(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("plan", self._plan)
        graph.add_node("lookup_protocol", self._lookup_protocol)
        graph.add_node("segment", self._segment)
        graph.add_node("review", self._review)
        graph.add_node("run_qc", self._run_qc)
        graph.add_node("finish", self._finish)
        graph.add_edge(START, "plan")
        graph.add_conditional_edges(
            "plan",
            self._route,
            {
                PlannerAction.LOOKUP_PROTOCOL.value: "lookup_protocol",
                PlannerAction.SEGMENT.value: "segment",
                PlannerAction.RUN_QC.value: "run_qc",
                PlannerAction.ASK_USER.value: "finish",
                PlannerAction.FINAL.value: "finish",
            },
        )
        graph.add_edge("lookup_protocol", "plan")
        graph.add_edge("segment", "review")
        graph.add_edge("review", "plan")
        graph.add_edge("run_qc", "plan")
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
            if step > self.settings.max_steps:
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
                    confidence=decision.confidence,
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
        request = SegmentRequest(
            case_id=state["case_id"],
            structures=decision.structures,
            purpose=state["question"],
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
        mask_artifacts = [
            item.mask.model_dump(mode="json")
            for item in result.measurements
            if item.mask is not None
        ]
        artifact_events = [event("artifact", artifact=item) for item in mask_artifacts]
        pending = {
            "tool": "segment",
            "evidence_ids": observation.evidence_ids,
            "summary": observation.summary,
            "artifacts": mask_artifacts,
        }
        return {
            "observations": [observation.model_dump(mode="json")],
            "pending_review": pending if mask_artifacts else None,
            "events": [
                event("tool_started", tool="segment", structures=request.structures),
                event("observation", observation=observation.model_dump(mode="json")),
                *artifact_events,
            ],
        }

    def _review(self, state: AgentState) -> dict:
        pending = state.get("pending_review")
        if not pending:
            return {}
        if not self.settings.require_mask_approval:
            approval = ApprovalDecision(decision=ApprovalKind.APPROVE)
        else:
            response = interrupt(
                {
                    "type": "approval_required",
                    "case_id": state["case_id"],
                    "run_id": state["run_id"],
                    "message": "Review the generated masks before the agent continues.",
                    **pending,
                    "allowed_decisions": [item.value for item in ApprovalKind],
                }
            )
            approval = ApprovalDecision.model_validate(response)
        additions: list[dict] = []
        if approval.decision != ApprovalKind.APPROVE:
            text = approval.feedback or "The reviewer did not approve the mask evidence."
            observation = ToolObservation(
                observation_id=f"observation_{uuid.uuid4().hex[:16]}",
                tool="evidence_critic",
                summary=f"Human review {approval.decision.value}: {text}",
                data={"approval": approval.model_dump(mode="json")},
                evidence_ids=list(pending.get("evidence_ids", [])),
            )
            additions.append(observation.model_dump(mode="json"))
        return {
            "approval": approval.model_dump(mode="json"),
            "pending_review": None,
            "observations": additions,
            "events": [event("approval_recorded", approval=approval.model_dump(mode="json"))],
        }

    def _run_qc(self, state: AgentState) -> dict:
        with self.tracing.span(
            "execute_tool",
            **{
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "run_qc",
                "segagent.case_id": state["case_id"],
            },
        ):
            report, observation = self.qc_tool.run(state["case_id"])
        artifact_events = [
            event("artifact", artifact=row.expert_mask.model_dump(mode="json"))
            for row in report.organs
            if row.expert_mask is not None
        ]
        return {
            "observations": [observation.model_dump(mode="json")],
            "events": [
                event("tool_started", tool="run_qc"),
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
    """Persists graph updates as typed events and exposes pause/resume streams."""

    def __init__(self, store: ResearchStore, graph: SegAgentGraph):
        self.store = store
        self.graph = graph

    def start(self, case_id: str, question: str) -> tuple[str, Iterator[RunEvent]]:
        question = " ".join(question.strip().split())
        if not question or len(question) > 4000:
            raise ValueError("question must contain 1-4000 characters")
        run = self.store.create_run(case_id, question)

        def stream() -> Iterator[RunEvent]:
            run.status = "running"
            self.store.save_run(run)
            yield self._persist(
                run.run_id,
                case_id,
                event("run_started", question=question),
            )
            initial: AgentState = {
                "case_id": case_id,
                "run_id": run.run_id,
                "question": question,
                "step": 0,
                "observations": [],
                "decisions": [],
                "events": [],
                "pending_review": None,
            }
            yield from self._execute(run.run_id, initial)

        return run.run_id, stream()

    def resume(self, run_id: str, decision: ApprovalDecision) -> Iterator[RunEvent]:
        run = self.store.get_run(run_id)
        if run.status != "waiting_approval":
            raise ValueError("run is not waiting for approval")
        run.status = "running"
        self.store.save_run(run)
        return self._execute(run_id, Command(resume=decision.model_dump(mode="json")))

    def _execute(self, run_id: str, value: AgentState | Command) -> Iterator[RunEvent]:
        run = self.store.get_run(run_id)
        config = {"configurable": {"thread_id": run_id}}
        try:
            for update in self.graph.graph.stream(value, config=config, stream_mode="updates"):
                if "__interrupt__" in update:
                    run = self.store.get_run(run_id)
                    run.status = "waiting_approval"
                    self.store.save_run(run)
                    for item in update["__interrupt__"]:
                        payload = getattr(item, "value", item)
                        yield self._persist(
                            run_id,
                            run.case_id,
                            event("approval_required", **dict(payload)),
                        )
                    return
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

