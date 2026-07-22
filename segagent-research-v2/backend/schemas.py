from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Intent(str, Enum):
    SEGMENT = "segment"
    QC = "qc"
    PROTOCOL = "protocol"
    QUESTION = "question"


class PlannerAction(str, Enum):
    LOOKUP_PROTOCOL = "lookup_protocol"
    SEGMENT = "segment"
    RUN_QC = "run_qc"
    ASK_USER = "ask_user"
    FINAL = "final"


class ApprovalKind(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    FEEDBACK = "feedback"


class ArtifactRef(BaseModel):
    artifact_id: str
    case_id: str
    kind: Literal["image", "contour", "mask", "overlay", "report", "export"]
    label: str
    media_type: str
    sha256: str
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CaseRecord(BaseModel):
    case_id: str
    created_at: datetime = Field(default_factory=utc_now)
    source_name: str
    image: ArtifactRef
    contours: list[ArtifactRef] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProtocolMatch(BaseModel):
    protocol_id: str
    site: str
    oars: list[str]
    score: float
    matched_by: Literal["keyword", "hybrid"]
    source: str
    source_version: str | None = None
    citations: list[str] = Field(default_factory=list)


class RetrievalHit(BaseModel):
    text: str
    source: str
    section: str | None = None
    score: float
    citation: str


class SegmentRequest(BaseModel):
    case_id: str
    structures: list[str] = Field(min_length=1, max_length=64)
    purpose: str = "answer the user's question"

    @field_validator("structures")
    @classmethod
    def normalize_structures(cls, value: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for item in value:
            normalized = " ".join(item.strip().split())
            if not normalized or len(normalized) > 160:
                raise ValueError("each structure must be a non-empty name <= 160 chars")
            key = normalized.casefold()
            if key not in seen:
                seen.add(key)
                result.append(normalized)
        if not result:
            raise ValueError("at least one structure is required")
        return result


class StructureMeasurement(BaseModel):
    structure: str
    found: bool
    voxels: int = 0
    volume_ml: float | None = None
    mean_intensity: float | None = None
    bbox_voxels: list[list[int]] | None = None
    centroid_voxels: list[float] | None = None
    mask: ArtifactRef | None = None
    overlays: list[ArtifactRef] = Field(default_factory=list)


class SegmentResult(BaseModel):
    request: SegmentRequest
    model_name: str
    model_version: str
    elapsed_ms: float
    measurements: list[StructureMeasurement]
    warnings: list[str] = Field(default_factory=list)


class Finding(BaseModel):
    check: str
    severity: Literal["ok", "warn", "error"]
    message: str
    metrics: dict[str, Any] = Field(default_factory=dict)


class QCOrganResult(BaseModel):
    organ: str
    status: Literal["ok", "warn", "error"]
    volume_ml: float | None = None
    expert_dice: float | None = None
    findings: list[Finding] = Field(default_factory=list)
    expert_mask: ArtifactRef | None = None


class QCReport(BaseModel):
    case_id: str
    organs: list[QCOrganResult]
    summary: dict[str, int]
    warnings: list[str] = Field(default_factory=list)


class ToolObservation(BaseModel):
    observation_id: str
    tool: Literal["lookup_protocol", "segment", "run_qc", "evidence_critic"]
    summary: str
    data: dict[str, Any]
    evidence_ids: list[str] = Field(default_factory=list)


class PlannerDecision(BaseModel):
    action: PlannerAction
    rationale_summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    structures: list[str] = Field(default_factory=list)
    site_query: str | None = None
    user_message: str | None = None
    final_answer: str | None = None

    @field_validator("structures")
    @classmethod
    def strip_structure_names(cls, value: list[str]) -> list[str]:
        return [" ".join(item.strip().split()) for item in value if item.strip()]

    @model_validator(mode="after")
    def validate_action_payload(self) -> "PlannerDecision":
        if self.action == PlannerAction.SEGMENT and not self.structures:
            raise ValueError("segment decisions require at least one structure")
        if self.action == PlannerAction.LOOKUP_PROTOCOL and not self.site_query:
            raise ValueError("lookup_protocol decisions require site_query")
        if self.action == PlannerAction.ASK_USER and not self.user_message:
            raise ValueError("ask_user decisions require user_message")
        if self.action == PlannerAction.FINAL and not self.final_answer:
            raise ValueError("final decisions require final_answer")
        return self


class ApprovalDecision(BaseModel):
    decision: ApprovalKind
    feedback: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def validate_feedback(self) -> "ApprovalDecision":
        if self.decision == ApprovalKind.FEEDBACK and not (self.feedback or "").strip():
            raise ValueError("feedback text is required when decision is feedback")
        return self


class RunEvent(BaseModel):
    event_id: str
    run_id: str
    case_id: str
    sequence: int
    type: Literal[
        "run_started",
        "planner_decision",
        "tool_started",
        "observation",
        "artifact",
        "approval_required",
        "approval_recorded",
        "answer",
        "error",
        "run_completed",
    ]
    timestamp: datetime = Field(default_factory=utc_now)
    payload: dict[str, Any] = Field(default_factory=dict)


class RunRecord(BaseModel):
    run_id: str
    case_id: str
    question: str
    status: Literal["created", "running", "waiting_approval", "completed", "failed"]
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    event_count: int = 0
    final_answer: str | None = None


class AgentCardSkill(BaseModel):
    id: str
    name: str
    description: str
    tags: list[str]
    examples: list[str]
