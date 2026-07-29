from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Intent(str, Enum):
    ORGAN = "organ"
    OAR = "oar"
    GTV = "gtv"
    PROTOCOL = "protocol"
    QUESTION = "question"


class TargetType(str, Enum):
    ORGAN = "organ"
    OAR = "oar"
    GTV = "gtv"
    CONTOUR = "contour"
    UNKNOWN = "unknown"


class PlannerAction(str, Enum):
    LOOKUP_PROTOCOL = "lookup_protocol"
    SEGMENT = "segment"
    ASK_USER = "ask_user"
    FINAL = "final"


class ChatRole(str, Enum):
    USER = "user"
    AGENT = "agent"


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
    target_type: "TargetType" = TargetType.UNKNOWN

    @field_validator("structures")
    @classmethod
    def normalize_structures(cls, value: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for item in value:
            normalized = " ".join(item.strip().split())
            # 400 chars: anatomical labels are short, but a GTV prompt may be a
            # full clinical sentence, which is what VoxTell's fusion needs.
            if not normalized or len(normalized) > 400:
                raise ValueError("each structure/prompt must be a non-empty string <= 400 chars")
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


class ToolObservation(BaseModel):
    observation_id: str
    tool: Literal["lookup_protocol", "segment", "evidence_critic"]
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
    # Set when a deterministic pre-model gate produced this decision, so the
    # gate is recorded as fired and cannot ask the same question twice.
    gate: str | None = None

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


class RunEvent(BaseModel):
    event_id: str
    run_id: str
    case_id: str
    sequence: int
    type: Literal[
        "run_started",
        "planner_decision",
        "thinking",
        "tool_started",
        "observation",
        "artifact",
        "question_pending",
        "reply_recorded",
        "answer",
        "error",
        "run_completed",
    ]
    timestamp: datetime = Field(default_factory=utc_now)
    payload: dict[str, Any] = Field(default_factory=dict)


class RunRecord(BaseModel):
    """One conversational turn: a user message and the work it triggered."""

    run_id: str
    case_id: str
    question: str
    status: Literal["created", "running", "waiting_reply", "completed", "failed"]
    conversation_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    event_count: int = 0
    final_answer: str | None = None


class ChatMessage(BaseModel):
    """A durable conversation turn, independent of the run that produced it."""

    message_id: str
    conversation_id: str
    case_id: str
    role: ChatRole
    text: str
    run_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationRecord(BaseModel):
    """Per-case dialogue state.

    One conversation per case gives the agent context across turns, and its id
    is the LangGraph thread id, so accumulated evidence (prior masks and
    measurements) survives from one question to the next.
    """

    conversation_id: str
    case_id: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    turn_count: int = 0
    # Set while the agent has asked something and is waiting for the user.
    pending_question: str | None = None
    pending_run_id: str | None = None


class AgentCardSkill(BaseModel):
    id: str
    name: str
    description: str
    tags: list[str]
    examples: list[str]


class TaskIntent(BaseModel):
    """Result of intent recognition: which task, and the material it needs."""

    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    targets: list[str] = Field(default_factory=list)
    site: str | None = None
    source_description: str | None = None
    rationale: str = ""


class EvidenceRecord(BaseModel):
    """A typed unit of evidence used only to ground the final answer.

    ``values`` holds the numeric quantities a tool actually emitted; the
    verifier grounds answer numbers against these (and citations against
    ``citations``), never against a serialized data blob.
    """

    model_config = ConfigDict(protected_namespaces=())

    record_id: str
    tool: str
    summary: str = ""
    target_type: TargetType = TargetType.UNKNOWN
    model_name: str | None = None
    model_version: str | None = None
    values: dict[str, float] = Field(default_factory=dict)
    citations: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    derived_from: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    data: dict[str, Any] = Field(default_factory=dict)
