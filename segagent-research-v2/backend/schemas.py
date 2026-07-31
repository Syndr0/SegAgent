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
    QC = "qc"
    REPORT = "report"
    PROTOCOL = "protocol"
    VQA = "vqa"
    MASK_QUESTION = "mask_question"
    AMBIGUOUS = "ambiguous"
    MISSING_INPUT = "missing_input"
    # Kept for stored checkpoints and third-party planners created before the
    # intent split. New classification uses one of the three explicit values
    # above instead of overloading QUESTION.
    QUESTION = "question"


class TargetType(str, Enum):
    ORGAN = "organ"
    OAR = "oar"
    GTV = "gtv"
    CONTOUR = "contour"
    UNKNOWN = "unknown"


class PromptCategory(str, Enum):
    """The three text regimes accepted by the segmentation model."""

    ANATOMICAL_CONCEPT = "anatomical_concept"
    PATHOLOGICAL_STRUCTURE = "pathological_structure"
    CLINICAL_DESCRIPTION = "clinical_description"


class PlannerAction(str, Enum):
    LOOKUP_PROTOCOL = "lookup_protocol"
    SEGMENT = "segment"
    INSPECT_MASK = "inspect_mask"
    RUN_QC = "run_qc"
    ANSWER_QUESTION = "answer_question"
    ASK_USER = "ask_user"
    FINAL = "final"


class ChatRole(str, Enum):
    USER = "user"
    AGENT = "agent"


class DialogueStage(str, Enum):
    """Explicit workflow stage; never inferred from free-form planner prose."""

    TUMOR_INITIAL = "tumor_initial"
    TUMOR_BLIND = "tumor_blind"
    TUMOR_NEIGHBOURS = "tumor_neighbours"
    TUMOR_REFINE = "tumor_refine"
    OAR_SITE = "oar_site"
    OAR_PROTOCOL = "oar_protocol"
    OAR_FULL = "oar_full"


class MessageArtifactContext(BaseModel):
    """Viewer context attached to one user turn.

    IDs are validated against the case by ``WorkflowService``. Keeping IDs in
    the message contract avoids resolving "this mask" from label substrings or
    from whichever artifact happened to be created most recently.
    """

    referenced_artifact_ids: list[str] = Field(default_factory=list, max_length=64)
    visible_mask_ids: list[str] = Field(default_factory=list, max_length=64)
    selected_mask_id: str | None = None

    @field_validator("referenced_artifact_ids", "visible_mask_ids")
    @classmethod
    def dedupe_artifact_ids(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(item.strip() for item in value if item.strip()))


class PendingInteraction(BaseModel):
    """Typed question suspended inside a turn."""

    gate: str
    question: str
    stage: DialogueStage | None = None
    proposed_site: str | None = None


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


class PromptNormalization(BaseModel):
    """One source prompt classified and normalized for VoxTell."""

    source_index: int = Field(ge=0)
    source_text: str
    category: PromptCategory
    voxtell_prompt: str
    confidence: float = Field(ge=0.0, le=1.0)
    normalized_by: Literal[
        "llm",
        "deterministic_fallback",
        "trusted_workflow",
    ] = "llm"
    fallback_reason: str | None = None
    # True only for a deliberately broad, user-confirmed first search. The
    # generated mask remains a hypothesis and must never imply a diagnosis.
    exploratory: bool = False

    @field_validator("source_text", "voxtell_prompt")
    @classmethod
    def validate_prompt_text(cls, value: str) -> str:
        normalized = " ".join(str(value or "").strip().split())
        if not normalized or len(normalized) > 400:
            raise ValueError("prompt text must be a non-empty string <= 400 chars")
        return normalized


class PromptNormalizationBatch(BaseModel):
    items: list[PromptNormalization] = Field(min_length=1, max_length=64)
    # Model thinking is displayed for transparency but never used as evidence.
    thinking: str | None = None

    @model_validator(mode="after")
    def source_indexes_are_sequential(self) -> "PromptNormalizationBatch":
        if [item.source_index for item in self.items] != list(
            range(len(self.items))
        ):
            raise ValueError(
                "prompt normalizations must use every source index once in order"
            )
        return self


class SegmentRequest(BaseModel):
    case_id: str
    structures: list[str] = Field(min_length=1, max_length=64)
    purpose: str = "answer the user's question"
    target_type: "TargetType" = TargetType.UNKNOWN
    prompt_normalizations: list[PromptNormalization] = Field(default_factory=list)

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

    @model_validator(mode="after")
    def prompt_normalizations_match_structures(self) -> "SegmentRequest":
        if not self.prompt_normalizations:
            return self
        if len(self.prompt_normalizations) != len(self.structures):
            raise ValueError("prompt_normalizations must align one-to-one with structures")
        indexes = [item.source_index for item in self.prompt_normalizations]
        if indexes != list(range(len(self.prompt_normalizations))):
            raise ValueError(
                "prompt_normalizations must use sequential source indexes"
            )
        for structure, item in zip(self.structures, self.prompt_normalizations):
            if structure.casefold() != item.voxtell_prompt.casefold():
                raise ValueError(
                    "each structure must equal its normalized VoxTell prompt"
                )
        if self.target_type in {TargetType.ORGAN, TargetType.OAR} and any(
            item.category != PromptCategory.ANATOMICAL_CONCEPT
            for item in self.prompt_normalizations
        ):
            raise ValueError(
                "organ/OAR requests require anatomical prompt categories"
            )
        if self.target_type == TargetType.GTV and any(
            item.category == PromptCategory.ANATOMICAL_CONCEPT
            for item in self.prompt_normalizations
        ):
            raise ValueError(
                "GTV requests require pathological or clinical prompt categories"
            )
        return self


class StructureMeasurement(BaseModel):
    structure: str
    found: bool
    voxels: int = 0
    volume_ml: float | None = None
    mean_intensity: float | None = None
    bbox_voxels: list[list[int]] | None = None
    centroid_voxels: list[float] | None = None
    centroid_mm: list[float] | None = None
    mask: ArtifactRef | None = None
    overlays: list[ArtifactRef] = Field(default_factory=list)


class SegmentResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

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


class MetricEvidence(BaseModel):
    """One measured quantity with enough context to verify a later claim.

    A bare float is ambiguous: 10 mL cannot ground a claim of HD95=10 mm or
    "10 structures".  Metric, unit, provenance, and source artifacts therefore
    travel with every QC/report value.
    """

    evidence_id: str
    structure: str | None = None
    canonical_roi: str | None = None
    metric: str
    value: float
    unit: str
    source: str
    artifact_ids: list[str] = Field(default_factory=list)


class QCComparison(BaseModel):
    """An uploaded organ contour measured against an independent model contour."""

    model_config = ConfigDict(protected_namespaces=())

    structure: str
    canonical_roi: str | None = None
    contour_artifact_id: str | None = None
    model_mask: ArtifactRef | None = None
    difference_overlays: list[ArtifactRef] = Field(default_factory=list)
    model_name: str | None = None
    model_version: str | None = None
    dice: float | None = None
    contour_volume_ml: float | None = None
    model_volume_ml: float | None = None
    volume_difference_pct: float | None = None
    mean_surface_mm: float | None = None
    hd95_mm: float | None = None
    status: Literal["ok", "warn", "error"] = "ok"
    note: str = ""


class QCTargetNote(BaseModel):
    """A target volume (GTV/CTV/PTV), described but never model-checked.

    An independent segmentation model has no way to validate a clinical target,
    so targets are taken as given and only described: where they sit and which
    contoured structures they lie near.
    """

    structure: str
    canonical_roi: str | None = None
    target_class: str | None = None
    assessment_status: Literal["not_automatically_assessed"] = "not_automatically_assessed"
    artifact_id: str | None = None
    volume_ml: float | None = None
    centroid_voxels: list[float] | None = None
    centroid_mm: list[float] | None = None
    near: list[str] = Field(default_factory=list)
    near_surface_mm: dict[str, float] = Field(default_factory=dict)


class QCAuxiliaryNote(BaseModel):
    """A non-anatomical or derived ROI excluded from organ-model comparison."""

    structure: str
    canonical_roi: str
    auxiliary_class: str
    classification_source: str
    artifact_id: str | None = None
    volume_ml: float | None = None
    note: str = (
        "Identified as an auxiliary planning structure and excluded from "
        "anatomy-model comparison."
    )


class QCUnclassifiedNote(BaseModel):
    """An ROI whose role cannot be established safely from metadata or name."""

    structure: str
    canonical_roi: str
    artifact_id: str | None = None
    volume_ml: float | None = None
    note: str = (
        "Role is unclassified; excluded from anatomy-model comparison until "
        "an explicit organ, target, or auxiliary mapping is supplied."
    )


class ProtocolApplicability(BaseModel):
    """Whether a protocol list can be applied to this case and on what basis."""

    status: Literal["applicable", "partially_applicable", "not_assessed"]
    reason: str
    source_status: Literal["institution_approved", "research_seed", "unknown"] = "unknown"
    excluded: list[str] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)


class QCCompleteness(BaseModel):
    """Which protocol structures are present, as a set difference — not a judgement."""

    site: str | None = None
    expected: list[str] = Field(default_factory=list)
    present: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    unmatched_contours: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    source: str | None = None
    source_version: str | None = None
    applicability: ProtocolApplicability | None = None


class QCReport(BaseModel):
    case_id: str
    comparisons: list[QCComparison] = Field(default_factory=list)
    targets: list[QCTargetNote] = Field(default_factory=list)
    auxiliaries: list[QCAuxiliaryNote] = Field(default_factory=list)
    unclassified: list[QCUnclassifiedNote] = Field(default_factory=list)
    completeness: QCCompleteness | None = None
    summary: dict[str, int] = Field(default_factory=dict)
    metric_evidence: list[MetricEvidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ReportReadiness(BaseModel):
    """Deterministic precondition check for structure-based reports."""

    scope: Literal["comprehensive", "available", "oar", "target"] = "comprehensive"
    ready: bool
    organ_count: int = 0
    target_count: int = 0
    missing_inputs: list[str] = Field(default_factory=list)
    reason: str


class ReportStructureEvidence(BaseModel):
    """Measured structure admitted to a report, tied to one concrete artifact."""

    structure: str
    canonical_roi: str
    role: Literal["organ", "target"]
    assessment_status: Literal["measured", "not_automatically_assessed"]
    artifact_id: str
    source: str
    volume_ml: float | None = None
    centroid_mm: list[float] | None = None
    near_surface_mm: dict[str, float] = Field(default_factory=dict)


class ReportEvidenceBundle(BaseModel):
    """Deterministic measurements supplied to report prose generation."""

    case_id: str
    readiness: ReportReadiness
    structures: list[ReportStructureEvidence] = Field(default_factory=list)
    metric_evidence: list[MetricEvidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ToolObservation(BaseModel):
    observation_id: str
    tool: Literal[
        "lookup_protocol",
        "segment",
        "inspect_mask",
        "run_qc",
        "build_report_evidence",
        "evidence_critic",
    ]
    summary: str
    data: dict[str, Any]
    evidence_ids: list[str] = Field(default_factory=list)


class PlannerDecision(BaseModel):
    action: PlannerAction
    rationale_summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    structures: list[str] = Field(default_factory=list)
    # Filled by the dedicated prompt normalizer after action selection. Keeping
    # this separate from ``structures`` preserves the original user text and the
    # selected language regime for audit.
    prompt_normalizations: list[PromptNormalization] = Field(default_factory=list)
    site_query: str | None = None
    user_message: str | None = None
    final_answer: str | None = None
    # Set when a deterministic pre-model gate produced this decision, so the
    # gate is recorded as fired and cannot ask the same question twice.
    gate: str | None = None
    # The model's own reasoning, captured from its thinking channel rather than
    # requested inside the JSON. Displayed to the user; never used as evidence.
    thinking: str | None = None
    # Safety-relevant workflow stage, assigned only by deterministic code.
    workflow_stage: DialogueStage | None = None

    @field_validator("structures")
    @classmethod
    def strip_structure_names(cls, value: list[str]) -> list[str]:
        return [" ".join(item.strip().split()) for item in value if item.strip()]

    @model_validator(mode="after")
    def validate_action_payload(self) -> "PlannerDecision":
        if self.action == PlannerAction.SEGMENT and not self.structures:
            raise ValueError("segment decisions require at least one structure")
        if self.action == PlannerAction.INSPECT_MASK and not self.structures:
            raise ValueError("inspect_mask decisions require at least one structure")
        if self.action == PlannerAction.LOOKUP_PROTOCOL and not self.site_query:
            raise ValueError("lookup_protocol decisions require site_query")
        if self.action == PlannerAction.ASK_USER and not self.user_message:
            raise ValueError("ask_user decisions require user_message")
        if self.action == PlannerAction.FINAL and not self.final_answer:
            raise ValueError("final decisions require final_answer")
        return self


class PlannerAnswer(BaseModel):
    """Prose response plus the model's separately displayed thinking channel."""

    text: str
    thinking: str | None = None


class RunEvent(BaseModel):
    event_id: str
    run_id: str
    case_id: str
    sequence: int
    type: Literal[
        "run_started",
        "planner_decision",
        "reasoning_summary",
        "thinking_delta",
        "thinking",
        "phase_started",
        "phase_completed",
        "phase_failed",
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
    pending_gate: str | None = None
    pending_context: dict[str, Any] = Field(default_factory=dict)


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
    metrics: list[MetricEvidence] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    derived_from: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    data: dict[str, Any] = Field(default_factory=dict)
