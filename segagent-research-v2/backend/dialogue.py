from __future__ import annotations

import re

from .intent import match_canonical_terms
from .prompts import is_generic_pathology_prompt
from .schemas import (
    DialogueStage,
    Intent,
    PlannerAction,
    PlannerDecision,
    TargetType,
)

"""Deterministic dialogue rules for the tumor and OAR workflows.

These run outside the model so the safety-relevant sequencing cannot be skipped
by a bad generation: organs at risk are never contoured before the treatment
site is confirmed, and a tumor target is never accepted without the user
confirming its location. The model still supplies the wording and the proposal;
these rules only decide when a question is mandatory.
"""


# Words that indicate the user actually located or described the target, which
# is what VoxTell's language fusion needs to beat a bare "tumor" prompt.
LATERALITY = ("left", "right", "bilateral", "midline", "左", "右", "双侧")
# "segment" is deliberately absent: it is this application's most common verb
# ("segment the tumor"), so as an anatomical term it would match almost every
# request and defeat the check.
POSITION = (
    "upper", "lower", "superior", "inferior", "anterior", "posterior", "medial",
    "lateral", "proximal", "distal", "apex", "base", "lobe", "hilar",
    "peripheral", "central", "adjacent", "next to", "near", "above", "below",
    "上", "下", "前", "后", "内", "外", "叶", "段", "邻近", "旁",
)
SIZE = re.compile(r"\d+\s*(?:mm|cm|millimet|centimet|毫米|厘米)", re.IGNORECASE)

AFFIRMATIVE = (
    "yes", "yeah", "correct", "that's right", "that is right", "looks right",
    "that's it", "thats it",
    "confirm", "confirmed", "ok", "okay", "good", "对", "是", "正确", "没错", "确认",
)
NEGATIVE = (
    "no", "not correct", "incorrect", "wrong", "not right", "nope", "bad",
    "不对", "不是", "错", "错误", "不正确",
)
UNKNOWN = (
    "don't know", "dont know", "do not know", "not sure", "unsure", "no idea",
    "can't tell", "cannot tell", "unknown", "不知道", "不确定", "说不清",
)
MASK_MEASUREMENT_TERMS = (
    "how big", "how large", "volume", "size", "measure", "measurement", "voxel",
    "centroid", "bounding box", "bbox", "体积", "大小", "测量", "质心",
)


def _norm(text: str) -> str:
    return " ".join((text or "").casefold().split())


def _contains(text: str, term: str) -> bool:
    """Match a term, using word boundaries for Latin-script terms.

    Substring matching would let "base" fire on "based on" and "no" on "nodule";
    CJK terms have no word boundaries, so they stay substring matches.
    """
    if term.isascii():
        return re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", text) is not None
    return term in text


def _has(text: str, terms) -> bool:
    return any(_contains(text, term) for term in terms)


def has_spatial_detail(text: str, organ_names=()) -> bool:
    """True when the text locates or describes a target, not just names it."""
    lowered = _norm(text)
    if not lowered:
        return False
    if _has(lowered, LATERALITY) or _has(lowered, POSITION) or SIZE.search(lowered):
        return True
    # Naming a neighbouring organ is itself a spatial anchor ("mass in the liver").
    return any(name and _contains(lowered, name.casefold()) for name in organ_names)


def reply_sentiment(text: str) -> str:
    """Classify a short confirmation reply as yes / no / unknown / other."""
    lowered = _norm(text)
    if not lowered:
        return "other"
    if _has(lowered, UNKNOWN):
        return "unknown"
    # Check negatives first: "not correct" contains "correct".
    if _has(lowered, NEGATIVE):
        return "no"
    if _has(lowered, AFFIRMATIVE):
        return "yes"
    return "other"


def _intent_of(state) -> str:
    return ((state.get("intent") or {}).get("intent")) or ""


def _answers(state) -> dict:
    return state.get("answers") or {}


def _fired(state, gate: str) -> bool:
    return gate in (state.get("gates_fired") or [])


def _normalized_initial_target(state) -> dict | None:
    items = list(state.get("prompt_normalizations") or [])
    if len(items) != 1:
        return None
    item = items[0]
    category = item.get("category")
    prompt = " ".join(str(item.get("voxtell_prompt") or "").split())
    if not prompt:
        return None
    # A category mistake must never make a bare target name (including
    # numbered instances such as "tumor 2") bypass the spatial-detail gate.
    if is_generic_pathology_prompt(prompt):
        return None
    if category == "clinical_description":
        return item
    if category == "pathological_structure":
        return item
    return None


def gtv_spatial_gate(state, organ_names=()) -> PlannerDecision | None:
    """A tumor request with no spatial description must ask for one first.

    A bare "tumor" prompt is the weakest thing VoxTell can be given; the spatial
    description the user supplies is what makes the prompt work, so it is worth
    one question before spending a segmentation.
    """
    if _intent_of(state) != Intent.GTV.value or _fired(state, "gtv_spatial"):
        return None
    intent_data = state.get("intent") or {}
    described = " ".join(
        filter(None, [intent_data.get("source_description"), state.get("question")])
    )
    # VoxTell also has a strong short-phrase regime for named pathological
    # structures (for example "bladder carcinoma"). Do not discard that
    # validated phrase in favor of the exploratory blind prompt.
    if _normalized_initial_target(state) is not None:
        return None
    if has_spatial_detail(described, organ_names):
        return None
    return PlannerDecision(
        action=PlannerAction.ASK_USER,
        rationale_summary="Tumor request has no spatial description; asking before segmenting.",
        confidence=1.0,
        user_message=(
            "Where is the tumor, roughly? Anything you can give helps: which organ "
            "or region it sits in, which side, its approximate size or shape, or "
            "which structures it lies next to. If you would rather I try without "
            "that, say you are not sure and I will make an exploratory first "
            "attempt using a generic report-style prompt. That prompt is only a "
            "search aid, not a diagnosis."
        ),
        gate="gtv_spatial",
    )


def gtv_confirm_gate(state) -> PlannerDecision | None:
    """After a first tumor mask, confirm its location before building on it.

    This applies to every tumor contour, however it was prompted: a described
    target can land in the wrong place just as a blind one can. Confirming lets
    the next prompt be enriched with neighbouring anatomy; rejecting stops the
    run rather than re-running a prompt that would return the same contour.
    """
    if _intent_of(state) != Intent.GTV.value:
        return None
    answers = _answers(state)
    # Only once a tumor mask actually exists: there is nothing to confirm before.
    if not answers.get("gtv_first_mask"):
        return None
    if answers.get("gtv_location_confirmed") or answers.get("gtv_rejected"):
        return None
    return PlannerDecision(
        action=PlannerAction.ASK_USER,
        rationale_summary="First tumor contour produced; confirming its location with the user.",
        confidence=1.0,
        user_message=(
            "Here is a first attempt. Is it in roughly the right place — the right "
            "organ or region? Reply yes and I will contour the neighbouring "
            "structures to sharpen it, or no if it is in the wrong place entirely."
        ),
        gate="gtv_confirm",
    )


def oar_site_gate(state) -> PlannerDecision | None:
    """Request a treatment site before any protocol lookup or segmentation."""

    if _intent_of(state) != Intent.OAR.value:
        return None
    intent_data = state.get("intent") or {}
    answers = _answers(state)
    if intent_data.get("targets"):
        return None
    if (intent_data.get("site") or answers.get("site") or "").strip():
        return None
    return PlannerDecision(
        action=PlannerAction.ASK_USER,
        rationale_summary="The OAR workflow is waiting for a confirmed treatment site.",
        confidence=1.0,
        user_message=(
            "Which treatment site should I use for the organs-at-risk protocol "
            "(for example, head and neck, thorax, upper abdomen, or pelvis)? "
            "I will retrieve that protocol and contour its complete OAR list in one batch."
        ),
        gate="oar_site",
        workflow_stage=DialogueStage.OAR_SITE,
    )


def oar_site_guard(state, decision: PlannerDecision) -> PlannerDecision | None:
    """Block OAR contouring until the treatment site is confirmed.

    The planner proposes the site from the image (see the system prompt); this
    guard only guarantees that nothing is contoured before the user agrees, so a
    confident but wrong guess cannot silently produce a whole OAR set.
    """
    if _intent_of(state) != Intent.OAR.value:
        return None
    intent_data = state.get("intent") or {}
    if (intent_data.get("site") or "").strip() or _answers(state).get("site"):
        return None
    if intent_data.get("targets"):
        return None
    if decision.action == PlannerAction.ASK_USER:
        return decision.model_copy(
            update={
                "gate": "oar_site",
                "workflow_stage": DialogueStage.OAR_SITE,
            }
        )
    return oar_site_gate(state)


def report_scope_gate(state, has_contours: bool) -> PlannerDecision | None:
    """A report with nothing contoured yet must offer to contour first.

    Contouring a full OAR set takes minutes, so it is offered rather than
    started silently just because a report was requested.
    """
    if _intent_of(state) != Intent.REPORT.value or _fired(state, "report_scope"):
        return None
    readiness = state.get("report_readiness") or {}
    if readiness.get("ready"):
        return None
    if not readiness and has_contours:
        return None
    missing = list(readiness.get("missing_inputs") or [])
    return PlannerDecision(
        action=PlannerAction.ASK_USER,
        rationale_summary="The requested report scope is missing deterministic structure evidence.",
        confidence=1.0,
        user_message=(
            "A comprehensive report needs both organ/OAR and target contours. "
            f"Missing now: {', '.join(missing) or 'required contours'}. "
            "Upload or request the missing contours, or reply “available” for a "
            "report limited explicitly to structures already present."
        ),
        gate="report_scope",
    )


def question_gate(state) -> PlannerDecision | None:
    """A descriptive question is answered, not acted on.

    "What is in this image?" is visual question answering: it must not trigger a
    protocol lookup or a segmentation run. Routing this deterministically keeps
    the planner from turning a question into work the user did not ask for.
    """
    if _intent_of(state) not in {Intent.VQA.value} or _fired(state, "vqa"):
        return None
    return PlannerDecision(
        action=PlannerAction.ANSWER_QUESTION,
        rationale_summary="A descriptive question about the image; answering it directly.",
        confidence=1.0,
        gate="vqa",
    )


def mask_question_gate(state) -> PlannerDecision | None:
    """Route selected-mask follow-ups by artifact ID, never by pronoun guessing."""

    if _intent_of(state) != Intent.MASK_QUESTION.value or _fired(state, "mask_question"):
        return None
    context = state.get("message_context") or {}
    intent_data = state.get("intent") or {}
    selected = context.get("selected_mask_id")
    referenced = list(context.get("referenced_artifact_ids") or [])
    visible = list(context.get("visible_mask_ids") or [])
    targets = list(intent_data.get("targets") or [])
    inventory = list(state.get("artifact_inventory") or [])
    mask_ids = {
        item.get("artifact_id")
        for item in inventory
        if item.get("kind") in {"mask", "contour"} and item.get("active", True)
    }
    referenced_masks = [
        artifact_id
        for artifact_id in referenced
        if not inventory or artifact_id in mask_ids
    ]
    exact_label_ids = [
        item.get("artifact_id")
        for item in inventory
        if item.get("kind") in {"mask", "contour"}
        and item.get("active", True)
        and _norm(str(item.get("label") or "")) == _norm(state.get("question", ""))
    ]
    if selected:
        measurement_structures = [selected]
    elif targets:
        measurement_structures = targets
    elif len(exact_label_ids) == 1:
        measurement_structures = exact_label_ids
    elif len(referenced_masks) == 1:
        measurement_structures = referenced_masks
    elif len(visible) == 1:
        measurement_structures = visible
    else:
        measurement_structures = []
    wants_measurement = _has(_norm(state.get("question", "")), MASK_MEASUREMENT_TERMS)
    if wants_measurement and measurement_structures:
        return PlannerDecision(
            action=PlannerAction.INSPECT_MASK,
            rationale_summary="A quantitative follow-up refers to a case-owned mask.",
            confidence=1.0,
            structures=measurement_structures,
            gate="mask_question",
        )
    if wants_measurement:
        return PlannerDecision(
            action=PlannerAction.ASK_USER,
            rationale_summary="The quantitative mask question is not tied to one exact artifact.",
            confidence=1.0,
            user_message="Please select the one mask you want me to measure.",
            gate="mask_reference",
        )
    if (
        selected
        or referenced_masks
        or targets
        or len(exact_label_ids) == 1
        or len(visible) == 1
    ):
        return PlannerDecision(
            action=PlannerAction.ANSWER_QUESTION,
            rationale_summary="A visual follow-up refers to existing contour evidence.",
            confidence=1.0,
            gate="mask_question",
        )
    if len(visible) > 1:
        return PlannerDecision(
            action=PlannerAction.ASK_USER,
            rationale_summary="The visual mask question is ambiguous across multiple visible masks.",
            confidence=1.0,
            user_message=(
                "Several masks are visible. Please name the contour you mean "
                "or leave only that mask visible before asking again."
            ),
            gate="mask_reference",
        )
    return PlannerDecision(
        action=PlannerAction.ASK_USER,
        rationale_summary="The mask follow-up did not identify a case artifact.",
        confidence=1.0,
        user_message="Please select or reference the mask you want me to inspect.",
        gate="mask_reference",
    )


def abstention_gate(state) -> PlannerDecision | None:
    """Turn explicit classifier abstentions into a focused clarification."""

    intent = _intent_of(state)
    if intent == Intent.MISSING_INPUT.value and not _fired(state, "missing_input"):
        return PlannerDecision(
            action=PlannerAction.ASK_USER,
            rationale_summary="The requested operation is missing required case input.",
            confidence=1.0,
            user_message=(
                "This request needs uploaded contour masks. Please upload the contours, "
                "then tell me to run the contour quality check again."
            ),
            gate="missing_input",
        )
    if intent == Intent.AMBIGUOUS.value and not _fired(state, "clarify_intent"):
        return PlannerDecision(
            action=PlannerAction.ASK_USER,
            rationale_summary="The request does not identify one safe workflow.",
            confidence=1.0,
            user_message=(
                "Please clarify whether you want a tumor target, a named organ, "
                "an organs-at-risk protocol, contour QC, or a description of the image."
            ),
            gate="clarify_intent",
        )
    return None


def pre_plan_gate(state, organ_names=(), has_contours: bool = False) -> PlannerDecision | None:
    """Deterministic question that must be asked before the model plans."""
    return (
        gtv_confirm_gate(state)
        or gtv_spatial_gate(state, organ_names)
        or report_scope_gate(state, has_contours)
        or mask_question_gate(state)
        or question_gate(state)
        or abstention_gate(state)
    )


# VoxTell was trained on report-derived language, so this deliberately uses a
# report-style pathology noun for the user-confirmed blind search. It must never
# replace the user's wording during ordinary normalization or imply histology.
TUMOR_PROMPT = "carcinoma"

# Stages the workflow dictates. Named constants rather than prose so the caller
# can branch on them safely (a stage decides the target type and which marker is
# recorded, which is too load-bearing for string prefixes).
STAGE_TUMOR_BLIND = "tumor_blind"
STAGE_TUMOR_INITIAL = "tumor_initial"
STAGE_TUMOR_NEIGHBOURS = "tumor_neighbours"
STAGE_TUMOR_REFINE = "tumor_refine"
STAGE_OAR_FULL = "oar_full"

STAGE_LABELS = {
    STAGE_TUMOR_INITIAL: "first target contour from the supplied spatial description",
    STAGE_TUMOR_BLIND: "exploratory first attempt with a report-style prompt",
    STAGE_TUMOR_NEIGHBOURS: "contouring neighbouring organs to locate the target",
    STAGE_TUMOR_REFINE: "re-contouring the target with a spatial prompt",
    STAGE_OAR_FULL: "contouring the confirmed protocol's full structure list",
}

# Which answer marker a completed stage records.
STAGE_MARKERS = {
    STAGE_TUMOR_INITIAL: "gtv_first_mask",
    STAGE_TUMOR_BLIND: "gtv_first_mask",
    STAGE_TUMOR_NEIGHBOURS: "gtv_neighbours",
    STAGE_TUMOR_REFINE: "gtv_refined",
    STAGE_OAR_FULL: "oar_contoured",
}


def _has_tool(observations, tool: str) -> bool:
    return any(item.get("tool") == tool for item in observations or [])


def _neighbour_names(state, organ_names=()) -> list[str]:
    answers = _answers(state)
    supplied = list(answers.get("gtv_neighbour_names") or [])
    if supplied:
        return list(dict.fromkeys(supplied))
    intent_data = state.get("intent") or {}
    named = list(intent_data.get("targets") or [])
    description = " ".join(
        filter(
            None,
            [
                intent_data.get("source_description"),
                answers.get("gtv_description"),
            ],
        )
    )
    named.extend(match_canonical_terms(description, organ_names))
    return list(dict.fromkeys(named))


def required_workflow_decision(
    state,
    protocol_oars=(),
    observations=(),
    organ_names=(),
) -> PlannerDecision | None:
    """Return the next mandatory FSM action, if one exists.

    The LLM still explains and summarizes, but it cannot skip site confirmation,
    protocol retrieval, a complete batched OAR segmentation, or a confirmed
    tumor refinement stage.
    """

    intent = _intent_of(state)
    answers = _answers(state)
    intent_data = state.get("intent") or {}

    if intent == Intent.QC.value:
        if not _has_tool(observations, "run_qc"):
            return PlannerDecision(
                action=PlannerAction.RUN_QC,
                rationale_summary="Run deterministic QC on the uploaded contour set.",
                confidence=1.0,
                site_query=intent_data.get("site"),
            )
        return None

    if intent == Intent.ORGAN.value:
        targets = list(intent_data.get("targets") or [])
        if targets and not _has_tool(observations, "segment"):
            return PlannerDecision(
                action=PlannerAction.SEGMENT,
                rationale_summary="Contour the directly requested anatomical structures.",
                confidence=1.0,
                structures=targets,
            )
        return None

    if intent == Intent.OAR.value:
        targets = list(intent_data.get("targets") or [])
        site = intent_data.get("site") or answers.get("site")
        if targets and not answers.get("oar_contoured"):
            return PlannerDecision(
                action=PlannerAction.SEGMENT,
                rationale_summary="Contour the explicitly requested OAR structures in one batch.",
                confidence=1.0,
                structures=targets,
                workflow_stage=DialogueStage.OAR_FULL,
            )
        if not site:
            # Let the multimodal planner inspect the scan and propose a site.
            # ``guard_decision`` converts every resulting action into a typed
            # oar_site question, so no lookup/segmentation can escape.
            return None
        if not _has_tool(observations, "lookup_protocol"):
            return PlannerDecision(
                action=PlannerAction.LOOKUP_PROTOCOL,
                rationale_summary="Retrieve the confirmed site's versioned OAR list.",
                confidence=1.0,
                site_query=str(site),
                workflow_stage=DialogueStage.OAR_PROTOCOL,
            )
        if protocol_oars and not answers.get("oar_contoured"):
            return PlannerDecision(
                action=PlannerAction.SEGMENT,
                rationale_summary="Contour the protocol's complete OAR list in one model batch.",
                confidence=1.0,
                structures=list(protocol_oars),
                workflow_stage=DialogueStage.OAR_FULL,
            )
        return None

    if intent != Intent.GTV.value:
        return None

    if not answers.get("gtv_first_mask"):
        if answers.get("gtv_blind_attempt"):
            return PlannerDecision(
                action=PlannerAction.SEGMENT,
                rationale_summary=(
                    "Make an exploratory first target attempt with the fixed "
                    "report-style fallback prompt; this is not a diagnosis."
                ),
                confidence=1.0,
                structures=[TUMOR_PROMPT],
                workflow_stage=DialogueStage.TUMOR_BLIND,
            )
        description = (
            answers.get("gtv_description")
            or intent_data.get("source_description")
            or state.get("question")
            or ""
        )
        normalized_target = _normalized_initial_target(state)
        if normalized_target is not None:
            return PlannerDecision(
                action=PlannerAction.SEGMENT,
                rationale_summary=(
                    "Contour the normalized pathological/report target prompt."
                ),
                confidence=1.0,
                structures=[normalized_target["voxtell_prompt"]],
                workflow_stage=DialogueStage.TUMOR_INITIAL,
            )
        if has_spatial_detail(str(description), organ_names):
            return PlannerDecision(
                action=PlannerAction.SEGMENT,
                rationale_summary="Make the first target contour from the supplied spatial description.",
                confidence=1.0,
                structures=[str(description)],
                workflow_stage=DialogueStage.TUMOR_INITIAL,
            )
        return None

    if answers.get("gtv_rejected") or not answers.get("gtv_location_confirmed"):
        return None

    if not answers.get("gtv_neighbours"):
        neighbours = _neighbour_names(state, organ_names)
        if neighbours:
            return PlannerDecision(
                action=PlannerAction.SEGMENT,
                rationale_summary="Contour the named anatomical anchors before target refinement.",
                confidence=1.0,
                structures=neighbours,
                workflow_stage=DialogueStage.TUMOR_NEIGHBOURS,
            )
        if answers.get("gtv_neighbours_skipped"):
            return None
        return PlannerDecision(
            action=PlannerAction.ASK_USER,
            rationale_summary="Target refinement needs at least one anatomical anchor.",
            confidence=1.0,
            user_message=(
                "Which nearby organ or structure should anchor the refinement? "
                "Name one or more, or say you are not sure to keep the first contour."
            ),
            gate="gtv_neighbours",
        )

    if not answers.get("gtv_refined"):
        enriched = build_tumor_prompt(observations)
        if enriched:
            return PlannerDecision(
                action=PlannerAction.SEGMENT,
                rationale_summary="Re-contour the target with measured spatial relations to its anchors.",
                confidence=1.0,
                structures=[enriched],
                workflow_stage=DialogueStage.TUMOR_REFINE,
            )
    return None


def _measurements(observations) -> list[dict]:
    rows: list[dict] = []
    for item in observations or []:
        data = item.get("data") or {}
        result = data.get("segment_result")
        if isinstance(result, dict):
            request = result.get("request") or {}
            target_type = request.get("target_type") or data.get("target_type")
            rows.extend(
                {
                    **row,
                    "_target_type": target_type,
                }
                for row in result.get("measurements", []) or []
                if row.get("found")
            )
    return rows


def _is_tumor_row(row) -> bool:
    if row.get("_target_type") == TargetType.GTV.value:
        return True
    return bool(
        re.search(
            r"(?<![a-z])(?:"
            r"tumou?rs?|lesions?|mass(?:es)?|carcinomas?"
            r"|(?:i?gtv|ctv|ptv|itv)(?:p|n|pn)?\d*"
            r")(?![a-z])",
            row.get("structure", ""),
            re.IGNORECASE,
        )
    )


def tumor_measurement(observations) -> dict | None:
    rows = [row for row in _measurements(observations) if _is_tumor_row(row)]
    return rows[-1] if rows else None


def organ_measurements(observations) -> list[dict]:
    return [row for row in _measurements(observations) if not _is_tumor_row(row)]


def build_tumor_prompt(observations) -> str | None:
    """Describe the tumor by its position relative to the organs around it.

    This is the enriched prompt: rather than re-running the word "tumor", which
    would return the same contour, the target is re-prompted with the spatial
    language VoxTell is actually good at, derived from measured centroids.
    """
    target = tumor_measurement(observations)
    organs = organ_measurements(observations)
    if target is None or not organs:
        return None
    point = target.get("centroid_mm") or target.get("centroid_voxels") or []
    if len(point) < 3:
        return None

    def offset(row):
        other = row.get("centroid_mm") or row.get("centroid_voxels") or []
        if len(other) < 3:
            return None
        return [float(point[i]) - float(other[i]) for i in range(3)]

    scored = []
    for row in organs:
        delta = offset(row)
        if delta is None:
            continue
        distance = sum(value * value for value in delta) ** 0.5
        scored.append((distance, row.get("structure", ""), delta))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0])

    # World centroids are RAS-oriented (with voxel centroids retained only as a
    # backward-compatible fallback), so directions and distances use physical
    # geometry rather than anisotropic raw voxel indices.
    def relation(delta) -> str | None:
        axis = max(range(3), key=lambda i: abs(delta[i]))
        if abs(delta[axis]) < 2.0:
            return None  # centroids coincide: the target sits within the organ
        if axis == 0:
            return "to the right of" if delta[0] > 0 else "to the left of"
        if axis == 1:
            return "anterior to" if delta[1] > 0 else "posterior to"
        return "superior to" if delta[2] > 0 else "inferior to"

    # The nearest structure is the anchor and carries a direction too: "adjacent
    # to the larynx" alone throws away the spatial cue the prompt exists for.
    _, nearest_name, nearest_delta = scored[0]
    heading = relation(nearest_delta)
    parts = [
        f"tumor within the {nearest_name}"
        if heading is None
        else f"tumor immediately {heading} the {nearest_name}"
    ]
    for _, name, delta in scored[1:3]:
        direction = relation(delta)
        parts.append(f"{direction} the {name}" if direction else f"overlapping the {name}")
    volume = target.get("volume_ml")
    if isinstance(volume, (int, float)):
        parts.append(f"approximately {round(float(volume), 1)} mL in volume")
    return ", ".join(parts)


def forced_structures(state, protocol_oars=(), observations=()) -> tuple[list[str], str] | None:
    """Structures the workflow dictates, overriding what the planner asked for.

    Deterministic so that the sequence the user specified actually happens: a
    confirmed OAR request contours the whole protocol list rather than a subset
    the model happened to name, and the tumor workflow walks its stages in order.
    """
    intent = _intent_of(state)
    answers = _answers(state)

    if intent == Intent.GTV.value:
        if not answers.get("gtv_first_mask"):
            description = (
                answers.get("gtv_description")
                or (state.get("intent") or {}).get("source_description")
            )
            if description and has_spatial_detail(description):
                return [description], STAGE_TUMOR_INITIAL
        if answers.get("gtv_blind_attempt") and not answers.get("gtv_first_mask"):
            return [TUMOR_PROMPT], STAGE_TUMOR_BLIND
        if answers.get("gtv_location_confirmed"):
            if not answers.get("gtv_neighbours"):
                neighbours = _neighbour_names(state, protocol_oars)
                if neighbours:
                    return neighbours, STAGE_TUMOR_NEIGHBOURS
            if not answers.get("gtv_refined"):
                enriched = build_tumor_prompt(observations)
                if enriched:
                    return [enriched], STAGE_TUMOR_REFINE
        return None

    if intent == Intent.OAR.value and protocol_oars:
        if not answers.get("oar_contoured"):
            return list(protocol_oars), STAGE_OAR_FULL
    return None


def guard_decision(state, decision: PlannerDecision) -> PlannerDecision:
    """Veto a planned action that would skip a mandatory confirmation."""
    return oar_site_guard(state, decision) or decision


def _resolve_site(reply: str, site_aliases=None) -> str | None:
    aliases = site_aliases or {}
    lowered = _norm(reply)
    matches = [
        (len(str(alias)), str(site))
        for alias, site in aliases.items()
        if _contains(lowered, str(alias).casefold())
    ]
    return max(matches, default=(0, None), key=lambda item: item[0])[1]


def apply_reply(state, gate: str, reply: str, organ_names=(), site_aliases=None) -> dict:
    """Fold a reply to a gated question back into the run state.

    Returns the state update: what was answered, how the tumor prompt should be
    enriched, and whether the run should stop because the user rejected the
    contour outright.
    """
    answers = dict(_answers(state))
    intent_data = dict(state.get("intent") or {})
    update: dict = {}
    mood = reply_sentiment(reply)

    if gate == "oar_site":
        proposed = answers.get("proposed_site")
        resolved = _resolve_site(reply, site_aliases)
        if mood == "yes" and proposed:
            resolved = str(proposed)
        if resolved:
            answers["site"] = resolved
            answers.pop("oar_site_unresolved", None)
            intent_data["site"] = resolved
            update["intent"] = intent_data
        else:
            answers.pop("proposed_site", None)
            answers["oar_site_unresolved"] = True

    elif gate == "gtv_spatial":
        if mood == "unknown" or not has_spatial_detail(reply, organ_names):
            # No usable description: use the fixed exploratory report-language
            # prompt and confirm the result rather than guessing at a location
            # or presenting the prompt as a diagnosis.
            answers["gtv_blind_attempt"] = True
        else:
            answers["gtv_description"] = reply
            described = " ".join(
                filter(None, [intent_data.get("source_description"), reply])
            ).strip()
            intent_data["source_description"] = described
            update["intent"] = intent_data

    elif gate == "gtv_confirm":
        correction = has_spatial_detail(reply, organ_names)
        if mood == "yes":
            answers["gtv_location_confirmed"] = True
            answers.pop("gtv_confirmation_uncertain", None)
        elif correction:
            # A correction is new target evidence, not a confirmation. Clear
            # only the target-stage markers so the corrected prompt gets a new
            # first attempt and a fresh confirmation.
            described = " ".join(
                filter(None, [intent_data.get("source_description"), reply])
            ).strip()
            intent_data["source_description"] = described
            update["intent"] = intent_data
            for key in (
                "gtv_first_mask",
                "gtv_location_confirmed",
                "gtv_rejected",
                "gtv_neighbours",
                "gtv_refined",
            ):
                answers.pop(key, None)
            answers["gtv_description"] = reply
            answers["gtv_confirmation_uncertain"] = False
        elif mood == "no":
            answers["gtv_rejected"] = True
        else:
            # "I'm not sure" and unrelated prose are explicitly not consent.
            answers["gtv_confirmation_uncertain"] = True

    elif gate == "gtv_neighbours":
        names = match_canonical_terms(reply, organ_names)
        if names:
            answers["gtv_neighbour_names"] = names
        elif mood == "unknown":
            answers["gtv_neighbours_skipped"] = True

    elif gate == "report_scope":
        lowered = _norm(reply)
        if _has(lowered, ("available", "existing", "current", "already present", "现有", "已有")):
            answers["report_scope"] = "available"
            update["report_scope"] = "available"

    update["answers"] = answers
    return update


def rejection_notice(state) -> str | None:
    """Message to end the turn when the user rejected the tumor contour."""
    if not _answers(state).get("gtv_rejected"):
        return None
    return (
        "Understood — I have stopped rather than trying the same prompt again, "
        "because repeating it would produce the same contour. If you can tell me "
        "which organ or region the tumor is in, I can try again with that."
    )
