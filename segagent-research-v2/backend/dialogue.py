from __future__ import annotations

import re

from .schemas import Intent, PlannerAction, PlannerDecision

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
    "yes", "yeah", "correct", "right", "looks right", "that's it", "thats it",
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
            "that, say you are not sure and I will make a first attempt for you to "
            "check."
        ),
        gate="gtv_spatial",
    )


def gtv_confirm_gate(state) -> PlannerDecision | None:
    """After a first tumor mask, confirm its location before building on it.

    This replaces re-running an identical prompt after a rejection: a confirmed
    location lets the next prompt be enriched with neighbouring anatomy, while a
    rejection stops the run instead of repeating itself.
    """
    if _intent_of(state) != Intent.GTV.value or _fired(state, "gtv_confirm"):
        return None
    if not _answers(state).get("gtv_blind_attempt"):
        return None
    produced = any(
        item.get("tool") == "segment" and not item.get("data", {}).get("skipped")
        for item in (state.get("observations") or [])
    )
    if not produced:
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


def oar_site_guard(state, decision: PlannerDecision) -> PlannerDecision | None:
    """Block OAR contouring until the treatment site is confirmed.

    The planner proposes the site from the image (see the system prompt); this
    guard only guarantees that nothing is contoured before the user agrees, so a
    confident but wrong guess cannot silently produce a whole OAR set.
    """
    if decision.action != PlannerAction.SEGMENT:
        return None
    if _intent_of(state) != Intent.OAR.value or _fired(state, "oar_site"):
        return None
    intent_data = state.get("intent") or {}
    if (intent_data.get("site") or "").strip() or _answers(state).get("site"):
        return None
    if intent_data.get("targets"):
        return None
    return PlannerDecision(
        action=PlannerAction.ASK_USER,
        rationale_summary="OAR contouring requires a confirmed treatment site.",
        confidence=1.0,
        user_message=(
            "Which treatment site should I use for the organs at risk — for example "
            "head and neck, thorax, abdomen, or pelvis? Tell me the site (or correct "
            "me if I proposed the wrong one) and I will contour that protocol's set."
        ),
        gate="oar_site",
    )


def pre_plan_gate(state, organ_names=()) -> PlannerDecision | None:
    """Deterministic question that must be asked before the model plans."""
    return gtv_confirm_gate(state) or gtv_spatial_gate(state, organ_names)


def guard_decision(state, decision: PlannerDecision) -> PlannerDecision:
    """Veto a planned action that would skip a mandatory confirmation."""
    return oar_site_guard(state, decision) or decision


def apply_reply(state, gate: str, reply: str, organ_names=()) -> dict:
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
        answers["site"] = reply
        intent_data["site"] = reply
        update["intent"] = intent_data

    elif gate == "gtv_spatial":
        if mood == "unknown" or not has_spatial_detail(reply, organ_names):
            # No usable description: fall back to a bare prompt and confirm the
            # result with the user rather than guessing at a location.
            answers["gtv_blind_attempt"] = True
        else:
            answers["gtv_description"] = reply
            described = " ".join(
                filter(None, [intent_data.get("source_description"), reply])
            ).strip()
            intent_data["source_description"] = described
            update["intent"] = intent_data

    elif gate == "gtv_confirm":
        if mood == "no":
            answers["gtv_rejected"] = True
        else:
            answers["gtv_location_confirmed"] = True

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
