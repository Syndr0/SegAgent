from __future__ import annotations

import re
from typing import Iterable

from .schemas import Intent, TargetType, TaskIntent


# Leading request phrasing, stripped before a description becomes a VoxTell
# prompt. VoxTell's text encoder was trained on report-derived descriptive
# language, so an imperative ("segment the ...") is out-of-distribution noise
# diluting the embedding that does the actual work.
_REQUEST_PREFIX = re.compile(
    r"^(?:"
    r"(?:please|kindly|now)\s+"
    r"|(?:can|could|would)\s+you\s+(?:please\s+)?"
    r"|i\s+(?:want|need)\s+(?:you\s+)?to\s+"
    r"|help\s+me\s+"
    r"|let['’]?s\s+"
    r")*"
    r"(?:"
    r"(?:segment|contour|delineate|outline|draw|find|locate|identify|mark|show|highlight)"
    r"(?:\s+(?:out|me|us))?"
    r"|where\s+(?:is|are)"
    r"|what\s+about"
    r")\s+"
    # Longest first: regex alternation takes the first match, so "a" before
    # "all" would leave "ll lesions ..." behind.
    r"(?:all|any|the|an|a)?\s*",
    re.IGNORECASE,
)

_CJK_REQUEST_PREFIX = re.compile(r"^(?:请|帮我|麻烦)?(?:分割|勾画|标注|找出|定位)(?:一下)?\s*")

_TRAILING = re.compile(r"[\s。，,.;；:：?？!！]+$")


def clean_prompt(text: str) -> str:
    """Turn a user request into a descriptive prompt for the segmentation model.

    Strips only the leading ask ("segment the", "where is the", "请分割") and
    trailing punctuation. Everything that carries meaning — anatomy, laterality,
    position, size, neighbouring structures — is left untouched, and a string
    that would be emptied by stripping is returned unchanged.
    """
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return ""
    stripped = _CJK_REQUEST_PREFIX.sub("", _REQUEST_PREFIX.sub("", cleaned)).strip()
    stripped = _TRAILING.sub("", stripped).strip()
    return stripped or _TRAILING.sub("", cleaned).strip()


_TARGET_BY_INTENT = {
    Intent.ORGAN: TargetType.ORGAN,
    Intent.OAR: TargetType.OAR,
    Intent.GTV: TargetType.GTV,
    Intent.QC: TargetType.CONTOUR,
}


def target_type_for(task_intent: TaskIntent | None) -> TargetType:
    if task_intent is None:
        return TargetType.UNKNOWN
    return _TARGET_BY_INTENT.get(task_intent.intent, TargetType.UNKNOWN)


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        cleaned = clean_prompt(item)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def build_prompts(task_intent: TaskIntent | None, structures: Iterable[str]) -> list[str]:
    """Construct the VoxTell prompt strings for a segmentation request.

    GTV (tier c): the clinical description is the prompt, so VoxTell's
    multi-stage fusion can act on the spatially-grounded language — this is the
    regime where it is categorically better than alternatives. The description is
    stripped of the user's request phrasing first, so the model receives
    "liver tumor", not "segment the liver tumor". OAR / organ (tier a): the
    descriptive structure labels are used directly.
    """
    structures = list(structures or [])
    if task_intent is not None and task_intent.intent == Intent.GTV:
        description = clean_prompt(task_intent.source_description or "")
        if description:
            lowered = description.casefold()
            prompts = [description]
            prompts.extend(s for s in structures if s.strip().casefold() not in lowered)
            return _dedupe(prompts)
    return _dedupe(structures)
