from __future__ import annotations

from typing import Iterable

from .schemas import Intent, TargetType, TaskIntent


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
        cleaned = " ".join(str(item).strip().split())
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def build_prompts(task_intent: TaskIntent | None, structures: Iterable[str]) -> list[str]:
    """Construct the VoxTell prompt strings for a segmentation request.

    GTV (tier c): the verbatim clinical description is the prompt, so VoxTell's
    multi-stage fusion can act on the spatially-grounded language — this is the
    regime where it is categorically better than alternatives. OAR / organ
    (tier a): the descriptive structure labels are used directly, which is the
    existing behaviour and is left unchanged.
    """
    structures = list(structures or [])
    if task_intent is not None and task_intent.intent == Intent.GTV:
        description = (task_intent.source_description or "").strip()
        if description:
            lowered = description.casefold()
            prompts = [description]
            prompts.extend(s for s in structures if s.strip().casefold() not in lowered)
            return _dedupe(prompts)
    return _dedupe(structures)
