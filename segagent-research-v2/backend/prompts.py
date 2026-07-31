from __future__ import annotations

import re
from typing import Iterable

from .schemas import (
    Intent,
    PromptCategory,
    PromptNormalization,
    PromptNormalizationBatch,
    TargetType,
    TaskIntent,
)


VOXTELL_PROMPT_NORMALIZER_SYSTEM = """You normalize text prompts for VoxTell, a
3D text-prompted segmentation model. Classify every input independently into
exactly one language regime:

- anatomical_concept: a concise anatomical noun phrase, including laterality.
- pathological_structure: a concise named lesion, tumor, or pathological entity.
- clinical_description: report-style language containing location, morphology,
  size, contact, extension, or another spatial relationship.

Return exactly one item for every indexed source, in the same order. Copy
source_index and source_text exactly. Keep text that is already a good VoxTell
prompt unchanged. You may normalize an unambiguous anatomical alias or notation
such as eye_L to left eye and eye_R to right eye. Do not replace validated model
terms with synonyms: in particular, keep "small intestine", not "small bowel".
Never mark an LLM normalization as exploratory; that flag belongs only to the
deterministic, user-confirmed blind-search workflow.
For pathological_structure and clinical_description, preserve the complete
cleaned source text; never invent, remove, or flip diagnosis, negation,
laterality, subregion, size, morphology, contact, or neighboring anatomy.

Examples:
- liver -> anatomical_concept -> liver
- stomach -> anatomical_concept -> stomach
- left kidney -> anatomical_concept -> left kidney
- pancreas -> anatomical_concept -> pancreas
- skin -> anatomical_concept -> skin
- small intestine -> anatomical_concept -> small intestine
- gallbladder -> anatomical_concept -> gallbladder
- eye_L -> anatomical_concept -> left eye
- bladder carcinoma -> pathological_structure -> bladder carcinoma
- Bronchial carcinoma in the right upper lobe with pleural contact
  -> clinical_description
  -> Bronchial carcinoma in the right upper lobe with pleural contact

Output one JSON object matching the supplied schema and no extra prose."""


# User-validated VoxTell prompts are protected from synonym substitution. This
# vocabulary is intentionally separate from the radiotherapy protocol registry:
# "skin" and "small intestine" can be excellent model prompts without being OAR
# labels in a particular local protocol.
VOXTELL_ANATOMICAL_PROMPTS = frozenset(
    {
        "liver",
        "stomach",
        "left kidney",
        "left eye",
        "right eye",
        "pancreas",
        "skin",
        "small intestine",
        "gallbladder",
    }
)
VOXTELL_ANATOMY_ALIASES = {
    "iver": "liver",
    "eye_l": "left eye",
    "eye l": "left eye",
    "eye_r": "right eye",
    "eye r": "right eye",
}
VOXTELL_PATHOLOGICAL_PROMPTS = frozenset({"bladder carcinoma"})
VOXTELL_CLINICAL_PROMPTS = frozenset(
    {"bronchial carcinoma in the right upper lobe with pleural contact"}
)

_PATHOLOGY = re.compile(
    r"(?<![a-z])(?:"
    r"carcinomas?|cancers?|tumou?rs?|neoplasms?|metastas(?:is|es|ic)?|"
    r"lesions?|mass(?:es)?|sarcomas?|lymphomas?|melanomas?|adenomas?"
    r")(?![a-z])",
    re.IGNORECASE,
)
_DESCRIPTION_CUE = re.compile(
    r"(?<![a-z])(?:"
    r"in|within|at|from|with|adjacent|abutting|contact|extending|"
    r"invading|involving|encasing|anterior|posterior|superior|inferior|"
    r"upper|lower|lobe|segment|wall|surface"
    r")(?![a-z])",
    re.IGNORECASE,
)
_NUMBER_OR_UNIT = re.compile(
    r"(?<![\w.])\d+(?:\.\d+)?(?:\s*(?:mm|cm|ml|cc))?(?![\w.])",
    re.IGNORECASE,
)
_MEASUREMENT = re.compile(
    r"(?<![\w.])\d+(?:\.\d+)?\s*(?:mm|cm|ml|cc)(?![\w.])",
    re.IGNORECASE,
)
_GENERIC_PATHOLOGY = re.compile(
    r"^(?:(?:possible|suspected|recurrent|multiple)\s+)?"
    r"(?:"
    r"tumou?rs?|mass(?:es)?|lesions?|cancers?|carcinomas?|neoplasms?"
    r"|(?:i?gtv|ctv|ptv|itv)(?:p|n|pn)?"
    r")"
    r"(?:[\s_-]*(?:#|no\.?|number)?\s*(?:\d+|[ivx]+|[a-z]))?$",
    re.IGNORECASE,
)
_LIST_CONTINUATION = re.compile(
    r"^(?:"
    r"and\s+(?:with|without|adjacent|abutting|contacting|extending|invading|"
    r"involving|encasing|measuring|located|arising)"
    r"|with|without|which|that|where|adjacent|abutting|contacting|extending|"
    r"invading|involving|encasing|measuring|located|arising"
    r"|\d+(?:\.\d+)?\s*(?:mm|cm|ml|cc)\b"
    r")",
    re.IGNORECASE,
)


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
    r"(?:(?:all|any|the|an|a)\s+)?",
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


def _source_category(text: str) -> PromptCategory:
    """Conservative deterministic category used only if the LLM is unavailable."""

    if _GENERIC_PATHOLOGY.fullmatch(text):
        return PromptCategory.PATHOLOGICAL_STRUCTURE
    if _PATHOLOGY.search(text):
        if _DESCRIPTION_CUE.search(text) or _MEASUREMENT.search(text):
            return PromptCategory.CLINICAL_DESCRIPTION
        return PromptCategory.PATHOLOGICAL_STRUCTURE
    # A long, spatially phrased unknown target is safer to preserve as report
    # language than to collapse into a guessed anatomy label.
    if len(text.split()) >= 6 and _DESCRIPTION_CUE.search(text):
        return PromptCategory.CLINICAL_DESCRIPTION
    return PromptCategory.ANATOMICAL_CONCEPT


def is_generic_pathology_prompt(text: str) -> bool:
    """Return whether text is only a generic target name or numbered instance."""

    return bool(_GENERIC_PATHOLOGY.fullmatch(clean_prompt(text)))


def _known_anatomy_keys(known_anatomy: Iterable[str] = ()) -> set[str]:
    from .qc import canonical_roi

    values = {
        *VOXTELL_ANATOMICAL_PROMPTS,
        *VOXTELL_ANATOMY_ALIASES,
        *VOXTELL_ANATOMY_ALIASES.values(),
        *(str(item) for item in known_anatomy if item),
    }
    return {canonical_roi(item) for item in values if canonical_roi(item)}


def _is_standalone_prompt(text: str, known_anatomy: Iterable[str] = ()) -> bool:
    """Conservatively identify list items that may be split around ``and``."""

    cleaned = clean_prompt(text)
    if not cleaned:
        return False
    if _is_known_anatomy_prompt(cleaned, known_anatomy):
        return True
    return (
        _source_category(cleaned) == PromptCategory.PATHOLOGICAL_STRUCTURE
        and not _DESCRIPTION_CUE.search(cleaned)
        and len(cleaned.split()) <= 5
    )


def _is_known_anatomy_prompt(
    text: str,
    known_anatomy: Iterable[str] = (),
) -> bool:
    from .qc import canonical_roi

    cleaned = clean_prompt(text)
    return bool(
        cleaned
        and canonical_roi(cleaned) in _known_anatomy_keys(known_anatomy)
    )


def _top_level_prompt_parts(source: str) -> list[tuple[str, str | None]]:
    """Split punctuation outside brackets while retaining each delimiter."""

    pairs = {"(": ")", "[": "]", "{": "}"}
    closing = set(pairs.values())
    stack: list[str] = []
    buffer: list[str] = []
    parts: list[tuple[str, str | None]] = []
    for char in source:
        if char in pairs:
            stack.append(pairs[char])
        elif char in closing and stack and char == stack[-1]:
            stack.pop()
        if char in {",", ";", "；", "、"} and not stack:
            parts.append(("".join(buffer).strip(), char))
            buffer = []
        else:
            buffer.append(char)
    parts.append(("".join(buffer).strip(), None))
    return parts


def _punctuation_is_list_boundary(
    current: str,
    following: str,
    delimiter: str,
    known_anatomy: Iterable[str],
) -> bool:
    """Return whether punctuation separates prompts rather than report clauses."""

    if not current or not following or _LIST_CONTINUATION.match(following):
        return False
    candidate = re.sub(r"^and\s+", "", following, flags=re.IGNORECASE)
    if _is_known_anatomy_prompt(current, known_anatomy):
        return True
    current_is_short_pathology = (
        _source_category(current) == PromptCategory.PATHOLOGICAL_STRUCTURE
        and len(current.split()) <= 5
    )
    if current_is_short_pathology and (
        _is_standalone_prompt(candidate, known_anatomy)
        or _PATHOLOGY.search(candidate)
    ):
        return True
    # A semicolon or ideographic list separator is stronger than a comma, but
    # still must not fragment an already descriptive clinical sentence.
    return (
        delimiter in {";", "；", "、"}
        and _source_category(current) != PromptCategory.CLINICAL_DESCRIPTION
    )


def split_prompt_sources(
    text: str,
    known_anatomy: Iterable[str] = (),
) -> list[str]:
    """Extract top-level VoxTell prompts without fragmenting report sentences.

    Commas, semicolons, and newlines separate user-supplied lists, except when a
    comma introduces a report continuation such as ``with pleural contact`` or
    a measurement. ``and`` is split only when every side is independently a
    known anatomy label or short named pathology; spatial phrases such as
    ``mass involving liver and stomach`` therefore remain intact.
    """

    known_anatomy = tuple(known_anatomy)
    source = clean_prompt(text)
    if not source:
        return []
    chunks: list[str] = []
    parts = _top_level_prompt_parts(source)
    current = parts[0][0] if parts else ""
    for index, (_, delimiter) in enumerate(parts[:-1]):
        following = parts[index + 1][0]
        delimiter = delimiter or ","
        if _punctuation_is_list_boundary(
            current,
            following,
            delimiter,
            known_anatomy,
        ):
            chunks.append(current)
            without_conjunction = re.sub(
                r"^and\s+",
                "",
                following,
                flags=re.IGNORECASE,
            )
            current = (
                without_conjunction
                if (
                    _is_standalone_prompt(without_conjunction, known_anatomy)
                    or _PATHOLOGY.search(without_conjunction)
                )
                else following
            )
        else:
            current = f"{current}{delimiter} {following}".strip()
    if current:
        chunks.append(current)

    split_chunks: list[str] = []
    for chunk in chunks:
        and_parts = [
            clean_prompt(item)
            for item in re.split(r"\s+\band\b\s+", chunk, flags=re.IGNORECASE)
        ]
        standalone = [
            _is_standalone_prompt(item, known_anatomy)
            for item in and_parts
        ]
        split_and = len(and_parts) > 1 and (
            all(standalone)
            or (
                standalone[0]
                and all(
                    is_standalone or bool(_PATHOLOGY.search(item))
                    for item, is_standalone in zip(
                        and_parts[1:],
                        standalone[1:],
                    )
                )
            )
        )
        if split_and:
            split_chunks.extend(and_parts)
        else:
            split_chunks.append(clean_prompt(chunk))

    generic = {
        "it",
        "this",
        "that",
        "image",
        "scan",
        "case",
        "oar",
        "oars",
        "organ at risk",
        "organs at risk",
        "all structures",
    }
    return [
        item
        for item in split_chunks
        if item and item.casefold() not in generic
    ]


def _canonical_anatomy(source: str, _candidate: str) -> str:
    """Resolve notation while protecting the empirically preferred vocabulary."""

    source_key = source.casefold()
    if source_key in VOXTELL_ANATOMICAL_PROMPTS:
        return source
    if source_key in VOXTELL_ANATOMY_ALIASES:
        return VOXTELL_ANATOMY_ALIASES[source_key]
    # The QC canonicalizer already owns conservative DICOM-style side aliases
    # such as Eye_L. Import locally to keep prompt construction lightweight.
    # Never accept an arbitrary LLM concept substitution here: "spleen" must
    # not silently become "liver", even if the model chose the right category.
    from .qc import canonical_roi

    canonical_source = canonical_roi(source)
    if canonical_source != source_key:
        return canonical_source
    return source


def _laterality(text: str) -> str | None:
    from .qc import canonical_roi

    tokens = canonical_roi(text).split()
    for side in ("left", "right", "bilateral"):
        if side in tokens:
            return side
    return None


def _validate_protected_details(source: str, candidate: str) -> None:
    source_side = _laterality(source)
    candidate_side = _laterality(candidate)
    if source_side and candidate_side != source_side:
        raise ValueError(
            f"normalization changed or removed laterality: {source_side}"
        )
    source_numbers = _NUMBER_OR_UNIT.findall(source)
    candidate_numbers = _NUMBER_OR_UNIT.findall(candidate)
    if source_numbers != candidate_numbers:
        raise ValueError("normalization changed or removed a number or unit")


def validate_prompt_normalizations(
    sources: Iterable[str],
    batch: PromptNormalizationBatch,
    known_anatomy: Iterable[str] = (),
) -> PromptNormalizationBatch:
    """Validate and protect one LLM result before any prompt reaches VoxTell."""

    cleaned_sources = [clean_prompt(item) for item in sources]
    if any(not item for item in cleaned_sources):
        raise ValueError("normalization sources must be non-empty")
    expected = list(range(len(cleaned_sources)))
    indexes = [item.source_index for item in batch.items]
    if indexes != expected:
        raise ValueError(
            "normalizer must return every source index once and in source order"
        )

    known_anatomy_keys = _known_anatomy_keys(known_anatomy)
    protected: list[PromptNormalization] = []
    for source, item in zip(cleaned_sources, batch.items):
        if clean_prompt(item.source_text) != source:
            raise ValueError("normalizer changed source_text")
        candidate = clean_prompt(item.voxtell_prompt)
        if not candidate:
            raise ValueError("normalizer returned an empty VoxTell prompt")

        source_key = source.casefold()
        from .qc import canonical_roi

        source_is_known_anatomy = canonical_roi(source) in known_anatomy_keys
        deterministic_category = _source_category(source)
        if (
            deterministic_category
            in {
                PromptCategory.PATHOLOGICAL_STRUCTURE,
                PromptCategory.CLINICAL_DESCRIPTION,
            }
            and item.category != deterministic_category
        ):
            raise ValueError(
                "pathological/report language was assigned the wrong prompt category"
            )
        if source_is_known_anatomy and (
            item.category != PromptCategory.ANATOMICAL_CONCEPT
        ):
            raise ValueError("a known anatomical prompt was misclassified")
        if source_key in VOXTELL_PATHOLOGICAL_PROMPTS and (
            item.category != PromptCategory.PATHOLOGICAL_STRUCTURE
        ):
            raise ValueError("the protected pathology prompt was misclassified")
        if source_key in VOXTELL_CLINICAL_PROMPTS and (
            item.category != PromptCategory.CLINICAL_DESCRIPTION
        ):
            raise ValueError("the protected clinical prompt was misclassified")

        if item.category == PromptCategory.ANATOMICAL_CONCEPT:
            _validate_protected_details(source, candidate)
            prompt = _canonical_anatomy(source, candidate)
            if (
                source_key not in VOXTELL_ANATOMICAL_PROMPTS
                and canonical_roi(candidate) != canonical_roi(prompt)
            ):
                raise ValueError(
                    "normalization substituted a different anatomical concept"
                )
        else:
            # The LLM selects the language regime, but report/pathology wording is
            # too clinically meaningful to paraphrase. Request wrappers were
            # already removed by clean_prompt().
            prompt = source
        _validate_protected_details(source, prompt)
        protected.append(
            item.model_copy(
                update={
                    "source_text": source,
                    "voxtell_prompt": prompt,
                    "normalized_by": "llm",
                    "fallback_reason": None,
                    "exploratory": False,
                }
            )
        )
    return PromptNormalizationBatch(items=protected, thinking=batch.thinking)


def fallback_prompt_normalizations(
    sources: Iterable[str],
    reason: str | None = None,
) -> PromptNormalizationBatch:
    """Safe, order-preserving fallback when structured LLM output is unusable."""

    items: list[PromptNormalization] = []
    for index, raw in enumerate(sources):
        source = clean_prompt(raw)
        if not source:
            continue
        category = _source_category(source)
        prompt = (
            _canonical_anatomy(source, source)
            if category == PromptCategory.ANATOMICAL_CONCEPT
            else source
        )
        items.append(
            PromptNormalization(
                source_index=len(items),
                source_text=source,
                category=category,
                voxtell_prompt=prompt,
                confidence=1.0 if prompt.casefold() != source.casefold() else 0.7,
                normalized_by="deterministic_fallback",
                fallback_reason=reason,
            )
        )
    if not items:
        raise ValueError("at least one non-empty prompt is required")
    return PromptNormalizationBatch(items=items)


def trusted_prompt_normalizations(
    sources: Iterable[str],
    category: PromptCategory,
    *,
    source_texts: Iterable[str] | None = None,
    exploratory: bool = False,
    reason: str | None = None,
) -> PromptNormalizationBatch:
    """Attach provenance to prompts selected by deterministic workflow stages."""

    prompts = [clean_prompt(raw) for raw in sources]
    prompts = [prompt for prompt in prompts if prompt]
    originals = (
        [clean_prompt(raw) for raw in source_texts]
        if source_texts is not None
        else list(prompts)
    )
    if not prompts or len(originals) != len(prompts) or any(
        not source for source in originals
    ):
        raise ValueError(
            "trusted prompt sources must align one-to-one with model prompts"
        )
    items: list[PromptNormalization] = []
    for source, prompt in zip(originals, prompts):
        items.append(
            PromptNormalization(
                source_index=len(items),
                source_text=source,
                category=category,
                voxtell_prompt=prompt,
                confidence=1.0,
                normalized_by="trusted_workflow",
                fallback_reason=reason,
                exploratory=exploratory,
            )
        )
    return PromptNormalizationBatch(items=items)


def dedupe_prompt_normalizations(
    batch: PromptNormalizationBatch,
) -> PromptNormalizationBatch:
    """Deduplicate model prompts without losing source-to-mask alignment."""

    seen: set[str] = set()
    items: list[PromptNormalization] = []
    for item in batch.items:
        key = item.voxtell_prompt.casefold()
        if key in seen:
            continue
        seen.add(key)
        items.append(item.model_copy(update={"source_index": len(items)}))
    return PromptNormalizationBatch(items=items, thinking=batch.thinking)


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
