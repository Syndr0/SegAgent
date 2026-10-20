from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

from .schemas import Intent, TaskIntent


@dataclass(frozen=True)
class IntentProfile:
    """What a recognized intent selects for the rest of the pipeline."""

    prompt_mode: str      # "label" | "sentence"
    knowledge: str        # "none" | "oar_protocol" | "target_guidelines"
    qc: str               # "geometry" | "oar" | "target"
    trust: str            # "review" | "mandatory_edit"


PROFILES: dict[Intent, IntentProfile] = {
    Intent.ORGAN: IntentProfile("label", "none", "geometry", "review"),
    Intent.OAR: IntentProfile("label", "oar_protocol", "oar", "review"),
    Intent.GTV: IntentProfile("sentence", "target_guidelines", "target", "mandatory_edit"),
    Intent.QC: IntentProfile("label", "oar_protocol", "oar", "review"),
    Intent.REPORT: IntentProfile("label", "oar_protocol", "geometry", "review"),
    Intent.PROTOCOL: IntentProfile("label", "oar_protocol", "geometry", "review"),
    Intent.VQA: IntentProfile("label", "none", "geometry", "review"),
    Intent.MASK_QUESTION: IntentProfile("label", "none", "geometry", "review"),
    Intent.AMBIGUOUS: IntentProfile("label", "none", "geometry", "review"),
    Intent.MISSING_INPUT: IntentProfile("label", "none", "geometry", "review"),
    Intent.QUESTION: IntentProfile("label", "none", "geometry", "review"),
}


def profile_for(intent: Intent) -> IntentProfile:
    return PROFILES[intent]


GTV_TERMS = (
    "gtv", "gross tumor", "gross tumour", "tumor", "tumors", "tumour", "tumours",
    "lesion", "lesions", "mass", "masses", "cancer", "cancers", "carcinoma",
    "carcinomas", "metastas", "malignan", "neoplasm", "neoplasms",
    "target volume", "ctv", "ptv", "itv", "靶区", "肿瘤", "病灶", "转移", "癌",
)
OAR_TERMS = (
    "oar", "organs at risk", "organ at risk", "risk organ", "spare", "radiotherapy",
    "radiation plan", "contour all", "危及器官", "放疗", "危及",
)
QC_STRONG = ("qc", "quality control", "quality-control", "质控", "勾画检查", "审核")
QC_VERBS = ("check", "audit", "review", "verify", "inspect", "validate", "核对")
CONTOUR_NOUNS = ("contour", "contours", "勾画", "delineation", "segmentation", "structures")
REPORT_TERMS = ("report", "summary", "summarize", "summarise", "报告", "总结", "汇总")
PROTOCOL_TERMS = (
    "protocol", "guideline", "which oar", "which structures", "what structures",
    "which organs", "指南", "协议", "规范",
)
SEGMENT_VERBS = (
    "segment", "contour", "delineate", "outline", "draw", "find", "locate",
    "identify", "mark", "highlight", "分割", "勾画", "标注", "找出", "定位",
)
MASK_REFERENCE_TERMS = (
    "mask", "contour", "segmentation", "this one", "that one", "selected",
    "visible", "it", "蒙版", "掩膜", "这个", "选中",
)
VQA_PATTERNS = (
    re.compile(r"\b(?:what|which)\b.+\b(?:image|scan|volume|view)\b", re.IGNORECASE),
    re.compile(r"\b(?:describe|explain)\b.+\b(?:image|scan|volume|view)\b", re.IGNORECASE),
    re.compile(r"\bwhat(?:'s| is)\s+(?:shown|visible|in)\b", re.IGNORECASE),
    re.compile(r"(?:图像|影像|扫描).*(?:是什么|有什么|显示|可见|描述)"),
)


# Common bilingual aliases. Canonical labels still come from the protocol
# registry; this table only maps natural-language mentions onto those labels.
_CJK_BASE = {
    "eye": "眼",
    "lens": "晶状体",
    "kidney": "肾",
    "lung": "肺",
    "parotid gland": "腮腺",
    "submandibular gland": "颌下腺",
    "optic nerve": "视神经",
    "cochlea": "耳蜗",
    "hippocampus": "海马",
    "adrenal gland": "肾上腺",
    "ovary": "卵巢",
    "femoral head": "股骨头",
    "humeral head": "肱骨头",
}
_CJK_EXACT = {
    "brain": ("脑",),
    "brainstem": ("脑干",),
    "spinal cord": ("脊髓",),
    "optic chiasm": ("视交叉",),
    "pituitary gland": ("垂体",),
    "larynx": ("喉", "喉部"),
    "oral cavity": ("口腔",),
    "mandible": ("下颌骨",),
    "thyroid gland": ("甲状腺",),
    "esophagus": ("食管",),
    "heart": ("心脏",),
    "trachea": ("气管",),
    "aorta": ("主动脉",),
    "liver": ("肝", "肝脏"),
    "spleen": ("脾", "脾脏"),
    "stomach": ("胃",),
    "duodenum": ("十二指肠",),
    "pancreas": ("胰腺",),
    "gallbladder": ("胆囊",),
    "small bowel": ("小肠",),
    "colon": ("结肠",),
    "bladder": ("膀胱",),
    "rectum": ("直肠",),
    "prostate": ("前列腺",),
    "uterus": ("子宫",),
    "cervix": ("宫颈",),
    "vagina": ("阴道",),
}


def _contains(text: str, term: str) -> bool:
    if not term:
        return False
    if term.isascii():
        return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None
    return term in text


def _has(text: str, terms: Iterable[str]) -> bool:
    # A few entries intentionally are stems (metastas-, malignan-).
    return any(term in text if term.endswith(("as", "an")) else _contains(text, term) for term in terms)


def aliases_for_organ(canonical: str) -> tuple[str, ...]:
    canonical = canonical.casefold().strip()
    aliases = [canonical]
    aliases.extend(_CJK_EXACT.get(canonical, ()))
    for side, cjk_side in (("left ", "左"), ("right ", "右")):
        if canonical.startswith(side):
            translated = _CJK_BASE.get(canonical[len(side):])
            if translated:
                aliases.append(cjk_side + translated)
    return tuple(dict.fromkeys(aliases))


def _term_spans(text: str, term: str) -> list[tuple[int, int]]:
    if not term:
        return []
    if term.isascii():
        pattern = re.compile(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])")
        return [(match.start(), match.end()) for match in pattern.finditer(text)]
    spans: list[tuple[int, int]] = []
    start = 0
    while (index := text.find(term, start)) >= 0:
        spans.append((index, index + len(term)))
        start = index + len(term)
    return spans


def match_canonical_terms(text: str, organ_names: Iterable[str]) -> list[str]:
    """Return longest non-overlapping anatomy matches in mention order.

    This prevents ``brain`` from being returned for ``brainstem`` and ``colon``
    from being returned for ``sigmoid colon`` while preserving distinct organs
    named in the same request.
    """

    lowered = " ".join((text or "").casefold().split())
    candidates: list[tuple[int, int, str]] = []
    for name in dict.fromkeys(item.casefold().strip() for item in organ_names if item):
        for alias in aliases_for_organ(name):
            candidates.extend((start, end, name) for start, end in _term_spans(lowered, alias))
    chosen: list[tuple[int, int, str]] = []
    for start, end, name in sorted(
        candidates, key=lambda item: (-(item[1] - item[0]), item[0], item[2])
    ):
        if any(not (end <= left or start >= right) for left, right, _ in chosen):
            continue
        chosen.append((start, end, name))
    chosen.sort(key=lambda item: item[0])
    return list(dict.fromkeys(name for _, _, name in chosen))


def _speech_act(text: str) -> str:
    """Classify the requested action before interpreting anatomy/site words."""

    if _has(text, QC_STRONG) or (_has(text, QC_VERBS) and _has(text, CONTOUR_NOUNS)):
        return "qc"
    if _has(text, REPORT_TERMS):
        return "report"
    if _has(text, PROTOCOL_TERMS):
        return "protocol"
    if _has(text, SEGMENT_VERBS):
        return "segment"
    if any(pattern.search(text) for pattern in VQA_PATTERNS):
        return "vqa"
    if (
        "?" in text
        or "？" in text
        or re.match(r"^(?:what|where|which|how|why|is|are|does|请问)\b", text)
        or re.match(
            r"^(?:can|could|do)\s+(?:you|we|i|this|that|it|there)\b",
            text,
        )
    ):
        return "question"
    return "statement"


_OPEN_EN = re.compile(
    r"(?:^|\b)(?:segment|contour|delineate|outline|draw|find|locate|identify|mark|highlight)"
    r"(?:\s+(?:out|me|us))?\s+"
    r"(?:(?:all|any|the|an|a|this|that)\s+)?(.+)$",
    re.IGNORECASE,
)
_OPEN_CJK = re.compile(r"^(?:请|帮我|麻烦)?(?:分割|勾画|标注|找出|定位)(?:一下)?\s*(.+)$")
_OPEN_TRAILING = re.compile(
    r"\s+(?:on|in|from)\s+(?:this|the)\s+(?:image|scan|volume|case)\s*$",
    re.IGNORECASE,
)


def extract_open_target(question: str) -> str | None:
    """Extract an unknown anatomical phrase from an explicit contouring command."""

    cleaned = " ".join((question or "").strip().split())
    match = _OPEN_CJK.search(cleaned) or _OPEN_EN.search(cleaned)
    if not match:
        return None
    target = _OPEN_TRAILING.sub("", match.group(1)).strip(" \t.,;:?!，。；：？！")
    if not target:
        return None
    generic = {
        "it", "this", "that", "thing", "something", "image", "scan", "case",
        "oar", "oars", "organ at risk", "organs at risk", "all structures",
        "它", "这个", "图像", "影像", "危及器官",
    }
    return None if target.casefold() in generic else target


class IntentClassifier:
    """Deterministic intent recognition with explicit abstention categories."""

    def __init__(
        self,
        organ_names: Iterable[str] | None = None,
        site_aliases: Iterable[str] | Mapping[str, str] | None = None,
    ):
        self.organ_names = tuple(
            sorted({name.casefold() for name in (organ_names or []) if name}, key=len, reverse=True)
        )
        if isinstance(site_aliases, Mapping):
            aliases = {
                str(alias).casefold(): str(site)
                for alias, site in site_aliases.items()
                if alias and site
            }
        else:
            aliases = {
                str(alias).casefold(): str(alias)
                for alias in (site_aliases or [])
                if alias
            }
        self.site_aliases = aliases

    @classmethod
    def from_knowledge(cls, knowledge) -> "IntentClassifier":
        organs = {organ for p in knowledge.protocols for organ in p.get("oars", [])}
        aliases: dict[str, str] = {}
        for protocol in knowledge.protocols:
            site = str(protocol.get("site", ""))
            if site:
                aliases[site] = site
            for alias in protocol.get("aliases", []):
                aliases[str(alias)] = site
        return cls(organs, aliases)

    def _found_organs(self, text: str) -> list[str]:
        return match_canonical_terms(text, self.organ_names)

    def _found_site(self, text: str) -> str | None:
        hits: list[tuple[int, str]] = []
        for alias, site in self.site_aliases.items():
            if _contains(text, alias):
                hits.append((len(alias), site))
        return max(hits, default=(0, None), key=lambda item: item[0])[1]

    def classify(
        self,
        question: str,
        *,
        has_contours: bool = False,
        source_description: str | None = None,
    ) -> TaskIntent:
        text = " ".join(question.casefold().split())
        act = _speech_act(text)
        organs = self._found_organs(text)
        site = self._found_site(text)
        has_report = bool(source_description and source_description.strip())

        gtv = _has(text, GTV_TERMS) or has_report
        explicit_oar = _has(text, OAR_TERMS)
        protocol = act == "protocol"

        if act == "qc":
            if has_contours:
                return TaskIntent(
                    intent=Intent.QC, confidence=0.9, targets=organs, site=site,
                    rationale="Contour check requested and contours are uploaded.",
                )
            return TaskIntent(
                intent=Intent.MISSING_INPUT, confidence=0.95,
                rationale="Contour QC requires uploaded contours, but this case has none.",
            )

        # The requested speech act wins over nouns mentioned in its scope:
        # "report on the GTV and OARs" is a report request, not a new tumor
        # contouring turn.
        if act == "report":
            return TaskIntent(
                intent=Intent.REPORT, confidence=0.85, targets=organs, site=site,
                rationale="A structure-based case report was requested.",
            )

        if gtv and explicit_oar and not protocol:
            if has_report:
                return TaskIntent(
                    intent=Intent.GTV, confidence=0.75, targets=organs, site=site,
                    source_description=source_description,
                    rationale="Clinical description present; treated as target contouring.",
                )
            return TaskIntent(
                intent=Intent.AMBIGUOUS, confidence=0.9, targets=organs, site=site,
                rationale="The request mixes tumor-target and OAR work.",
            )

        if gtv:
            return TaskIntent(
                intent=Intent.GTV, confidence=0.85, targets=organs, site=site,
                source_description=source_description or question,
                rationale="Tumor/target language detected.",
            )

        if protocol:
            return TaskIntent(
                intent=Intent.PROTOCOL, confidence=0.85, targets=organs, site=site,
                rationale="The user asks for a protocol or structure list.",
            )

        if explicit_oar:
            return TaskIntent(
                intent=Intent.OAR, confidence=0.9, targets=organs, site=site,
                rationale="Explicit organs-at-risk or radiotherapy language detected.",
            )

        # Speech act wins over site aliases: an explicitly named organ remains a
        # direct organ request even if "lung" or "kidney" is also a site alias.
        if act == "segment" and organs:
            return TaskIntent(
                intent=Intent.ORGAN, confidence=0.95, targets=organs,
                rationale="Direct segmentation of canonical anatomical structures requested.",
            )

        if act == "segment" and site:
            return TaskIntent(
                intent=Intent.OAR, confidence=0.75, site=site,
                rationale="A radiotherapy region was named without an individual structure.",
            )

        if act == "segment":
            open_target = extract_open_target(question)
            if open_target:
                return TaskIntent(
                    intent=Intent.ORGAN, confidence=0.65, targets=[open_target],
                    rationale="An explicit segmentation command names an open-vocabulary target.",
                )
            return TaskIntent(
                intent=Intent.AMBIGUOUS, confidence=0.8,
                rationale="A segmentation action was requested without a resolvable target.",
            )

        if act == "vqa":
            return TaskIntent(
                intent=Intent.VQA, confidence=0.9, targets=organs,
                rationale="A descriptive visual question about the image was asked.",
            )

        if act == "question" and _has(text, MASK_REFERENCE_TERMS):
            return TaskIntent(
                intent=Intent.MASK_QUESTION, confidence=0.9, targets=organs,
                rationale="A follow-up question refers to an existing mask or contour.",
            )

        if act == "question" and organs:
            return TaskIntent(
                intent=Intent.VQA, confidence=0.8, targets=organs,
                rationale="A descriptive anatomical question was asked without a contouring action.",
            )

        if act == "question":
            return TaskIntent(
                intent=Intent.VQA, confidence=0.65,
                rationale="An otherwise unclassified question falls back to visual question answering.",
            )

        if organs:
            return TaskIntent(
                intent=Intent.ORGAN, confidence=0.7, targets=organs,
                rationale="An anatomical label was supplied as a direct structure request.",
            )

        return TaskIntent(
            intent=Intent.AMBIGUOUS, confidence=0.85,
            rationale="No supported task, structure, site, or visual question was identified.",
        )
