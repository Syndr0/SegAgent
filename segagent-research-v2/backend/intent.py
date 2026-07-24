from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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
    Intent.PROTOCOL: IntentProfile("label", "oar_protocol", "geometry", "review"),
    Intent.QUESTION: IntentProfile("label", "none", "geometry", "review"),
}


def profile_for(intent: Intent) -> IntentProfile:
    return PROFILES[intent]


GTV_TERMS = (
    "gtv", "gross tumor", "gross tumour", "tumor", "tumour", "lesion", "carcinoma",
    "metastas", "malignan", "neoplasm", "target volume", "ctv", "靶区", "肿瘤",
    "病灶", "转移", "癌",
)
OAR_TERMS = (
    "oar", "organs at risk", "organ at risk", "risk organ", "spare", "radiotherapy",
    "radiation plan", "contour all", "危及器官", "放疗", "危及",
)
QC_STRONG = ("qc", "quality control", "quality-control", "质控", "勾画检查", "审核")
QC_VERBS = ("check", "audit", "review", "verify", "inspect", "核对")
CONTOUR_NOUNS = ("contour", "contours", "勾画", "delineation", "segmentation")
PROTOCOL_TERMS = (
    "protocol", "guideline", "which oar", "which structures", "what structures",
    "which organs", "指南", "协议", "规范",
)


def _has(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


class IntentClassifier:
    """Deterministic, abstaining intent recognition.

    Distinguishes organ / OAR / GTV / QC / protocol, and returns QUESTION
    (abstain) when the request is ambiguous or under-specified — because
    misreading an OAR request as a tumor target, or vice versa, has real
    clinical consequences.
    """

    def __init__(
        self,
        organ_names: Iterable[str] | None = None,
        site_aliases: Iterable[str] | None = None,
    ):
        self.organ_names = sorted(
            {name.casefold() for name in (organ_names or [])}, key=len, reverse=True
        )
        organ_set = set(self.organ_names)
        # Region aliases (thorax, pelvis, ...) signal an RT site; organ-name
        # aliases (liver, kidney) do not, so "segment the liver" stays ORGAN.
        self.region_aliases = sorted(
            {
                alias.casefold()
                for alias in (site_aliases or [])
                if alias and alias.casefold() not in organ_set
            },
            key=len,
            reverse=True,
        )

    @classmethod
    def from_knowledge(cls, knowledge) -> "IntentClassifier":
        organs = {organ for p in knowledge.protocols for organ in p.get("oars", [])}
        aliases: set[str] = set()
        for protocol in knowledge.protocols:
            aliases.add(protocol.get("site", ""))
            aliases.update(protocol.get("aliases", []))
        return cls(organs, aliases)

    def _found_organs(self, text: str) -> list[str]:
        return [name for name in self.organ_names if name in text]

    def _found_site(self, text: str) -> str | None:
        hits = [alias for alias in self.region_aliases if alias in text]
        return max(hits, key=len) if hits else None

    def classify(
        self,
        question: str,
        *,
        has_contours: bool = False,
        source_description: str | None = None,
    ) -> TaskIntent:
        text = " ".join(question.casefold().split())
        organs = self._found_organs(text)
        site = self._found_site(text)
        has_report = bool(source_description and source_description.strip())

        gtv = _has(text, GTV_TERMS) or has_report
        oar = _has(text, OAR_TERMS) or site is not None
        qc = _has(text, QC_STRONG) or (_has(text, QC_VERBS) and _has(text, CONTOUR_NOUNS))
        protocol = _has(text, PROTOCOL_TERMS)

        if qc:
            if has_contours:
                return TaskIntent(
                    intent=Intent.QC, confidence=0.9, targets=organs, site=site,
                    rationale="Explicit QC request with uploaded contours.",
                )
            return TaskIntent(
                intent=Intent.QUESTION, confidence=0.4,
                rationale="QC requested but no contours are uploaded to audit.",
            )

        if gtv and oar and not protocol:
            if has_report:
                return TaskIntent(
                    intent=Intent.GTV, confidence=0.75, targets=organs, site=site,
                    source_description=source_description,
                    rationale="Clinical description present; treated as target contouring.",
                )
            return TaskIntent(
                intent=Intent.QUESTION, confidence=0.4, site=site,
                rationale="Request mixes tumor-target and OAR cues; needs clarification.",
            )

        if gtv:
            return TaskIntent(
                intent=Intent.GTV, confidence=0.8, targets=organs, site=site,
                source_description=source_description or question,
                rationale="Tumor/target language detected.",
            )
        if protocol and not organs:
            return TaskIntent(
                intent=Intent.PROTOCOL, confidence=0.75, site=site,
                rationale="Asks about a protocol or structure list.",
            )
        if oar:
            return TaskIntent(
                intent=Intent.OAR, confidence=0.8, targets=organs, site=site,
                rationale="OAR / radiotherapy context.",
            )
        if organs:
            return TaskIntent(
                intent=Intent.ORGAN, confidence=0.7, targets=organs,
                rationale="Plain anatomical structure(s) named.",
            )
        return TaskIntent(
            intent=Intent.QUESTION, confidence=0.3,
            rationale="No structure, site, or task cue identified.",
        )
