from __future__ import annotations

import re
from typing import Iterable, Literal

from pydantic import BaseModel

from .schemas import EvidenceRecord


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


class GroundingFinding(BaseModel):
    check: str
    severity: Literal["warn", "error"]
    message: str


class GroundingVerifier:
    """Deterministic answer guard.

    Numbers in the answer are grounded against the numeric quantities tools
    actually emitted (``EvidenceRecord.values``), compared numerically with a
    tolerance — not substring-matched against a serialized data blob, which the
    v2 critic did and which let almost any fabricated integer pass. Citations
    are grounded against the sources of admitted retrieval evidence.
    """

    NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
    CITATION = re.compile(r"\[[^\]\n]+\]")

    def __init__(self, rel_tolerance: float = 5e-3):
        self.rel_tolerance = rel_tolerance

    @staticmethod
    def allowed_values(records: Iterable[EvidenceRecord]) -> list[float]:
        values: list[float] = []
        for record in records:
            values.extend(float(v) for v in record.values.values())
        return values

    @staticmethod
    def allowed_citations(records: Iterable[EvidenceRecord]) -> set[str]:
        citations: set[str] = set()
        for record in records:
            citations.update(_norm(c) for c in record.citations if c.strip())
        return citations

    def _grounded_number(self, token: str, allowed: list[float]) -> bool:
        try:
            value = float(token)
        except ValueError:
            return True
        for candidate in allowed:
            tolerance = 1e-6 + self.rel_tolerance * abs(candidate)
            if abs(value - candidate) <= tolerance:
                return True
        return False

    def verify(
        self, answer: str, admitted: list[EvidenceRecord]
    ) -> list[GroundingFinding]:
        findings: list[GroundingFinding] = []

        allowed = self.allowed_values(admitted)
        unsupported = [
            token for token in self.NUMBER.findall(answer)
            if not self._grounded_number(token, allowed)
        ]
        if unsupported:
            distinct = list(dict.fromkeys(unsupported))
            findings.append(
                GroundingFinding(
                    check="numeric_grounding",
                    severity="error",
                    message=(
                        "The answer contains numbers absent from tool evidence: "
                        + ", ".join(distinct[:8])
                    ),
                )
            )

        allowed_citations = self.allowed_citations(admitted)
        unsupported_citations = []
        for token in self.CITATION.findall(answer):
            normalized = _norm(token.strip("[]"))
            if not any(
                normalized == source or normalized in source or source in normalized
                for source in allowed_citations
            ):
                unsupported_citations.append(token)
        if unsupported_citations:
            distinct = list(dict.fromkeys(unsupported_citations))
            findings.append(
                GroundingFinding(
                    check="citation_grounding",
                    severity="error",
                    message=(
                        "The answer cites sources not in the retrieved evidence: "
                        + ", ".join(distinct[:8])
                    ),
                )
            )

        if answer.strip() and not admitted:
            findings.append(
                GroundingFinding(
                    check="evidence_presence",
                    severity="warn",
                    message="The answer was produced without any admitted evidence.",
                )
            )
        return findings
