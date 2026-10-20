from __future__ import annotations

import re
from typing import Iterable, Literal

from pydantic import BaseModel

from .schemas import EvidenceRecord, MetricEvidence


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

    # Ignore digits embedded in words such as "3D", sequence names, or artifact
    # identifiers; only standalone numeric claims require measurement evidence.
    NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])")
    UNIT_AFTER = re.compile(
        r"^\s*(?P<unit>"
        r"mL|millilit(?:er|re)s?|cc|cm(?:\^?3|³)|"
        r"mm|millimet(?:er|re)s?|cm|centimet(?:er|re)s?|"
        r"%|percent|voxels?|structures?|organs?|contours?|HU"
        r")\b|^\s*(?P<percent>%)",
        re.IGNORECASE,
    )
    # A citation must contain a letter: measurements are legitimately reported as
    # bracketed arrays ("bbox [[240, 244, 81], ...]"), and treating those as
    # fabricated sources blocked otherwise well-grounded answers.
    CITATION = re.compile(r"\[[^\]\n]*[^\W\d_][^\]\n]*\]")

    def __init__(self, rel_tolerance: float = 5e-3):
        self.rel_tolerance = rel_tolerance

    @staticmethod
    def allowed_values(records: Iterable[EvidenceRecord]) -> list[float]:
        values: list[float] = []
        for record in records:
            values.extend(float(v) for v in record.values.values())
            values.extend(float(item.value) for item in record.metrics)
        return values

    @staticmethod
    def _unit(unit: str) -> tuple[str, float]:
        normalized = (unit or "").strip().casefold().replace(" ", "")
        if normalized in {
            "ml", "milliliter", "milliliters", "millilitre", "millilitres",
            "cc", "cm3", "cm^3", "cm³",
        }:
            return "ml", 1.0
        if normalized in {"mm", "millimeter", "millimeters", "millimetre", "millimetres"}:
            return "mm", 1.0
        if normalized in {"cm", "centimeter", "centimeters", "centimetre", "centimetres"}:
            return "mm", 10.0
        if normalized in {"%", "percent"}:
            return "percent", 1.0
        if normalized in {"voxel", "voxels"}:
            return "voxel", 1.0
        if normalized in {"structure", "structures", "organ", "organs", "contour", "contours", "count"}:
            return "count", 1.0
        if normalized == "hu":
            return "hu", 1.0
        if normalized in {"ratio", "unitless", "1"}:
            return "ratio", 1.0
        return normalized, 1.0

    @classmethod
    def _legacy_metric(cls, key: str, value: float) -> MetricEvidence:
        lowered = key.casefold()
        if "hd95" in lowered:
            metric, unit = "hd95", "mm"
        elif "mean_surface" in lowered or "surface_distance" in lowered:
            metric, unit = "mean_surface_distance", "mm"
        elif "volume_difference" in lowered:
            metric, unit = "volume_difference", "%"
        elif "model_volume" in lowered:
            metric, unit = "model_volume", "mL"
        elif "contour_volume" in lowered:
            metric, unit = "contour_volume", "mL"
        elif "volume" in lowered:
            metric, unit = "volume", "mL"
        elif "dice" in lowered:
            metric, unit = "dice", "ratio"
        elif "centroid_mm" in lowered:
            metric, unit = "centroid", "mm"
        elif "centroid" in lowered:
            metric, unit = "centroid", "voxel"
        elif "bbox" in lowered:
            metric, unit = "bbox", "voxel"
        elif "voxel" in lowered:
            metric, unit = "voxel_count", "voxel"
        elif "intensity" in lowered:
            metric, unit = "mean_intensity", "HU"
        elif "count" in lowered:
            metric, unit = lowered, "count"
        else:
            metric, unit = lowered, ""
        return MetricEvidence(
            evidence_id=f"legacy:{key}",
            metric=metric,
            value=float(value),
            unit=unit,
            source="legacy evidence value",
        )

    @classmethod
    def allowed_metrics(cls, records: Iterable[EvidenceRecord]) -> list[MetricEvidence]:
        metrics: list[MetricEvidence] = []
        for record in records:
            metrics.extend(record.metrics)
            represented = {
                (item.metric.casefold(), float(item.value)) for item in record.metrics
            }
            for key, value in record.values.items():
                inferred = cls._legacy_metric(key, float(value))
                if (inferred.metric.casefold(), inferred.value) not in represented:
                    metrics.append(inferred)
        return metrics

    @staticmethod
    def allowed_citations(records: Iterable[EvidenceRecord]) -> set[str]:
        citations: set[str] = set()
        for record in records:
            citations.update(_norm(c) for c in record.citations if c.strip())
        return citations

    def _close(self, value: float, candidate: float) -> bool:
        tolerance = 1e-6 + self.rel_tolerance * abs(candidate)
        return abs(value - candidate) <= tolerance

    def _grounded_number(self, token: str, allowed: list[float]) -> bool:
        try:
            value = float(token)
        except ValueError:
            return True
        for candidate in allowed:
            if self._close(value, candidate):
                return True
        return False

    @staticmethod
    def _claim_metric(context: str, unit: str | None) -> str | None:
        text = context.casefold()
        if "hd95" in text or "hausdorff" in text:
            return "hd95"
        if "mean surface" in text or "average surface" in text:
            return "mean_surface_distance"
        if "surface distance" in text or "distance to" in text:
            return "surface_distance"
        if "dice" in text:
            return "dice"
        if "volume difference" in text or "volume delta" in text:
            return "volume_difference"
        if "model volume" in text or re.search(r"\bmodel\s*[=:]", text):
            return "model_volume"
        if "contour volume" in text or re.search(r"\bcontour\s*[=:]", text):
            return "contour_volume"
        if "volume" in text:
            return "volume"
        if "centroid" in text:
            return "centroid"
        if "bbox" in text or "bounding box" in text:
            return "bbox"
        if "mean intensity" in text:
            return "mean_intensity"
        if unit == "count":
            if "missing" in text:
                return "missing_structure_count"
            if "present" in text:
                return "present_structure_count"
            if "expected" in text:
                return "expected_structure_count"
            return "structure_count"
        if unit == "voxel":
            return "voxel_count"
        return None

    @staticmethod
    def _metric_compatible(claim: str | None, admitted: str) -> bool:
        if claim is None:
            return True
        candidate = admitted.casefold()
        if claim == candidate:
            return True
        if claim == "volume":
            return candidate in {"volume", "contour_volume", "model_volume"}
        if claim == "surface_distance":
            return "surface_distance" in candidate
        if claim == "centroid":
            return candidate.startswith("centroid")
        if claim == "bbox":
            return candidate.startswith("bbox")
        if claim == "structure_count":
            return candidate.endswith("structure_count")
        return False

    def _grounded_claim(
        self,
        token: str,
        claim_unit: str | None,
        claim_metric: str | None,
        metrics: list[MetricEvidence],
        allowed_values: list[float],
    ) -> bool:
        try:
            raw_value = float(token)
        except ValueError:
            return True
        normalized_claim_unit = None
        claim_factor = 1.0
        if claim_unit:
            normalized_claim_unit, claim_factor = self._unit(claim_unit)
        claim_value = raw_value * claim_factor

        for item in metrics:
            admitted_unit, admitted_factor = self._unit(item.unit)
            if normalized_claim_unit and admitted_unit != normalized_claim_unit:
                continue
            if not self._metric_compatible(claim_metric, item.metric):
                continue
            if self._close(claim_value, float(item.value) * admitted_factor):
                return True

        # Unlabelled array coordinates and other legacy prose remain supported,
        # but a claim that names a unit or metric must match that context.
        if normalized_claim_unit is None and claim_metric is None:
            return self._grounded_number(token, allowed_values)
        return False

    def verify(
        self, answer: str, admitted: list[EvidenceRecord]
    ) -> list[GroundingFinding]:
        findings: list[GroundingFinding] = []

        allowed = self.allowed_values(admitted)
        metrics = self.allowed_metrics(admitted)
        unsupported: list[str] = []
        for match in self.NUMBER.finditer(answer):
            token = match.group(0)
            tail = answer[match.end():match.end() + 24]
            unit_match = self.UNIT_AFTER.search(tail)
            raw_unit = None
            if unit_match:
                raw_unit = unit_match.group("unit") or unit_match.group("percent")
            normalized_unit = self._unit(raw_unit)[0] if raw_unit else None
            left = max(answer.rfind(mark, 0, match.start()) for mark in ",;\n")
            right_candidates = [
                index
                for mark in ",;\n"
                if (index := answer.find(mark, match.end())) >= 0
            ]
            right = min(right_candidates) if right_candidates else len(answer)
            context_end = min(right, match.end() + 32) if normalized_unit == "count" else match.end()
            context = answer[max(left + 1, match.start() - 64):context_end]
            metric = self._claim_metric(context, normalized_unit)
            if not self._grounded_claim(token, raw_unit, metric, metrics, allowed):
                unsupported.append(token)
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

        # A purely descriptive answer (visual question answering) legitimately has
        # no tool evidence, so only flag an evidence-free answer when it asserts
        # specifics — numbers or citations — that ought to have come from a tool.
        if answer.strip() and not admitted and (
            self.NUMBER.search(answer) or self.CITATION.search(answer)
        ):
            findings.append(
                GroundingFinding(
                    check="evidence_presence",
                    severity="warn",
                    message="The answer states specific values or sources but no tool evidence was recorded.",
                )
            )
        return findings
