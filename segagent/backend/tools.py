from __future__ import annotations

import json
import math
import re
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .knowledge import HybridKnowledgeBase
from .qc import ContourQCEngine, canonical_roi
from .schemas import (
    EvidenceRecord,
    Finding,
    MetricEvidence,
    ProtocolMatch,
    QCReport,
    SegmentRequest,
    SegmentResult,
    StructureMeasurement,
    ToolObservation,
)
from .storage import ResearchStore
from .verifier import GroundingVerifier

if TYPE_CHECKING:
    from .expert import SegmentationBackend


def observation_id() -> str:
    return f"observation_{uuid.uuid4().hex[:16]}"


class ProtocolLookupTool:
    name = "lookup_protocol"

    def __init__(self, knowledge: HybridKnowledgeBase, retrieve_k: int = 4):
        self.knowledge = knowledge
        self.retrieve_k = retrieve_k

    def run(self, query: str) -> tuple[ProtocolMatch | None, ToolObservation]:
        protocol = self.knowledge.lookup_protocol(query)
        passages = self.knowledge.retrieve_guidelines(query, self.retrieve_k)
        if protocol is None:
            summary = (
                "No protocol matched confidently. Ask for a supported site instead of "
                "inventing an OAR list."
            )
            data = {
                "matched": False,
                "known_sites": [item.get("site") for item in self.knowledge.protocols],
                "guideline_hits": [item.model_dump(mode="json") for item in passages],
            }
        else:
            citations = [*protocol.citations, *(item.citation for item in passages)]
            citation_text = " ".join(dict.fromkeys(citations)) or "[local protocol registry]"
            summary = (
                f'Matched {protocol.site} via {protocol.matched_by}; protocol structures: '
                f'{"; ".join(protocol.oars)}. Sources: {citation_text}'
            )
            data = {
                "matched": True,
                "protocol": protocol.model_dump(mode="json"),
                "site": protocol.site,
                "oars": protocol.oars,
                # Record the citation actually shown in the summary, including
                # the registry fallback, so quoting it back is grounded.
                "citations": list(dict.fromkeys([*citations, citation_text])),
                "guideline_hits": [item.model_dump(mode="json") for item in passages],
            }
        return protocol, ToolObservation(
            observation_id=observation_id(),
            tool="lookup_protocol",
            summary=summary,
            data=data,
        )


class SegmentationTool:
    name = "segment"

    def __init__(
        self,
        store: ResearchStore,
        backend: SegmentationBackend,
        overlay_slices: int = 3,
    ):
        self.store = store
        self.backend = backend
        self.overlay_slices = overlay_slices

    def run(self, request: SegmentRequest) -> tuple[SegmentResult, ToolObservation]:
        from .imaging import mask_geometry, render_overlays, save_mask

        case = self.store.get_case(request.case_id)
        image_path = self.store.artifact_path(case.image)
        started = time.perf_counter()
        volume, masks = self.backend.segment(image_path, request.structures)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        measurements: list[StructureMeasurement] = []
        warnings: list[str] = []
        voxel_ml = float(np.prod(volume.spacing) / 1000.0)
        for index, structure in enumerate(request.structures):
            normalization = (
                request.prompt_normalizations[index]
                if index < len(request.prompt_normalizations)
                else None
            )
            display_structure = (
                normalization.source_text
                if normalization is not None and normalization.exploratory
                else structure
            )
            mask = np.asarray(masks[index], dtype=bool)
            voxels, bbox, centroid = mask_geometry(mask)
            if voxels == 0:
                measurements.append(
                    StructureMeasurement(
                        structure=display_structure,
                        found=False,
                        voxels=0,
                    )
                )
                warnings.append(
                    f'No voxels were returned for "{display_structure}".'
                )
                continue
            artifact_id, mask_path = self.store.allocate_artifact_path(
                request.case_id, ".nii.gz"
            )
            save_mask(mask, volume, mask_path)
            mask_ref = self.store.register_artifact(
                request.case_id,
                artifact_id,
                mask_path,
                "mask",
                display_structure,
                "application/gzip",
                {
                    "source_model": self.backend.name,
                    "source_version": self.backend.version,
                    "artifact_version": 1,
                    "purpose": request.purpose,
                    "target_type": request.target_type.value,
                    "source_prompt": (
                        normalization.source_text if normalization else structure
                    ),
                    "voxtell_prompt": structure,
                    "prompt_category": (
                        normalization.category.value if normalization else None
                    ),
                    "prompt_normalized_by": (
                        normalization.normalized_by if normalization else None
                    ),
                    "prompt_normalization_confidence": (
                        normalization.confidence if normalization else None
                    ),
                    "prompt_fallback_reason": (
                        normalization.fallback_reason if normalization else None
                    ),
                    "prompt_exploratory": (
                        normalization.exploratory if normalization else False
                    ),
                },
            )
            overlays = []
            for z, image in render_overlays(volume.data, mask, self.overlay_slices):
                overlay_id, overlay_path = self.store.allocate_artifact_path(
                    request.case_id, ".png"
                )
                image.save(overlay_path, format="PNG")
                overlays.append(
                    self.store.register_artifact(
                        request.case_id,
                        overlay_id,
                        overlay_path,
                        "overlay",
                        f"{display_structure}, axial z={z}",
                        "image/png",
                        {
                            "structure": display_structure,
                            "voxtell_prompt": structure,
                            "slice_index": z,
                            "mask_id": mask_ref.artifact_id,
                            "prompt_exploratory": (
                                normalization.exploratory
                                if normalization
                                else False
                            ),
                        },
                    )
                )
            measurements.append(
                StructureMeasurement(
                    structure=display_structure,
                    found=True,
                    voxels=voxels,
                    volume_ml=round(voxels * voxel_ml, 3),
                    mean_intensity=round(float(volume.data[mask].mean()), 3),
                    bbox_voxels=bbox,
                    centroid_voxels=centroid,
                    centroid_mm=(
                        np.round(
                            volume.affine
                            @ np.asarray([*centroid, 1.0], dtype=float),
                            2,
                        )[:3].tolist()
                        if centroid is not None
                        else None
                    ),
                    mask=mask_ref,
                    overlays=overlays,
                )
            )
        result = SegmentResult(
            request=request,
            model_name=self.backend.name,
            model_version=self.backend.version,
            elapsed_ms=round(elapsed_ms, 2),
            measurements=measurements,
            warnings=warnings,
        )
        lines: list[str] = []
        evidence_ids: list[str] = []
        for item in measurements:
            if not item.found:
                lines.append(f'{item.structure}: not found (0 voxels)')
                continue
            lines.append(
                f"{item.structure}: {item.voxels} voxels, {item.volume_ml} mL, "
                f"mean intensity {item.mean_intensity}, bbox {item.bbox_voxels}"
            )
            if item.mask:
                evidence_ids.append(item.mask.artifact_id)
            evidence_ids.extend(overlay.artifact_id for overlay in item.overlays)
        summary = "Segmentation evidence from " + self.backend.name + ":\n- " + "\n- ".join(lines)
        observation = ToolObservation(
            observation_id=observation_id(),
            tool="segment",
            summary=summary,
            data={
                "segment_result": result.model_dump(mode="json"),
                "target_type": request.target_type.value,
            },
            evidence_ids=evidence_ids,
        )
        return result, observation


class MaskInspectionTool:
    """Recall an existing contour and re-measure it, instead of re-segmenting.

    This is what lets a follow-up question ("how big is that left eye mask?")
    use the contour already on screen: the numbers are recomputed from the
    stored voxels, so they are real measurements rather than remembered text.
    """

    name = "inspect_mask"

    def __init__(self, store: ResearchStore):
        self.store = store

    def _find(self, case, name: str):
        wanted = " ".join(name.strip().split()).casefold()
        if not wanted:
            return None
        candidates = [
            item
            for item in case.artifacts
            if item.kind in {"mask", "contour"}
        ]
        # Resolve only an exact case-owned id or exact label. Pronouns and
        # partial label guesses are resolved earlier from MessageArtifactContext.
        for item in reversed(candidates):
            if item.artifact_id == name:
                return item
        for item in reversed(candidates):
            if item.label.casefold() == wanted:
                return item
        # A normalized model prompt may deliberately differ from what the user
        # typed (for example, ``eye_L`` becomes ``left eye``). Preserve exact
        # source-prompt lookup so a later turn can still use the user's original
        # name rather than depending on the mask being selected in the viewer.
        for item in reversed(candidates):
            source_prompt = item.metadata.get("source_prompt")
            if (
                isinstance(source_prompt, str)
                and " ".join(source_prompt.strip().split()).casefold() == wanted
            ):
                return item

        # Finally resolve only conservative canonical aliases. ``canonical_roi``
        # normalizes separators and explicit side aliases but intentionally does
        # no substring or fuzzy matching, so ``eye`` cannot select ``left eye``.
        from .qc import canonical_roi

        canonical_wanted = canonical_roi(name)
        if not canonical_wanted:
            return None
        for item in reversed(candidates):
            aliases = [
                item.label,
                item.metadata.get("source_prompt"),
                item.metadata.get("voxtell_prompt"),
            ]
            if any(
                isinstance(alias, str)
                and alias.strip()
                and canonical_roi(alias) == canonical_wanted
                for alias in aliases
            ):
                return item
        return None

    def run(self, case_id: str, structures: list[str]) -> tuple[list[StructureMeasurement], ToolObservation]:
        from .imaging import load_mask, load_volume, mask_geometry

        case = self.store.get_case(case_id)
        volume = load_volume(self.store.artifact_path(case.image))
        voxel_ml = float(np.prod(volume.spacing) / 1000.0)
        measurements: list[StructureMeasurement] = []
        lines: list[str] = []
        evidence_ids: list[str] = []
        missing: list[str] = []

        for name in structures:
            ref = self._find(case, name)
            if ref is None:
                missing.append(name)
                lines.append(f'{name}: no contour with that name exists in this case')
                continue
            try:
                mask = load_mask(self.store.artifact_path(ref), volume)
            except Exception as exc:
                lines.append(f"{ref.label}: could not be read ({exc})")
                continue
            voxels, bbox, centroid = mask_geometry(mask)
            if voxels == 0:
                measurements.append(
                    StructureMeasurement(structure=ref.label, found=False, voxels=0)
                )
                lines.append(f"{ref.label}: empty contour (0 voxels)")
                continue
            measurement = StructureMeasurement(
                structure=ref.label,
                found=True,
                voxels=voxels,
                volume_ml=round(voxels * voxel_ml, 3),
                mean_intensity=round(float(volume.data[mask].mean()), 3),
                bbox_voxels=bbox,
                centroid_voxels=centroid,
                centroid_mm=(
                    np.round(
                        volume.affine
                        @ np.asarray([*centroid, 1.0], dtype=float),
                        2,
                    )[:3].tolist()
                    if centroid is not None
                    else None
                ),
                mask=ref,
            )
            measurements.append(measurement)
            evidence_ids.append(ref.artifact_id)
            source = ref.metadata.get("source_model") or (
                "uploaded" if ref.kind == "contour" else "unknown"
            )
            lines.append(
                f"{ref.label} [{source}]: {measurement.voxels} voxels, "
                f"{measurement.volume_ml} mL, mean intensity {measurement.mean_intensity}, "
                f"bbox {measurement.bbox_voxels}, voxel centroid "
                f"{measurement.centroid_voxels}, world centroid "
                f"{measurement.centroid_mm} mm"
            )

        summary = "Existing contour measurements:\n- " + "\n- ".join(lines or ["nothing to inspect"])
        observation = ToolObservation(
            observation_id=observation_id(),
            tool="inspect_mask",
            summary=summary,
            data={
                "segment_result": {
                    "measurements": [item.model_dump(mode="json") for item in measurements]
                },
                "requested": list(structures),
                "missing": missing,
            },
            evidence_ids=evidence_ids,
        )
        return measurements, observation


class ContourQCTool:
    """Turn a QC report into an observation the planner can narrate.

    The summary states measured disagreement and protocol set differences only;
    it never asserts that a contour or a plan is wrong, because an anatomy model
    is a second opinion rather than ground truth.
    """

    name = "run_qc"

    def __init__(self, engine: ContourQCEngine):
        self.engine = engine

    def run(self, case_id: str, site_query: str | None = None) -> tuple[QCReport, ToolObservation]:
        report = self.engine.run(case_id, site_query)
        lines: list[str] = []
        evidence_ids: list[str] = []

        if report.comparisons:
            lines.append(
                f"Comparison with an independent model contour "
                f"({report.summary.get('ok', 0)} close, {report.summary.get('warn', 0)} moderate, "
                f"{report.summary.get('error', 0)} large disagreement):"
            )
            for item in report.comparisons:
                lines.append(
                    f"{item.structure} [{item.status}]: Dice={item.dice}, "
                    f"contour={item.contour_volume_ml} mL, model={item.model_volume_ml} mL, "
                    f"volume difference={item.volume_difference_pct}%, "
                    f"mean surface={item.mean_surface_mm} mm, HD95={item.hd95_mm} mm. {item.note}"
                )
        if report.targets:
            lines.append(
                "Target volumes are not automatically assessed against the anatomy model; "
                "their uploaded geometry is described only:"
            )
            for item in report.targets:
                near = ", ".join(item.near) if item.near else "no contoured organs nearby"
                lines.append(
                    f"{item.structure} [{item.assessment_status}]: {item.volume_ml} mL, "
                    f"world centroid {item.centroid_mm} mm, closest contoured "
                    f"structures by surface distance: {near}"
                )
        if report.auxiliaries:
            lines.append(
                "Non-organ auxiliary structures were identified and excluded from "
                "anatomy-model comparison:"
            )
            for item in report.auxiliaries:
                volume = (
                    f"{item.volume_ml} mL"
                    if item.volume_ml is not None
                    else "volume unavailable"
                )
                lines.append(
                    f"{item.structure} [auxiliary/{item.auxiliary_class}]: {volume}. "
                    f"Classification source: {item.classification_source}."
                )
        if report.unclassified:
            lines.append(
                "Structures with no safe role mapping were measured but not sent to "
                "the anatomy comparison model:"
            )
            for item in report.unclassified:
                volume = (
                    f"{item.volume_ml} mL"
                    if item.volume_ml is not None
                    else "volume unavailable"
                )
                lines.append(f"{item.structure} [unclassified]: {volume}. {item.note}")
        if report.completeness:
            completeness = report.completeness
            applicability = completeness.applicability
            if applicability and applicability.status == "not_assessed":
                lines.append(f"Protocol completeness not assessed: {applicability.reason}")
            else:
                missing = ", ".join(completeness.missing) if completeness.missing else "none"
                source = " ".join(completeness.citations) or completeness.source or "protocol registry"
                lines.append(
                    f"Protocol structure list for {completeness.site}: "
                    f"structures with no exact canonical contour match: {missing}. "
                    f"Applicability: {applicability.reason if applicability else 'not recorded'} "
                    f"Source: {source}"
                )
        for warning in report.warnings:
            lines.append(f"Warning: {warning}")
        for item in report.metric_evidence:
            evidence_ids.extend(item.artifact_ids)

        return report, ToolObservation(
            observation_id=observation_id(),
            tool="run_qc",
            summary="\n- ".join(["Contour QC evidence:", *lines]) if lines else "No QC evidence.",
            data={"qc_report": report.model_dump(mode="json")},
            evidence_ids=list(dict.fromkeys(evidence_ids)),
        )


class EvidenceCritic:
    """Deterministic final-answer guard; it never invents replacement evidence.

    Numbers are grounded against the quantities tools actually emitted, and
    citations against retrieved sources, via GroundingVerifier — not against a
    serialized data blob, which the previous substring check did and which let
    almost any short fabricated number pass by matching a hash or id.
    """

    CLINICAL_JUDGMENTS = re.compile(
        r"\b(normal|abnormal|enlarged|atrophic|malignant|benign|diagnosis|disease)\b",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        self._verifier = GroundingVerifier()

    def review(self, answer: str, observations: list[ToolObservation]) -> list[Finding]:
        records = [self._record(item) for item in observations]
        findings: list[Finding] = [
            Finding(check=item.check, severity=item.severity, message=item.message)
            for item in self._verifier.verify(answer, records)
        ]
        if self.CLINICAL_JUDGMENTS.search(answer):
            has_reference = any(item.tool == "lookup_protocol" for item in observations)
            if not has_reference:
                findings.append(
                    Finding(
                        check="clinical_grounding",
                        severity="warn",
                        message=(
                            "A clinical judgment appears without protocol/QC evidence; "
                            "phrase it as a limitation or request expert review."
                        ),
                    )
                )
        return findings

    def safe_answer(
        self,
        candidate: str,
        observations: list[ToolObservation],
        deterministic_fallback: str,
        completion_fallback: str,
    ) -> tuple[str, list[Finding], bool]:
        """Select a verified answer without exposing a critic rejection page."""

        findings = self.review(candidate, observations)
        if not any(item.severity == "error" for item in findings):
            return candidate, findings, False
        for fallback in (
            deterministic_fallback,
            completion_fallback,
            "Analysis completed.",
        ):
            if fallback.strip() and not any(
                item.severity == "error"
                for item in self.review(fallback, observations)
            ):
                return fallback, findings, True
        # The final literal contains no numeric or citation claims and is kept as
        # a total-function guard if verifier rules expand in the future.
        return "Analysis completed.", findings, True

    @staticmethod
    def _record(observation: ToolObservation) -> EvidenceRecord:
        metrics = EvidenceCritic._metrics(observation)
        return EvidenceRecord(
            record_id=observation.observation_id,
            tool=observation.tool,
            summary=observation.summary,
            values=EvidenceCritic._values(observation, metrics),
            metrics=metrics,
            citations=EvidenceCritic._citations(observation),
            artifact_ids=list(observation.evidence_ids),
        )

    @staticmethod
    def _values(
        observation: ToolObservation,
        admitted_metrics: list[MetricEvidence] | None = None,
    ) -> dict[str, float]:
        """Pull the numeric quantities a tool emitted from its structured data."""
        values: dict[str, float] = {}
        try:
            data = observation.data or {}
            segment = data.get("segment_result")
            if isinstance(segment, dict):
                for index, item in enumerate(segment.get("measurements", []) or []):
                    for key in ("voxels", "volume_ml", "mean_intensity"):
                        value = item.get(key)
                        if isinstance(value, (int, float)):
                            values[f"seg{index}_{key}"] = float(value)
                    for c_index, corner in enumerate(item.get("bbox_voxels") or []):
                        for axis, coordinate in enumerate(corner or []):
                            if isinstance(coordinate, (int, float)):
                                values[f"seg{index}_bbox{c_index}_{axis}"] = float(coordinate)
                    for axis, coordinate in enumerate(item.get("centroid_voxels") or []):
                        if isinstance(coordinate, (int, float)):
                            values[f"seg{index}_centroid_{axis}"] = float(coordinate)
                    for axis, coordinate in enumerate(item.get("centroid_mm") or []):
                        if isinstance(coordinate, (int, float)):
                            values[f"seg{index}_centroid_mm_{axis}"] = float(coordinate)
            qc_report = data.get("qc_report")
            if isinstance(qc_report, dict):
                for index, item in enumerate(qc_report.get("metric_evidence", []) or []):
                    value = item.get("value") if isinstance(item, dict) else None
                    metric = item.get("metric") if isinstance(item, dict) else None
                    if isinstance(value, (int, float)) and metric:
                        values[f"qc{index}_{metric}"] = float(value)
            report_evidence = data.get("report_evidence")
            if isinstance(report_evidence, dict):
                for index, item in enumerate(
                    report_evidence.get("metric_evidence", []) or []
                ):
                    value = item.get("value") if isinstance(item, dict) else None
                    metric = item.get("metric") if isinstance(item, dict) else None
                    if isinstance(value, (int, float)) and metric:
                        values[f"report{index}_{metric}"] = float(value)
        except Exception:
            pass
        for index, item in enumerate(
            admitted_metrics
            if admitted_metrics is not None
            else EvidenceCritic._metrics(observation)
        ):
            values[f"metric{index}_{item.metric}"] = float(item.value)
        return values

    @staticmethod
    def _finite_float(value) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None

    @staticmethod
    def _metric_key(item: MetricEvidence) -> tuple[str, str]:
        return canonical_roi(item.structure or ""), item.metric.casefold()

    @classmethod
    def _legacy_qc_metrics(
        cls,
        observation: ToolObservation,
        metrics: list[MetricEvidence],
    ) -> list[MetricEvidence]:
        """Read-time migration for QC observations saved before typed evidence.

        Only explicit, whitelisted QC fields are admitted.  Natural-language
        summaries and arbitrary JSON numbers remain outside the trust boundary.
        Existing typed metrics always win over legacy values.
        """

        data = observation.data or {}
        report = data.get("qc_report")
        if not isinstance(report, dict):
            return metrics
        admitted = {cls._metric_key(item) for item in metrics}

        def artifact_ids(*values) -> list[str]:
            flattened: list[str] = list(observation.evidence_ids)
            for value in values:
                if isinstance(value, str) and value:
                    flattened.append(value)
                elif isinstance(value, dict):
                    artifact_id = value.get("artifact_id")
                    if artifact_id:
                        flattened.append(str(artifact_id))
                elif isinstance(value, list):
                    for nested in value:
                        if isinstance(nested, dict) and nested.get("artifact_id"):
                            flattened.append(str(nested["artifact_id"]))
                        elif isinstance(nested, str) and nested:
                            flattened.append(nested)
            return list(dict.fromkeys(flattened))

        def add(
            *,
            evidence_id: str,
            structure: str | None,
            metric: str,
            value,
            unit: str,
            source: str,
            artifacts=(),
        ) -> None:
            parsed = cls._finite_float(value)
            key = (canonical_roi(structure or ""), metric.casefold())
            if parsed is None or key in admitted:
                return
            metrics.append(
                MetricEvidence(
                    evidence_id=evidence_id,
                    structure=structure or None,
                    canonical_roi=canonical_roi(structure or "") or None,
                    metric=metric,
                    value=parsed,
                    unit=unit,
                    source=source,
                    artifact_ids=artifact_ids(*artifacts),
                )
            )
            admitted.add(key)

        comparison_fields = (
            ("dice", "dice", "ratio"),
            ("contour_volume_ml", "contour_volume", "mL"),
            ("model_volume_ml", "model_volume", "mL"),
            ("volume_difference_pct", "volume_difference", "%"),
            ("mean_surface_mm", "mean_surface_distance", "mm"),
            ("hd95_mm", "hd95", "mm"),
        )
        for index, item in enumerate(report.get("comparisons", []) or []):
            if not isinstance(item, dict):
                continue
            structure = str(item.get("structure") or "")
            artifacts = (
                item.get("contour_artifact_id"),
                item.get("model_mask"),
                item.get("difference_overlays") or [],
            )
            for field, metric, unit in comparison_fields:
                add(
                    evidence_id=(
                        f"{observation.observation_id}:legacy:comparison:"
                        f"{index}:{metric}"
                    ),
                    structure=structure,
                    metric=metric,
                    value=item.get(field),
                    unit=unit,
                    source="legacy structured QC comparison",
                    artifacts=artifacts,
                )

        for index, item in enumerate(report.get("targets", []) or []):
            if not isinstance(item, dict):
                continue
            structure = str(item.get("structure") or "")
            artifacts = (item.get("artifact_id"),)
            add(
                evidence_id=f"{observation.observation_id}:legacy:target:{index}:volume",
                structure=structure,
                metric="volume",
                value=item.get("volume_ml"),
                unit="mL",
                source="legacy uploaded target contour",
                artifacts=artifacts,
            )
            for axis, value in zip(("x", "y", "z"), item.get("centroid_mm") or []):
                add(
                    evidence_id=(
                        f"{observation.observation_id}:legacy:target:"
                        f"{index}:centroid_{axis}"
                    ),
                    structure=structure,
                    metric=f"centroid_{axis}",
                    value=value,
                    unit="mm",
                    source="legacy NIfTI world geometry",
                    artifacts=artifacts,
                )
            for nearby, value in (item.get("near_surface_mm") or {}).items():
                add(
                    evidence_id=(
                        f"{observation.observation_id}:legacy:target:{index}:"
                        f"surface_distance:{canonical_roi(str(nearby))}"
                    ),
                    structure=structure,
                    metric=f"surface_distance_to:{canonical_roi(str(nearby))}",
                    value=value,
                    unit="mm",
                    source="legacy physical-space mask geometry",
                    artifacts=artifacts,
                )

        for collection, source in (
            ("auxiliaries", "legacy uploaded auxiliary contour"),
            ("unclassified", "legacy uploaded unclassified contour"),
        ):
            for index, item in enumerate(report.get(collection, []) or []):
                if not isinstance(item, dict):
                    continue
                structure = str(item.get("structure") or "")
                add(
                    evidence_id=(
                        f"{observation.observation_id}:legacy:{collection}:"
                        f"{index}:volume"
                    ),
                    structure=structure,
                    metric="volume",
                    value=item.get("volume_ml"),
                    unit="mL",
                    source=source,
                    artifacts=(item.get("artifact_id"),),
                )

        for name, value in (report.get("summary") or {}).items():
            if str(name) not in {
                "ok",
                "warn",
                "error",
                "target",
                "auxiliary",
                "unclassified",
            }:
                continue
            add(
                evidence_id=(
                    f"{observation.observation_id}:legacy:summary:{name}_count"
                ),
                structure=None,
                metric=f"{name}_structure_count",
                value=value,
                unit="count",
                source="structured QC summary",
            )

        completeness = report.get("completeness")
        if isinstance(completeness, dict):
            for name in ("expected", "present", "missing"):
                items = completeness.get(name)
                if isinstance(items, list):
                    add(
                        evidence_id=(
                            f"{observation.observation_id}:legacy:protocol:"
                            f"{name}_structure_count"
                        ),
                        structure=None,
                        metric=f"{name}_structure_count",
                        value=len(items),
                        unit="count",
                        source=str(
                            completeness.get("source") or "protocol registry"
                        ),
                    )
        return metrics

    @classmethod
    def _metrics(cls, observation: ToolObservation) -> list[MetricEvidence]:
        """Preserve metric, unit, source, and artifacts for contextual checking."""
        metrics: list[MetricEvidence] = []
        data = observation.data or {}
        qc_report = data.get("qc_report")
        if isinstance(qc_report, dict):
            for item in qc_report.get("metric_evidence", []) or []:
                if not isinstance(item, dict):
                    continue
                try:
                    metrics.append(MetricEvidence.model_validate(item))
                except (TypeError, ValueError):
                    continue
        report_evidence = data.get("report_evidence")
        if isinstance(report_evidence, dict):
            for item in report_evidence.get("metric_evidence", []) or []:
                if not isinstance(item, dict):
                    continue
                try:
                    metrics.append(MetricEvidence.model_validate(item))
                except (TypeError, ValueError):
                    continue

        try:
            segment = data.get("segment_result")
            if isinstance(segment, dict):
                model_name = str(segment.get("model_name") or observation.tool)
                for index, item in enumerate(segment.get("measurements", []) or []):
                    if not isinstance(item, dict):
                        continue
                    structure = str(item.get("structure") or "")
                    mask = item.get("mask") if isinstance(item.get("mask"), dict) else {}
                    artifact_ids = [
                        value
                        for value in [mask.get("artifact_id"), *observation.evidence_ids]
                        if value
                    ]
                    for metric, key, unit in (
                        ("voxel_count", "voxels", "voxel"),
                        ("volume", "volume_ml", "mL"),
                        ("mean_intensity", "mean_intensity", "HU"),
                    ):
                        value = item.get(key)
                        if isinstance(value, (int, float)):
                            metrics.append(
                                MetricEvidence(
                                    evidence_id=(
                                        f"{observation.observation_id}:seg{index}:{metric}"
                                    ),
                                    structure=structure or None,
                                    metric=metric,
                                    value=float(value),
                                    unit=unit,
                                    source=model_name,
                                    artifact_ids=list(dict.fromkeys(artifact_ids)),
                                )
                            )
                    for axis, value in zip(
                        ("x", "y", "z"), item.get("centroid_voxels") or []
                    ):
                        if isinstance(value, (int, float)):
                            metrics.append(
                                MetricEvidence(
                                    evidence_id=(
                                        f"{observation.observation_id}:seg{index}:"
                                        f"centroid_{axis}"
                                    ),
                                    structure=structure or None,
                                    metric=f"centroid_{axis}",
                                    value=float(value),
                                    unit="voxel",
                                    source=model_name,
                                    artifact_ids=list(dict.fromkeys(artifact_ids)),
                                )
                            )
                    for axis, value in zip(
                        ("x", "y", "z"), item.get("centroid_mm") or []
                    ):
                        if isinstance(value, (int, float)):
                            metrics.append(
                                MetricEvidence(
                                    evidence_id=(
                                        f"{observation.observation_id}:seg{index}:"
                                        f"centroid_mm_{axis}"
                                    ),
                                    structure=structure or None,
                                    metric=f"centroid_{axis}",
                                    value=float(value),
                                    unit="mm",
                                    source="NIfTI world geometry",
                                    artifact_ids=list(dict.fromkeys(artifact_ids)),
                                )
                            )
        except (TypeError, ValueError):
            pass
        return cls._legacy_qc_metrics(observation, metrics)

    @staticmethod
    def _citations(observation: ToolObservation) -> list[str]:
        citations: list[str] = []
        try:
            data = observation.data or {}
            citations.extend(str(c) for c in data.get("citations", []) or [])
            protocol = data.get("protocol")
            if isinstance(protocol, dict):
                citations.extend(str(c) for c in protocol.get("citations", []) or [])
            qc_report = data.get("qc_report")
            if isinstance(qc_report, dict):
                completeness = qc_report.get("completeness")
                if isinstance(completeness, dict):
                    citations.extend(
                        str(c) for c in completeness.get("citations", []) or []
                    )
                    source = completeness.get("source")
                    if source:
                        citations.append(str(source))
            for hit in data.get("guideline_hits", []) or []:
                citation = hit.get("citation") if isinstance(hit, dict) else None
                if citation:
                    citations.append(str(citation))
        except Exception:
            return citations
        return citations
