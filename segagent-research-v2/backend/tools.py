from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

import numpy as np

from .expert import SegmentationBackend
from .imaging import load_mask, load_volume, mask_geometry, render_overlays, save_mask
from .knowledge import HybridKnowledgeBase
from .qc import ContourQCEngine
from .schemas import (
    EvidenceRecord,
    EvidenceStatus,
    Finding,
    ProtocolMatch,
    QCReport,
    SegmentRequest,
    SegmentResult,
    StructureMeasurement,
    ToolObservation,
)
from .storage import ResearchStore
from .verifier import GroundingVerifier


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
        case = self.store.get_case(request.case_id)
        image_path = self.store.artifact_path(case.image)
        started = time.perf_counter()
        volume, masks = self.backend.segment(image_path, request.structures)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        measurements: list[StructureMeasurement] = []
        warnings: list[str] = []
        voxel_ml = float(np.prod(volume.spacing) / 1000.0)
        for index, structure in enumerate(request.structures):
            mask = np.asarray(masks[index], dtype=bool)
            voxels, bbox, centroid = mask_geometry(mask)
            if voxels == 0:
                measurements.append(
                    StructureMeasurement(structure=structure, found=False, voxels=0)
                )
                warnings.append(f'No voxels were returned for "{structure}".')
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
                structure,
                "application/gzip",
                {
                    "source_model": self.backend.name,
                    "source_version": self.backend.version,
                    "purpose": request.purpose,
                    "target_type": request.target_type.value,
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
                        f"{structure}, axial z={z}",
                        "image/png",
                        {"structure": structure, "slice_index": z, "mask_id": mask_ref.artifact_id},
                    )
                )
            measurements.append(
                StructureMeasurement(
                    structure=structure,
                    found=True,
                    voxels=voxels,
                    volume_ml=round(voxels * voxel_ml, 3),
                    mean_intensity=round(float(volume.data[mask].mean()), 3),
                    bbox_voxels=bbox,
                    centroid_voxels=centroid,
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

    def measure_edited(
        self, case_id: str, edited_mask_id: str
    ) -> tuple[StructureMeasurement, ToolObservation]:
        """Re-measure a human-edited mask so it grounds like a segmentation.

        The reviewer's edited contour supersedes the model's mask, so its volume
        and intensity must be recomputed from the edited voxels — otherwise the
        answer would cite the model's numbers for a contour the human changed.
        """
        case = self.store.get_case(case_id)
        volume = load_volume(self.store.artifact_path(case.image))
        ref, mask_path = self.store.get_artifact(case_id, edited_mask_id)
        mask = load_mask(mask_path, volume)  # validates grid against the case image
        voxels, bbox, centroid = mask_geometry(mask)
        voxel_ml = float(np.prod(volume.spacing) / 1000.0)
        if voxels == 0:
            measurement = StructureMeasurement(structure=ref.label, found=False, voxels=0)
            summary = f'{ref.label} (human-edited): empty contour.'
        else:
            measurement = StructureMeasurement(
                structure=ref.label,
                found=True,
                voxels=voxels,
                volume_ml=round(voxels * voxel_ml, 3),
                mean_intensity=round(float(volume.data[mask].mean()), 3),
                bbox_voxels=bbox,
                centroid_voxels=centroid,
                mask=ref,
            )
            summary = (
                f'{ref.label} (human-edited): {measurement.voxels} voxels, '
                f'{measurement.volume_ml} mL, mean intensity {measurement.mean_intensity}'
            )
        observation = ToolObservation(
            observation_id=observation_id(),
            tool="segment",
            summary="Human-edited contour evidence:\n- " + summary,
            data={
                "segment_result": {"measurements": [measurement.model_dump(mode="json")]},
                "edited": True,
                "derived_from": ref.metadata.get("derived_from"),
                "target_type": ref.metadata.get("target_type", "unknown"),
            },
            evidence_ids=[ref.artifact_id],
        )
        return measurement, observation


class ContourQCTool:
    name = "run_qc"

    def __init__(self, engine: ContourQCEngine):
        self.engine = engine

    def run(self, case_id: str) -> tuple[QCReport, ToolObservation]:
        report = self.engine.run(case_id)
        lines = [
            f"QC summary: {report.summary.get('ok', 0)} ok, "
            f"{report.summary.get('warn', 0)} review, "
            f"{report.summary.get('error', 0)} error."
        ]
        evidence_ids: list[str] = []
        for row in report.organs:
            details = "; ".join(item.message for item in row.findings) or "no flags"
            lines.append(
                f"{row.organ} [{row.status}], volume={row.volume_ml} mL, "
                f"expert Dice={row.expert_dice}: {details}"
            )
            if row.expert_mask:
                evidence_ids.append(row.expert_mask.artifact_id)
        return report, ToolObservation(
            observation_id=observation_id(),
            tool="run_qc",
            summary="\n".join(lines),
            data={"qc_report": report.model_dump(mode="json")},
            evidence_ids=evidence_ids,
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
            has_reference = any(item.tool in {"lookup_protocol", "run_qc"} for item in observations)
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

    @staticmethod
    def _record(observation: ToolObservation) -> EvidenceRecord:
        return EvidenceRecord(
            record_id=observation.observation_id,
            tool=observation.tool,
            status=EvidenceStatus.ADMITTED,
            summary=observation.summary,
            values=EvidenceCritic._values(observation),
            citations=EvidenceCritic._citations(observation),
        )

    @staticmethod
    def _values(observation: ToolObservation) -> dict[str, float]:
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
            report = data.get("qc_report")
            if isinstance(report, dict):
                for index, organ in enumerate(report.get("organs", []) or []):
                    for key in ("volume_ml", "expert_dice"):
                        value = organ.get(key)
                        if isinstance(value, (int, float)):
                            values[f"qc{index}_{key}"] = float(value)
                    for finding in organ.get("findings", []) or []:
                        for name, value in (finding.get("metrics") or {}).items():
                            if isinstance(value, (int, float)):
                                values[f"qc{index}_{name}"] = float(value)
        except Exception:
            return values
        return values

    @staticmethod
    def _citations(observation: ToolObservation) -> list[str]:
        citations: list[str] = []
        try:
            data = observation.data or {}
            protocol = data.get("protocol")
            if isinstance(protocol, dict):
                citations.extend(str(c) for c in protocol.get("citations", []) or [])
            for hit in data.get("guideline_hits", []) or []:
                citation = hit.get("citation") if isinstance(hit, dict) else None
                if citation:
                    citations.append(str(citation))
        except Exception:
            return citations
        return citations

