from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

import numpy as np

from .expert import SegmentationBackend
from .imaging import mask_geometry, render_overlays, save_mask
from .knowledge import HybridKnowledgeBase
from .qc import ContourQCEngine
from .schemas import (
    Finding,
    ProtocolMatch,
    QCReport,
    SegmentRequest,
    SegmentResult,
    StructureMeasurement,
    ToolObservation,
)
from .storage import ResearchStore


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
            data={"segment_result": result.model_dump(mode="json")},
            evidence_ids=evidence_ids,
        )
        return result, observation


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
    """Deterministic final-answer guard; it never invents replacement evidence."""

    CLINICAL_JUDGMENTS = re.compile(
        r"\b(normal|abnormal|enlarged|atrophic|malignant|benign|diagnosis|disease)\b",
        re.IGNORECASE,
    )
    NUMBER = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?")

    def review(self, answer: str, observations: list[ToolObservation]) -> list[Finding]:
        evidence_text = "\n".join(
            [item.summary for item in observations]
            + [json.dumps(item.data, ensure_ascii=False, default=str) for item in observations]
        )
        findings: list[Finding] = []
        unsupported_numbers = [
            number for number in self.NUMBER.findall(answer) if number not in evidence_text
        ]
        if unsupported_numbers:
            findings.append(
                Finding(
                    check="numeric_grounding",
                    severity="error",
                    message=(
                        "The proposed answer contains numbers absent from tool evidence: "
                        + ", ".join(unsupported_numbers[:8])
                    ),
                )
            )
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
        if not observations and answer.strip():
            findings.append(
                Finding(
                    check="evidence_presence",
                    severity="warn",
                    message="The answer was produced without a tool observation.",
                )
            )
        return findings

