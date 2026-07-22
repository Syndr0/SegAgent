from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion, binary_fill_holes, distance_transform_edt, label

from .expert import SegmentationBackend
from .imaging import VolumeData, load_mask, load_volume, save_mask
from .schemas import ArtifactRef, Finding, QCOrganResult, QCReport
from .storage import ResearchStore


SEVERITY = {"ok": 0, "warn": 1, "error": 2}


def dice(a: np.ndarray, b: np.ndarray) -> float:
    left, right = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    total = int(left.sum() + right.sum())
    if total == 0:
        return 1.0
    return 2.0 * float(np.logical_and(left, right).sum()) / total


def surface_metrics(
    a: np.ndarray, b: np.ndarray, spacing: tuple[float, float, float]
) -> tuple[float | None, float | None]:
    left, right = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if not left.any() or not right.any():
        return None, None
    left_surface = left & ~binary_erosion(left)
    right_surface = right & ~binary_erosion(right)
    to_right = distance_transform_edt(~right_surface, sampling=spacing)[left_surface]
    to_left = distance_transform_edt(~left_surface, sampling=spacing)[right_surface]
    distances = np.concatenate([to_right, to_left])
    return float(distances.mean()), float(np.percentile(distances, 95))


class ContourQCEngine:
    def __init__(
        self,
        store: ResearchStore,
        backend: SegmentationBackend,
        references_path: Path,
        with_surface: bool = False,
    ):
        self.store = store
        self.backend = backend
        payload = json.loads(Path(references_path).read_text(encoding="utf-8"))
        self.references = {
            self._normal(name): data for name, data in payload.get("organs", {}).items()
        }
        self.with_surface = with_surface

    @staticmethod
    def _normal(value: str) -> str:
        return " ".join(value.casefold().replace("_", " ").replace("-", " ").split())

    @staticmethod
    def _status(findings: list[Finding]) -> str:
        return max((item.severity for item in findings), key=SEVERITY.get, default="ok")

    def run(self, case_id: str) -> QCReport:
        case = self.store.get_case(case_id)
        if not case.contours:
            return QCReport(
                case_id=case_id,
                organs=[],
                summary={"ok": 0, "warn": 0, "error": 0},
                warnings=["No uploaded contours are registered for this case."],
            )
        image_path = self.store.artifact_path(case.image)
        reference_image = load_volume(image_path)
        names = [item.label for item in case.contours]
        volume, expert_masks = self.backend.segment(image_path, names)
        if volume.data.shape != reference_image.data.shape or not np.allclose(
            volume.affine, reference_image.affine, atol=1e-3
        ):
            raise ValueError("expert image grid does not match the stored case")
        rows: list[QCOrganResult] = []
        warnings: list[str] = []
        for index, contour in enumerate(case.contours):
            try:
                user_mask = load_mask(self.store.artifact_path(contour), reference_image)
            except Exception as exc:
                rows.append(
                    QCOrganResult(
                        organ=contour.label,
                        status="error",
                        findings=[
                            Finding(
                                check="geometry",
                                severity="error",
                                message=f"Contour grid validation failed: {exc}",
                            )
                        ],
                    )
                )
                continue
            expert_mask = np.asarray(expert_masks[index], dtype=bool)
            findings = self._geometry_findings(
                contour.label, user_mask, reference_image, self.references.get(self._normal(contour.label))
            )
            expert_ref: ArtifactRef | None = None
            expert_dice: float | None = None
            if expert_mask.any():
                expert_dice = round(dice(user_mask, expert_mask), 3)
                mean_surface = hd95 = None
                if self.with_surface:
                    mean_surface, hd95 = surface_metrics(
                        user_mask, expert_mask, reference_image.spacing
                    )
                severity = "ok" if expert_dice >= 0.70 else "warn" if expert_dice >= 0.50 else "error"
                findings.append(
                    Finding(
                        check="expert_agreement",
                        severity=severity,
                        message=(
                            f"Contour differs from the expert model with Dice {expert_dice}; "
                            "treat the expert as a reference, not ground truth."
                        ),
                        metrics={
                            "dice": expert_dice,
                            "mean_surface_distance_mm": (
                                round(mean_surface, 2) if mean_surface is not None else None
                            ),
                            "hausdorff95_mm": round(hd95, 2) if hd95 is not None else None,
                        },
                    )
                )
                artifact_id, path = self.store.allocate_artifact_path(case_id, ".nii.gz")
                save_mask(expert_mask, reference_image, path)
                expert_ref = self.store.register_artifact(
                    case_id,
                    artifact_id,
                    path,
                    "mask",
                    f"{contour.label} (expert reference)",
                    "application/gzip",
                    {"source_model": self.backend.name, "source_version": self.backend.version},
                )
            else:
                warnings.append(f'Expert model returned an empty mask for "{contour.label}".')
            volume_ml = float(user_mask.sum() * np.prod(reference_image.spacing) / 1000.0)
            rows.append(
                QCOrganResult(
                    organ=contour.label,
                    status=self._status(findings),
                    volume_ml=round(volume_ml, 2),
                    expert_dice=expert_dice,
                    findings=findings,
                    expert_mask=expert_ref,
                )
            )
        summary = {status: sum(row.status == status for row in rows) for status in SEVERITY}
        return QCReport(case_id=case_id, organs=rows, summary=summary, warnings=warnings)

    def _geometry_findings(
        self, organ: str, mask: np.ndarray, volume: VolumeData, reference: dict[str, Any] | None
    ) -> list[Finding]:
        findings: list[Finding] = []
        voxels = int(mask.sum())
        if voxels == 0:
            return [Finding(check="empty", severity="error", message="Contour is empty.")]
        if voxels < 10:
            findings.append(
                Finding(
                    check="empty",
                    severity="warn",
                    message=f"Contour is nearly empty ({voxels} voxels).",
                )
            )
        reference = reference or {}
        volume_ml = float(voxels * np.prod(volume.spacing) / 1000.0)
        expected_range = reference.get("volume_ml")
        if expected_range:
            lower, upper = expected_range
            if volume_ml < lower or volume_ml > upper:
                severe = volume_ml < lower * 0.5 or volume_ml > upper * 2.0
                findings.append(
                    Finding(
                        check="volume",
                        severity="error" if severe else "warn",
                        message=(
                            f"Volume {volume_ml:.1f} mL is outside the reference interval "
                            f"[{lower}, {upper}] mL."
                        ),
                        metrics={"volume_ml": round(volume_ml, 2), "reference_ml": expected_range},
                    )
                )
        labels, component_count = label(mask)
        expected_components = reference.get("expected_components")
        if expected_components is not None and component_count > expected_components:
            sizes = np.bincount(labels.ravel())[1:]
            significant = int((sizes >= max(int(sizes.max() * 0.05), 10)).sum()) if sizes.size else 0
            if significant > expected_components:
                findings.append(
                    Finding(
                        check="components",
                        severity="warn",
                        message=f"Found {significant} significant connected components.",
                        metrics={"components": int(component_count), "significant": significant},
                    )
                )
        hole_voxels = int(binary_fill_holes(mask).sum() - voxels)
        if hole_voxels > 0.03 * voxels:
            findings.append(
                Finding(
                    check="holes",
                    severity="warn",
                    message=f"Interior holes occupy {100.0 * hole_voxels / voxels:.1f}% of the contour.",
                    metrics={"hole_voxels": hole_voxels},
                )
            )
        laterality = reference.get("laterality")
        if laterality in {"left", "right"}:
            center_x = float(np.argwhere(mask)[:, 0].mean())
            is_left = center_x < mask.shape[0] / 2.0
            if is_left != (laterality == "left"):
                findings.append(
                    Finding(
                        check="laterality",
                        severity="error",
                        message=f'Centroid location conflicts with the "{laterality}" label.',
                    )
                )
        return findings

