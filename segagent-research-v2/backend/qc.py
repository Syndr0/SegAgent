from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # imaging pulls in nibabel; keep this module importable without it
    from .expert import SegmentationBackend
    from .knowledge import HybridKnowledgeBase

from .schemas import (
    QCCompleteness,
    QCComparison,
    QCReport,
    QCTargetNote,
)
from .storage import ResearchStore


# Target volumes are clinical constructs: an anatomy model cannot confirm or
# refute them, so they are described rather than compared.
# Letter lookarounds rather than \b, because structure sets are routinely named
# "CTV_high" or "PTV60" and an underscore or digit is a word character.
TARGET_PATTERN = re.compile(
    r"(?<![a-z])(gtv|ctv|ptv|itv|igtv|boost)(?![a-z])|靶区|肿瘤区|临床靶区|计划靶区",
    re.IGNORECASE,
)

# Dice thresholds for flagging disagreement between an uploaded contour and an
# independent model contour. Disagreement means "worth a human look", never
# "the contour is wrong" — the model is a second opinion, not ground truth.
DICE_WARN = 0.80
DICE_ERROR = 0.60


def is_target(label: str) -> bool:
    return bool(TARGET_PATTERN.search(label or ""))


def dice(a: np.ndarray, b: np.ndarray) -> float:
    left, right = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    total = int(left.sum() + right.sum())
    if total == 0:
        return 1.0
    return 2.0 * float(np.logical_and(left, right).sum()) / total


def surface_metrics(
    a: np.ndarray, b: np.ndarray, spacing: tuple[float, float, float]
) -> tuple[float | None, float | None]:
    """Mean and 95th-percentile symmetric surface distance, in millimetres."""
    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt
    except ImportError:
        return None, None
    left, right = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if not left.any() or not right.any():
        return None, None
    left_surface = left & ~binary_erosion(left)
    right_surface = right & ~binary_erosion(right)
    if not left_surface.any() or not right_surface.any():
        return None, None
    to_right = distance_transform_edt(~right_surface, sampling=spacing)[left_surface]
    to_left = distance_transform_edt(~left_surface, sampling=spacing)[right_surface]
    distances = np.concatenate([to_right, to_left])
    return float(distances.mean()), float(np.percentile(distances, 95))


def _normal(value: str) -> str:
    return " ".join((value or "").casefold().replace("_", " ").replace("-", " ").split())


class ContourQCEngine:
    """Compare uploaded contours with an independent model contour of the same name.

    Three separate things are produced, and they are deliberately different in
    kind: a geometric comparison for organs, a description for target volumes,
    and a set difference against the site protocol for completeness. None of
    them is a clinical judgement about the plan.
    """

    def __init__(
        self,
        store: ResearchStore,
        backend: "SegmentationBackend",
        knowledge: "HybridKnowledgeBase | None" = None,
        with_surface: bool = True,
    ):
        self.store = store
        self.backend = backend
        self.knowledge = knowledge
        self.with_surface = with_surface

    def run(self, case_id: str, site_query: str | None = None) -> QCReport:
        from .imaging import load_mask, load_volume

        case = self.store.get_case(case_id)
        if not case.contours:
            return QCReport(
                case_id=case_id,
                warnings=["No uploaded contours are registered for this case."],
            )
        image_path = self.store.artifact_path(case.image)
        volume = load_volume(image_path)
        voxel_ml = float(np.prod(volume.spacing) / 1000.0)
        warnings: list[str] = []

        organs = [item for item in case.contours if not is_target(item.label)]
        targets = [item for item in case.contours if is_target(item.label)]

        loaded: dict[str, np.ndarray] = {}
        for ref in case.contours:
            try:
                loaded[ref.artifact_id] = load_mask(self.store.artifact_path(ref), volume)
            except Exception as exc:
                warnings.append(f'Could not read the contour "{ref.label}": {exc}')

        comparisons = self._compare_organs(
            image_path, volume, voxel_ml, organs, loaded, warnings
        )
        target_notes = self._describe_targets(voxel_ml, targets, organs, loaded)
        completeness = self._completeness(case, site_query)

        summary = {"ok": 0, "warn": 0, "error": 0}
        for item in comparisons:
            summary[item.status] = summary.get(item.status, 0) + 1
        return QCReport(
            case_id=case_id,
            comparisons=comparisons,
            targets=target_notes,
            completeness=completeness,
            summary=summary,
            warnings=warnings,
        )

    def _compare_organs(
        self, image_path, volume, voxel_ml, organs, loaded, warnings
    ) -> list[QCComparison]:
        usable = [ref for ref in organs if ref.artifact_id in loaded]
        if not usable:
            return []
        names = [ref.label for ref in usable]
        try:
            model_volume, model_masks = self.backend.segment(image_path, names)
        except Exception as exc:
            warnings.append(f"The comparison model could not be run: {exc}")
            return []
        if model_volume.data.shape != volume.data.shape:
            warnings.append("The comparison model returned a mismatched grid; skipping comparison.")
            return []

        results: list[QCComparison] = []
        for index, ref in enumerate(usable):
            contour = loaded[ref.artifact_id]
            model_mask = np.asarray(model_masks[index], dtype=bool)
            contour_voxels = int(np.count_nonzero(contour))
            model_voxels = int(np.count_nonzero(model_mask))
            if model_voxels == 0:
                results.append(
                    QCComparison(
                        structure=ref.label,
                        contour_volume_ml=round(contour_voxels * voxel_ml, 3),
                        model_volume_ml=0.0,
                        status="warn",
                        note=(
                            "The comparison model found nothing for this name, so the "
                            "contour could not be checked automatically."
                        ),
                    )
                )
                continue
            score = dice(contour, model_mask)
            mean_mm, hd95 = (
                surface_metrics(contour, model_mask, volume.spacing)
                if self.with_surface
                else (None, None)
            )
            contour_ml = contour_voxels * voxel_ml
            model_ml = model_voxels * voxel_ml
            difference = (
                round((contour_ml - model_ml) / model_ml * 100.0, 1) if model_ml else None
            )
            if score < DICE_ERROR:
                status, note = "error", "Large disagreement with the model contour; review this structure."
            elif score < DICE_WARN:
                status, note = "warn", "Moderate disagreement with the model contour; worth a look."
            else:
                status, note = "ok", "Close agreement with the model contour."
            results.append(
                QCComparison(
                    structure=ref.label,
                    dice=round(score, 3),
                    contour_volume_ml=round(contour_ml, 3),
                    model_volume_ml=round(model_ml, 3),
                    volume_difference_pct=difference,
                    mean_surface_mm=round(mean_mm, 2) if mean_mm is not None else None,
                    hd95_mm=round(hd95, 2) if hd95 is not None else None,
                    status=status,
                    note=note,
                )
            )
        return results

    def _describe_targets(self, voxel_ml, targets, organs, loaded) -> list[QCTargetNote]:
        """Describe each target and which contoured organs it lies near."""
        from .imaging import mask_geometry

        notes: list[QCTargetNote] = []
        centroids: dict[str, np.ndarray] = {}
        for ref in organs:
            mask = loaded.get(ref.artifact_id)
            if mask is None:
                continue
            voxels, _, centroid = mask_geometry(mask)
            if voxels:
                centroids[ref.label] = np.asarray(centroid, dtype=float)

        for ref in targets:
            mask = loaded.get(ref.artifact_id)
            if mask is None:
                continue
            voxels, _, centroid = mask_geometry(mask)
            if not voxels:
                notes.append(QCTargetNote(structure=ref.label, volume_ml=0.0))
                continue
            point = np.asarray(centroid, dtype=float)
            ranked = sorted(
                centroids.items(), key=lambda item: float(np.linalg.norm(item[1] - point))
            )
            notes.append(
                QCTargetNote(
                    structure=ref.label,
                    volume_ml=round(voxels * voxel_ml, 3),
                    centroid_voxels=[round(float(c), 1) for c in point],
                    near=[name for name, _ in ranked[:3]],
                )
            )
        return notes

    def _completeness(self, case, site_query: str | None) -> QCCompleteness | None:
        """Set difference between the site's protocol structures and what is contoured.

        This states which protocol structures have no contour in this case. It
        is not an opinion about whether the plan is adequate.
        """
        if self.knowledge is None:
            return None
        query = site_query or " ".join(item.label for item in case.contours)
        protocol = self.knowledge.lookup_protocol(query)
        if protocol is None:
            return None
        present_labels = {_normal(item.label) for item in case.contours}
        expected = list(protocol.oars)
        missing = [
            name
            for name in expected
            if not any(
                _normal(name) == label
                or _normal(name) in label
                or label in _normal(name)
                for label in present_labels
            )
        ]
        return QCCompleteness(
            site=protocol.site,
            expected=expected,
            present=sorted(item.label for item in case.contours),
            missing=missing,
            citations=list(protocol.citations),
        )
