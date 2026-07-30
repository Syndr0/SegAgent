from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # imaging pulls in nibabel; keep this module importable without it
    from .expert import SegmentationBackend
    from .knowledge import HybridKnowledgeBase

from .schemas import (
    MetricEvidence,
    ProtocolApplicability,
    QCCompleteness,
    QCComparison,
    QCReport,
    QCTargetNote,
    ReportEvidenceBundle,
    ReportReadiness,
    ReportStructureEvidence,
)
from .storage import ResearchStore


# Target volumes are clinical constructs: an anatomy model cannot confirm or
# refute them, so they are described rather than compared.
# The token expression deliberately recognizes common target subclasses
# (GTVp/GTVn/CTVn/PTVp, numbered targets, and iGTV) while refusing arbitrary
# words that merely contain those letters.
TARGET_TOKEN = re.compile(
    r"^(?P<class>igtv|gtv|ctv|ptv|itv)"
    r"(?:(?:p|n|pn)(?:\d+)?)?"
    r"(?:(?:\d+(?:\.\d+)?)(?:gy)?)?$",
    re.IGNORECASE,
)
TARGET_WORDS = {"boost", "靶区", "肿瘤区", "临床靶区", "计划靶区"}
TARGET_PHRASES = {
    "gross tumor volume": "GTV",
    "gross tumour volume": "GTV",
    "clinical target volume": "CTV",
    "planning target volume": "PTV",
    "internal target volume": "ITV",
}
TARGET_NOUNS = {"tumor", "tumour", "lesion", "mass"}

# Conservative, explicit aliases only. Equality is always checked after this
# mapping; there is intentionally no substring/fuzzy fallback (so brain is not
# brainstem, colon is not sigmoid colon, and eye is not left eye).
ROI_ALIASES = {
    "brain stem": "brainstem",
    "spinalcord": "spinal cord",
    "optic chiasma": "optic chiasm",
    "eye left": "left eye",
    "eye right": "right eye",
    "kidney left": "left kidney",
    "kidney right": "right kidney",
    "lung left": "left lung",
    "lung right": "right lung",
}
SIDE_SUFFIXES = {
    "l": "left",
    "lt": "left",
    "left": "left",
    "r": "right",
    "rt": "right",
    "right": "right",
}
ROI_BASE_ALIASES = {
    "parotid": "parotid gland",
    "submandibular": "submandibular gland",
    "sm gland": "submandibular gland",
    "opticnrv": "optic nerve",
    "optic nerve": "optic nerve",
    "adrenal": "adrenal gland",
}

MALE_ONLY_ROIS = {
    "prostate",
    "seminal vesicles",
    "penile bulb",
}
FEMALE_ONLY_ROIS = {
    "uterus",
    "cervix",
    "vagina",
    "left ovary",
    "right ovary",
}

# Dice thresholds for flagging disagreement between an uploaded contour and an
# independent model contour. Disagreement means "worth a human look", never
# "the contour is wrong" — the model is a second opinion, not ground truth.
DICE_WARN = 0.80
DICE_ERROR = 0.60


def canonical_roi(label: str) -> str:
    """Return the exact canonical identifier used for ROI equality checks."""
    value = unicodedata.normalize("NFKC", label or "").casefold()
    value = re.sub(r"[\s_.:/\\-]+", " ", value)
    value = " ".join(value.split())
    value = ROI_ALIASES.get(value, value)
    tokens = value.split()
    if len(tokens) > 1 and tokens[-1] in SIDE_SUFFIXES:
        side = SIDE_SUFFIXES[tokens[-1]]
        base = " ".join(tokens[:-1])
        base = ROI_BASE_ALIASES.get(base, base)
        value = f"{side} {base}"
    return ROI_ALIASES.get(value, value)


def target_class(label: str) -> str | None:
    """Return the base target class, including subclasses such as ``GTVp``."""
    normalized = canonical_roi(label)
    for phrase, target_type in TARGET_PHRASES.items():
        if phrase in normalized:
            return target_type
    if any(word in normalized for word in TARGET_WORDS if not word.isascii()):
        return "TARGET"
    for token in normalized.split():
        match = TARGET_TOKEN.fullmatch(token)
        if match:
            base = match.group("class").upper()
            return "IGTV" if base == "IGTV" else base
        if token == "boost":
            return "BOOST"
        if token in TARGET_NOUNS:
            return "TARGET"
    return None


def is_target(label: str) -> bool:
    return target_class(label) is not None


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
    except (ImportError, AttributeError):
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


def nearest_surface_distance_mm(
    a: np.ndarray, b: np.ndarray, spacing: tuple[float, float, float]
) -> float | None:
    """Shortest distance between two masks in physical millimetres."""
    left, right = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if not left.any() or not right.any():
        return None
    if np.logical_and(left, right).any():
        return 0.0
    try:
        from scipy.ndimage import distance_transform_edt
    except (ImportError, AttributeError):
        distance_transform_edt = None
    if distance_transform_edt is not None:
        distance_to_right = distance_transform_edt(~right, sampling=spacing)
        return float(distance_to_right[left].min())

    # Minimal dependency fallback for research/test environments where scipy is
    # absent. Use exact centre-to-centre distances for modest masks; for very
    # large masks, the physical bounding-box gap is a conservative fallback.
    left_points = np.argwhere(left).astype(float) * np.asarray(spacing, dtype=float)
    right_points = np.argwhere(right).astype(float) * np.asarray(spacing, dtype=float)
    if left_points.shape[0] * right_points.shape[0] <= 2_000_000:
        best = float("inf")
        for start in range(0, left_points.shape[0], 256):
            delta = (
                left_points[start:start + 256, None, :]
                - right_points[None, :, :]
            )
            best = min(best, float(np.sqrt(np.square(delta).sum(axis=2)).min()))
        return best
    left_min, left_max = left_points.min(axis=0), left_points.max(axis=0)
    right_min, right_max = right_points.min(axis=0), right_points.max(axis=0)
    gap = np.maximum(np.maximum(left_min - right_max, right_min - left_max), 0.0)
    return float(np.linalg.norm(gap))


def centroid_mm(centroid_voxels, affine: np.ndarray) -> list[float] | None:
    """Transform a voxel centroid into the NIfTI world coordinate system."""
    if centroid_voxels is None or len(centroid_voxels) < 3:
        return None
    point = np.asarray([*centroid_voxels[:3], 1.0], dtype=float)
    world = np.asarray(affine, dtype=float) @ point
    return [round(float(value), 2) for value in world[:3]]


def render_qc_difference_overlays(
    image: np.ndarray,
    contour: np.ndarray,
    model_mask: np.ndarray,
    count: int = 3,
):
    """Render overlap (green), contour-only (red), and model-only (cyan)."""
    from PIL import Image

    from .imaging import to_uint8, window_bounds

    uploaded = np.asarray(contour, dtype=bool)
    reference = np.asarray(model_mask, dtype=bool)
    union = uploaded | reference
    areas = union.sum(axis=(0, 1))
    z_values = [int(z) for z in np.argsort(areas)[::-1] if areas[z] > 0][:count]
    z_values.sort()
    lo, hi = window_bounds(image)
    rendered = []
    for z in z_values:
        base = to_uint8(image[..., z], lo, hi).astype(np.float32)
        rgb = np.stack([base, base, base], axis=-1)
        overlap = uploaded[..., z] & reference[..., z]
        contour_only = uploaded[..., z] & ~reference[..., z]
        model_only = reference[..., z] & ~uploaded[..., z]
        for region, color in (
            (overlap, np.array([35.0, 220.0, 90.0], dtype=np.float32)),
            (contour_only, np.array([255.0, 45.0, 45.0], dtype=np.float32)),
            (model_only, np.array([30.0, 180.0, 255.0], dtype=np.float32)),
        ):
            rgb[region] = 0.35 * rgb[region] + 0.65 * color
        oriented = np.transpose(rgb.astype(np.uint8), (1, 0, 2))[::-1]
        rendered.append((z, Image.fromarray(oriented)))
    return rendered


def _evidence(
    *,
    evidence_id: str,
    structure: str | None,
    metric: str,
    value: float | int | None,
    unit: str,
    source: str,
    artifact_ids=(),
) -> MetricEvidence | None:
    if value is None:
        return None
    return MetricEvidence(
        evidence_id=evidence_id,
        structure=structure,
        canonical_roi=canonical_roi(structure or "") or None,
        metric=metric,
        value=float(value),
        unit=unit,
        source=source,
        artifact_ids=list(dict.fromkeys(str(item) for item in artifact_ids if item)),
    )


def _patient_sex(metadata: dict) -> str | None:
    value = str(
        metadata.get("patient_sex")
        or metadata.get("sex")
        or metadata.get("PatientSex")
        or ""
    ).strip().casefold()
    if value in {"m", "male", "man", "男"}:
        return "male"
    if value in {"f", "female", "woman", "女"}:
        return "female"
    return None


def _protocol_source_status(source: str) -> str:
    lowered = (source or "").casefold()
    if "research" in lowered or "seed" in lowered:
        return "research_seed"
    if "approved" in lowered:
        return "institution_approved"
    return "unknown"


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

    def _persist_comparison_artifacts(
        self, case_id, volume, uploaded_ref, uploaded_mask, model_mask, warnings
    ):
        """Persist the model reference and visual diff used by QC metrics."""
        from .imaging import save_mask

        model_ref = None
        overlays = []
        try:
            artifact_id, mask_path = self.store.allocate_artifact_path(
                case_id, ".nii.gz"
            )
            save_mask(model_mask, volume, mask_path)
            model_ref = self.store.register_artifact(
                case_id,
                artifact_id,
                mask_path,
                "mask",
                f"{uploaded_ref.label} (QC model reference)",
                "application/gzip",
                {
                    "artifact_role": "qc_reference",
                    "canonical_roi": canonical_roi(uploaded_ref.label),
                    "compared_to": uploaded_ref.artifact_id,
                    "source_model": getattr(self.backend, "name", "comparison model"),
                    "source_version": getattr(self.backend, "version", "unknown"),
                    "target_type": "organ",
                },
            )
        except Exception as exc:
            warnings.append(
                f'Could not persist the comparison-model mask for "{uploaded_ref.label}": {exc}'
            )
            return None, []

        try:
            for z, image in render_qc_difference_overlays(
                volume.data, uploaded_mask, model_mask
            ):
                overlay_id, overlay_path = self.store.allocate_artifact_path(
                    case_id, ".png"
                )
                image.save(overlay_path, format="PNG")
                overlays.append(
                    self.store.register_artifact(
                        case_id,
                        overlay_id,
                        overlay_path,
                        "overlay",
                        f"{uploaded_ref.label} QC comparison, axial z={z}",
                        "image/png",
                        {
                            "artifact_role": "qc_difference_overlay",
                            "structure": uploaded_ref.label,
                            "canonical_roi": canonical_roi(uploaded_ref.label),
                            "slice_index": z,
                            "uploaded_contour_id": uploaded_ref.artifact_id,
                            "model_mask_id": model_ref.artifact_id,
                            "legend": {
                                "overlap": "green",
                                "uploaded_only": "red",
                                "model_only": "cyan",
                            },
                        },
                    )
                )
        except Exception as exc:
            warnings.append(
                f'Could not render QC difference overlays for "{uploaded_ref.label}": {exc}'
            )
        return model_ref, overlays

    def run(self, case_id: str, site_query: str | None = None) -> QCReport:
        from .imaging import load_mask, load_volume

        case = self.store.get_case(case_id)
        if not case.contours:
            completeness = self._completeness(case, site_query)
            metric_evidence: list[MetricEvidence] = []
            if (
                completeness
                and completeness.applicability
                and completeness.applicability.status != "not_assessed"
            ):
                for metric, value in (
                    ("expected_structure_count", len(completeness.expected)),
                    ("present_structure_count", 0),
                    ("missing_structure_count", len(completeness.missing)),
                ):
                    item = _evidence(
                        evidence_id=f"qc:{case_id}:protocol:{metric}",
                        structure=None,
                        metric=metric,
                        value=value,
                        unit="count",
                        source=completeness.source or "protocol registry",
                    )
                    if item:
                        metric_evidence.append(item)
            return QCReport(
                case_id=case_id,
                completeness=completeness,
                metric_evidence=metric_evidence,
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

        comparisons, comparison_evidence = self._compare_organs(
            image_path,
            volume,
            voxel_ml,
            organs,
            loaded,
            warnings,
            case.image.artifact_id,
        )
        target_notes, target_evidence = self._describe_targets(
            volume, voxel_ml, targets, organs, loaded, case.image.artifact_id
        )
        completeness = self._completeness(case, site_query)

        summary = {"ok": 0, "warn": 0, "error": 0}
        for item in comparisons:
            summary[item.status] = summary.get(item.status, 0) + 1
        metric_evidence = [*comparison_evidence, *target_evidence]
        if (
            completeness
            and completeness.applicability
            and completeness.applicability.status != "not_assessed"
        ):
            for metric, value in (
                ("expected_structure_count", len(completeness.expected)),
                ("present_structure_count", len(completeness.present)),
                ("missing_structure_count", len(completeness.missing)),
            ):
                item = _evidence(
                    evidence_id=f"qc:{case_id}:protocol:{metric}",
                    structure=None,
                    metric=metric,
                    value=value,
                    unit="count",
                    source=completeness.source or "protocol registry",
                    artifact_ids=[ref.artifact_id for ref in case.contours],
                )
                if item:
                    metric_evidence.append(item)
        return QCReport(
            case_id=case_id,
            comparisons=comparisons,
            targets=target_notes,
            completeness=completeness,
            summary=summary,
            metric_evidence=metric_evidence,
            warnings=warnings,
        )

    def _compare_organs(
        self,
        image_path,
        volume,
        voxel_ml,
        organs,
        loaded,
        warnings,
        image_artifact_id,
    ) -> tuple[list[QCComparison], list[MetricEvidence]]:
        usable = [ref for ref in organs if ref.artifact_id in loaded]
        if not usable:
            return [], []
        names = [ref.label for ref in usable]
        try:
            model_volume, model_masks = self.backend.segment(image_path, names)
        except Exception as exc:
            warnings.append(f"The comparison model could not be run: {exc}")
            return [], []
        if (
            model_volume.data.shape != volume.data.shape
            or not np.allclose(model_volume.affine, volume.affine, atol=1e-3, rtol=1e-4)
        ):
            warnings.append("The comparison model returned a mismatched grid; skipping comparison.")
            return [], []

        results: list[QCComparison] = []
        evidence: list[MetricEvidence] = []
        for index, ref in enumerate(usable):
            contour = loaded[ref.artifact_id]
            model_mask = np.asarray(model_masks[index], dtype=bool)
            contour_voxels = int(np.count_nonzero(contour))
            model_voxels = int(np.count_nonzero(model_mask))
            if model_voxels == 0:
                results.append(
                    QCComparison(
                        structure=ref.label,
                        canonical_roi=canonical_roi(ref.label),
                        contour_artifact_id=ref.artifact_id,
                        model_name=getattr(self.backend, "name", None),
                        model_version=getattr(self.backend, "version", None),
                        contour_volume_ml=round(contour_voxels * voxel_ml, 3),
                        model_volume_ml=0.0,
                        status="warn",
                        note=(
                            "The comparison model found nothing for this name, so the "
                            "contour could not be checked automatically."
                        ),
                    )
                )
                for metric, value, unit, source, artifact_ids in (
                    (
                        "contour_volume",
                        round(contour_voxels * voxel_ml, 3),
                        "mL",
                        "uploaded contour",
                        [ref.artifact_id, image_artifact_id],
                    ),
                    (
                        "model_volume",
                        0.0,
                        "mL",
                        f"{getattr(self.backend, 'name', 'comparison model')} output",
                        [image_artifact_id],
                    ),
                ):
                    item = _evidence(
                        evidence_id=f"qc:{ref.artifact_id}:{metric}",
                        structure=ref.label,
                        metric=metric,
                        value=value,
                        unit=unit,
                        source=source,
                        artifact_ids=artifact_ids,
                    )
                    if item:
                        evidence.append(item)
                continue
            model_ref, difference_overlays = self._persist_comparison_artifacts(
                uploaded_ref=ref,
                uploaded_mask=contour,
                model_mask=model_mask,
                case_id=ref.case_id,
                volume=volume,
                warnings=warnings,
            )
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
            comparison = QCComparison(
                structure=ref.label,
                canonical_roi=canonical_roi(ref.label),
                contour_artifact_id=ref.artifact_id,
                model_mask=model_ref,
                difference_overlays=difference_overlays,
                model_name=getattr(self.backend, "name", None),
                model_version=getattr(self.backend, "version", None),
                dice=round(score, 3),
                contour_volume_ml=round(contour_ml, 3),
                model_volume_ml=round(model_ml, 3),
                volume_difference_pct=difference,
                mean_surface_mm=round(mean_mm, 2) if mean_mm is not None else None,
                hd95_mm=round(hd95, 2) if hd95 is not None else None,
                status=status,
                note=note,
            )
            results.append(comparison)
            model_artifact_id = model_ref.artifact_id if model_ref else None
            overlay_ids = [item.artifact_id for item in difference_overlays]
            comparison_artifacts = [
                ref.artifact_id,
                model_artifact_id,
                image_artifact_id,
                *overlay_ids,
            ]
            values = (
                (
                    "dice",
                    comparison.dice,
                    "ratio",
                    "uploaded contour vs comparison model",
                    comparison_artifacts,
                ),
                (
                    "contour_volume",
                    comparison.contour_volume_ml,
                    "mL",
                    "uploaded contour",
                    [ref.artifact_id, image_artifact_id],
                ),
                (
                    "model_volume",
                    comparison.model_volume_ml,
                    "mL",
                    f"{getattr(self.backend, 'name', 'comparison model')} output",
                    [model_artifact_id, image_artifact_id],
                ),
                (
                    "volume_difference",
                    comparison.volume_difference_pct,
                    "%",
                    "uploaded contour vs comparison model",
                    comparison_artifacts,
                ),
                (
                    "mean_surface_distance",
                    comparison.mean_surface_mm,
                    "mm",
                    "uploaded contour vs comparison model",
                    comparison_artifacts,
                ),
                (
                    "hd95",
                    comparison.hd95_mm,
                    "mm",
                    "uploaded contour vs comparison model",
                    comparison_artifacts,
                ),
            )
            for metric, value, unit, source, artifact_ids in values:
                item = _evidence(
                    evidence_id=f"qc:{ref.artifact_id}:{metric}",
                    structure=ref.label,
                    metric=metric,
                    value=value,
                    unit=unit,
                    source=source,
                    artifact_ids=artifact_ids,
                )
                if item:
                    evidence.append(item)
        return results, evidence

    def _describe_targets(
        self, volume, voxel_ml, targets, organs, loaded, image_artifact_id
    ) -> tuple[list[QCTargetNote], list[MetricEvidence]]:
        """Describe each target and which contoured organs it lies near."""
        from .imaging import mask_geometry

        notes: list[QCTargetNote] = []
        evidence: list[MetricEvidence] = []
        organ_masks: list[tuple[str, str, np.ndarray]] = []
        for ref in organs:
            mask = loaded.get(ref.artifact_id)
            if mask is None:
                continue
            voxels, _, centroid = mask_geometry(mask)
            if voxels:
                organ_masks.append((ref.label, ref.artifact_id, mask))

        for ref in targets:
            mask = loaded.get(ref.artifact_id)
            if mask is None:
                continue
            voxels, _, centroid = mask_geometry(mask)
            if not voxels:
                note = QCTargetNote(
                    structure=ref.label,
                    canonical_roi=canonical_roi(ref.label),
                    target_class=target_class(ref.label),
                    artifact_id=ref.artifact_id,
                    volume_ml=0.0,
                )
                notes.append(note)
                item = _evidence(
                    evidence_id=f"qc:{ref.artifact_id}:volume",
                    structure=ref.label,
                    metric="volume",
                    value=0.0,
                    unit="mL",
                    source="uploaded target contour",
                    artifact_ids=[ref.artifact_id, image_artifact_id],
                )
                if item:
                    evidence.append(item)
                continue
            point = np.asarray(centroid, dtype=float)
            ranked: list[tuple[float, str, str]] = []
            for organ_name, organ_artifact_id, organ_mask in organ_masks:
                distance = nearest_surface_distance_mm(mask, organ_mask, volume.spacing)
                if distance is not None:
                    ranked.append((distance, organ_name, organ_artifact_id))
            ranked.sort(key=lambda item: (item[0], canonical_roi(item[1])))
            nearby = {
                name: round(float(distance), 2) for distance, name, _ in ranked[:3]
            }
            note = QCTargetNote(
                structure=ref.label,
                canonical_roi=canonical_roi(ref.label),
                target_class=target_class(ref.label),
                artifact_id=ref.artifact_id,
                volume_ml=round(voxels * voxel_ml, 3),
                centroid_voxels=[round(float(c), 2) for c in point],
                centroid_mm=centroid_mm(point, volume.affine),
                near=list(nearby),
                near_surface_mm=nearby,
            )
            notes.append(note)
            artifacts = [ref.artifact_id, image_artifact_id]
            item = _evidence(
                evidence_id=f"qc:{ref.artifact_id}:volume",
                structure=ref.label,
                metric="volume",
                value=note.volume_ml,
                unit="mL",
                source="uploaded target contour",
                artifact_ids=artifacts,
            )
            if item:
                evidence.append(item)
            for axis, value in zip(("x", "y", "z"), note.centroid_mm or []):
                item = _evidence(
                    evidence_id=f"qc:{ref.artifact_id}:centroid_{axis}",
                    structure=ref.label,
                    metric=f"centroid_{axis}",
                    value=value,
                    unit="mm",
                    source="NIfTI world geometry",
                    artifact_ids=artifacts,
                )
                if item:
                    evidence.append(item)
            near_artifacts = {
                name: organ_artifact_id
                for _, name, organ_artifact_id in ranked[:3]
            }
            for organ_name, distance in nearby.items():
                item = _evidence(
                    evidence_id=(
                        f"qc:{ref.artifact_id}:near_surface:"
                        f"{canonical_roi(organ_name).replace(' ', '_')}"
                    ),
                    structure=ref.label,
                    metric=f"surface_distance_to:{canonical_roi(organ_name)}",
                    value=distance,
                    unit="mm",
                    source="physical-space mask geometry",
                    artifact_ids=[*artifacts, near_artifacts.get(organ_name)],
                )
                if item:
                    evidence.append(item)
        return notes, evidence

    def _completeness(self, case, site_query: str | None) -> QCCompleteness | None:
        """Set difference between the site's protocol structures and what is contoured.

        This states which protocol structures have no contour in this case. It
        is not an opinion about whether the plan is adequate.
        """
        if self.knowledge is None:
            return None
        query = (site_query or "").strip()
        if not query:
            return QCCompleteness(
                applicability=ProtocolApplicability(
                    status="not_assessed",
                    reason=(
                        "Protocol completeness requires a confirmed treatment site; "
                        "ROI names alone are not used to infer one."
                    ),
                )
            )
        protocol = self.knowledge.lookup_protocol(query)
        if protocol is None:
            return QCCompleteness(
                applicability=ProtocolApplicability(
                    status="not_assessed",
                    reason=f'No protocol matched the confirmed site query "{query}".',
                )
            )

        expected = list(protocol.oars)
        patient_sex = _patient_sex(case.metadata or {})
        excluded: list[str] = []
        unresolved: list[str] = []
        if patient_sex == "male":
            excluded = [name for name in expected if canonical_roi(name) in FEMALE_ONLY_ROIS]
        elif patient_sex == "female":
            excluded = [name for name in expected if canonical_roi(name) in MALE_ONLY_ROIS]
        else:
            unresolved = [
                name
                for name in expected
                if canonical_roi(name) in MALE_ONLY_ROIS | FEMALE_ONLY_ROIS
            ]
        excluded_ids = {canonical_roi(name) for name in [*excluded, *unresolved]}
        applicable_expected = [
            name for name in expected if canonical_roi(name) not in excluded_ids
        ]

        # Exact equality after conservative canonicalization. Never accept
        # "brainstem" for "brain", "sigmoid colon" for "colon", or "eye" for
        # "left eye".
        organ_contours = [item for item in case.contours if not is_target(item.label)]
        present_by_id: dict[str, str] = {}
        for item in organ_contours:
            present_by_id.setdefault(canonical_roi(item.label), item.label)
        expected_by_id = {canonical_roi(name): name for name in applicable_expected}
        present = [
            name for canonical, name in expected_by_id.items() if canonical in present_by_id
        ]
        missing = [
            name for canonical, name in expected_by_id.items() if canonical not in present_by_id
        ]
        unmatched = sorted(
            label
            for canonical, label in present_by_id.items()
            if canonical not in expected_by_id
        )
        source_status = _protocol_source_status(protocol.source)
        if unresolved:
            status = "partially_applicable"
            reason = (
                "The treatment site matched, but sex-specific protocol structures "
                "were excluded from completeness until patient applicability is known."
            )
        else:
            status = "applicable"
            reason = "The protocol matches the supplied treatment-site query."
        if source_status == "research_seed":
            reason += (
                " This registry entry is research seed data and requires "
                "institutional approval before clinical use."
            )
        return QCCompleteness(
            site=protocol.site,
            expected=applicable_expected,
            present=present,
            missing=missing,
            unmatched_contours=unmatched,
            citations=list(protocol.citations),
            source=protocol.source,
            source_version=protocol.source_version,
            applicability=ProtocolApplicability(
                status=status,
                reason=reason,
                source_status=source_status,
                excluded=excluded,
                unresolved=unresolved,
            ),
        )


def _report_candidates(case) -> list:
    """Choose one active artifact per canonical ROI.

    Explicit human edits and ``derived_from`` descendants win over generated
    predecessors; otherwise the newest artifact wins. A derivation chain also
    suppresses its parent if the editor renamed the ROI. Distinct canonical
    target subclasses (for example GTVp and GTVn) remain separate.
    """
    seen: set[str] = set()
    candidates = []
    for item in case.artifacts:
        if item.artifact_id in seen or item.kind not in {"mask", "contour"}:
            continue
        if item.metadata.get("artifact_role") == "qc_reference":
            continue
        seen.add(item.artifact_id)
        candidates.append(item)

    by_id = {item.artifact_id: item for item in candidates}
    superseded: set[str] = set()
    for item in candidates:
        parent_id = item.metadata.get("derived_from")
        visited: set[str] = set()
        while parent_id in by_id and parent_id not in visited:
            visited.add(parent_id)
            superseded.add(parent_id)
            parent_id = by_id[parent_id].metadata.get("derived_from")

    grouped: dict[str, list[tuple[int, object]]] = {}
    for index, item in enumerate(candidates):
        if item.artifact_id in superseded:
            continue
        grouped.setdefault(canonical_roi(item.label), []).append((index, item))

    def rank(index_and_item) -> tuple[int, int, object, int]:
        index, item = index_and_item
        metadata = item.metadata or {}
        source = str(metadata.get("source_model") or "").casefold()
        human = int(source == "human_edit")
        derived = int(bool(metadata.get("derived_from")))
        created = getattr(item, "created_at", None)
        # ISO timestamps and datetime objects are both consistently orderable
        # within a stored case; the append index is the deterministic fallback.
        created_key = created.isoformat() if hasattr(created, "isoformat") else str(created or "")
        return human, derived, created_key, index

    selected = [max(items, key=rank) for items in grouped.values()]
    selected.sort(key=lambda index_and_item: index_and_item[0])
    return [item for _, item in selected]


def _artifact_is_target(ref) -> bool:
    role = str(ref.metadata.get("target_type") or "").casefold()
    return role in {"gtv", "ctv", "ptv", "itv", "target"} or is_target(ref.label)


def assess_report_readiness(case, scope: str = "comprehensive") -> ReportReadiness:
    """Check report prerequisites without asking a model to infer readiness.

    ``comprehensive`` requires both organ/OAR and target evidence.  Callers may
    explicitly request an available-structures, OAR-only, or target-only report;
    an absent target is never silently generated from an unstated diagnosis.
    """
    if scope not in {"comprehensive", "available", "oar", "target"}:
        raise ValueError("report scope must be comprehensive, available, oar, or target")
    candidates = _report_candidates(case)
    organ_ids = {
        canonical_roi(item.label) for item in candidates if not _artifact_is_target(item)
    }
    target_ids = {
        canonical_roi(item.label) for item in candidates if _artifact_is_target(item)
    }
    missing: list[str] = []
    if scope in {"comprehensive", "oar"} and not organ_ids:
        missing.append("organ_or_oar_contours")
    if scope in {"comprehensive", "target"} and not target_ids:
        missing.append("target_contours")
    if scope == "available" and not candidates:
        missing.append("contours")
    ready = not missing
    if ready:
        reason = (
            f"{len(organ_ids)} organ/OAR and {len(target_ids)} target canonical "
            "structures are available for the requested report scope."
        )
    else:
        reason = "Missing deterministic report inputs: " + ", ".join(missing) + "."
    return ReportReadiness(
        scope=scope,
        ready=ready,
        organ_count=len(organ_ids),
        target_count=len(target_ids),
        missing_inputs=missing,
        reason=reason,
    )


class ReportEvidenceBuilder:
    """Measure the exact artifacts that may be used to generate report prose."""

    def __init__(self, store: ResearchStore):
        self.store = store

    def build(
        self, case_id: str, scope: str = "comprehensive"
    ) -> ReportEvidenceBundle:
        from .imaging import load_mask, load_volume, mask_geometry

        case = self.store.get_case(case_id)
        readiness = assess_report_readiness(case, scope)
        volume = load_volume(self.store.artifact_path(case.image))
        voxel_ml = float(np.prod(volume.spacing) / 1000.0)
        structures: list[ReportStructureEvidence] = []
        metrics: list[MetricEvidence] = []
        warnings: list[str] = []
        loaded: dict[str, np.ndarray] = {}
        candidates = _report_candidates(case)

        for ref in candidates:
            try:
                mask = load_mask(self.store.artifact_path(ref), volume)
            except Exception as exc:
                warnings.append(f'Could not read the report contour "{ref.label}": {exc}')
                continue
            loaded[ref.artifact_id] = mask
            voxels, _, centroid = mask_geometry(mask)
            role = "target" if _artifact_is_target(ref) else "organ"
            source = str(
                ref.metadata.get("source_model")
                or ("uploaded contour" if ref.kind == "contour" else "generated mask")
            )
            volume_ml = round(voxels * voxel_ml, 3)
            world = centroid_mm(centroid, volume.affine)
            structures.append(
                ReportStructureEvidence(
                    structure=ref.label,
                    canonical_roi=canonical_roi(ref.label),
                    role=role,
                    assessment_status=(
                        "not_automatically_assessed" if role == "target" else "measured"
                    ),
                    artifact_id=ref.artifact_id,
                    source=source,
                    volume_ml=volume_ml,
                    centroid_mm=world,
                )
            )
            artifacts = [ref.artifact_id, case.image.artifact_id]
            item = _evidence(
                evidence_id=f"report:{ref.artifact_id}:volume",
                structure=ref.label,
                metric="volume",
                value=volume_ml,
                unit="mL",
                source=source,
                artifact_ids=artifacts,
            )
            if item:
                metrics.append(item)
            for axis, value in zip(("x", "y", "z"), world or []):
                item = _evidence(
                    evidence_id=f"report:{ref.artifact_id}:centroid_{axis}",
                    structure=ref.label,
                    metric=f"centroid_{axis}",
                    value=value,
                    unit="mm",
                    source="NIfTI world geometry",
                    artifact_ids=artifacts,
                )
                if item:
                    metrics.append(item)

        organs = [item for item in structures if item.role == "organ"]
        targets = [item for item in structures if item.role == "target"]
        by_artifact = {item.artifact_id: item for item in structures}
        for target in targets:
            target_mask = loaded.get(target.artifact_id)
            if target_mask is None:
                continue
            ranked: list[tuple[float, ReportStructureEvidence]] = []
            for organ in organs:
                organ_mask = loaded.get(organ.artifact_id)
                if organ_mask is None:
                    continue
                distance = nearest_surface_distance_mm(
                    target_mask, organ_mask, volume.spacing
                )
                if distance is not None:
                    ranked.append((distance, organ))
            ranked.sort(key=lambda item: (item[0], item[1].canonical_roi))
            nearby = {
                organ.structure: round(float(distance), 2)
                for distance, organ in ranked[:3]
            }
            by_artifact[target.artifact_id].near_surface_mm = nearby
            for organ_name, distance in nearby.items():
                item = _evidence(
                    evidence_id=(
                        f"report:{target.artifact_id}:surface_distance:"
                        f"{canonical_roi(organ_name).replace(' ', '_')}"
                    ),
                    structure=target.structure,
                    metric=f"surface_distance_to:{canonical_roi(organ_name)}",
                    value=distance,
                    unit="mm",
                    source="physical-space mask geometry",
                    artifact_ids=[
                        target.artifact_id,
                        next(
                            organ.artifact_id
                            for organ in organs
                            if organ.structure == organ_name
                        ),
                        case.image.artifact_id,
                    ],
                )
                if item:
                    metrics.append(item)

        return ReportEvidenceBundle(
            case_id=case_id,
            readiness=readiness,
            structures=structures,
            metric_evidence=metrics,
            warnings=warnings,
        )
