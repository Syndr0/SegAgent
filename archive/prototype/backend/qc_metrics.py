"""Geometric comparison metrics for contour QC.

Pure-numpy where possible; the surface-distance metric imports scipy lazily so
this module stays importable (and the overlap metrics stay unit-testable)
without scipy present.
"""

from typing import Optional, Sequence, Tuple

import numpy as np


def _b(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask) > 0


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice similarity coefficient of two binary masks (1.0 == identical)."""
    a, b = _b(a), _b(b)
    sa, sb = int(a.sum()), int(b.sum())
    if sa == 0 and sb == 0:
        return 1.0
    if sa == 0 or sb == 0:
        return 0.0
    inter = int(np.logical_and(a, b).sum())
    return 2.0 * inter / (sa + sb)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union (Jaccard) of two binary masks."""
    a, b = _b(a), _b(b)
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return int(np.logical_and(a, b).sum()) / union


def volume_ml(mask: np.ndarray, spacing_mm: Optional[Sequence[float]]) -> Optional[float]:
    """Physical volume in mL, or None if spacing is unknown."""
    n = int(_b(mask).sum())
    if spacing_mm is None or len(spacing_mm) < 3:
        return None
    vox_mm3 = float(abs(np.prod(list(spacing_mm)[:3])))
    return n * vox_mm3 / 1000.0


def volume_ratio(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Voxel-count ratio |a| / |b| (e.g. user vs expert). None if b empty."""
    na, nb = int(_b(a).sum()), int(_b(b).sum())
    if nb == 0:
        return None
    return na / nb


def centroid(mask: np.ndarray) -> Optional[np.ndarray]:
    """Voxel centroid (x, y, z) of a binary mask, or None if empty."""
    coords = np.argwhere(_b(mask))
    if coords.size == 0:
        return None
    return coords.mean(axis=0)


def centroid_distance_mm(a: np.ndarray, b: np.ndarray,
                         spacing_mm: Optional[Sequence[float]]) -> Optional[float]:
    """Distance between the two masks' centroids in mm (or voxels if no spacing)."""
    ca, cb = centroid(a), centroid(b)
    if ca is None or cb is None:
        return None
    d = ca - cb
    if spacing_mm is not None and len(spacing_mm) >= 3:
        d = d * np.asarray(spacing_mm[:3], dtype=float)
    return float(np.linalg.norm(d))


def _surface_distances(a: np.ndarray, b: np.ndarray,
                       spacing_mm: Optional[Sequence[float]]):
    """Symmetric surface distances (a->b and b->a) in mm. Requires scipy."""
    from scipy.ndimage import binary_erosion, distance_transform_edt

    a, b = _b(a), _b(b)
    if a.sum() == 0 or b.sum() == 0:
        return None
    sp = (tuple(float(s) for s in spacing_mm[:3])
          if spacing_mm is not None and len(spacing_mm) >= 3 else None)

    def border(m):
        return np.logical_and(m, np.logical_not(binary_erosion(m)))

    ba, bb = border(a), border(b)
    # distance from every voxel to the nearest surface voxel of the other mask
    dt_b = distance_transform_edt(np.logical_not(bb), sampling=sp)
    dt_a = distance_transform_edt(np.logical_not(ba), sampling=sp)
    return dt_b[ba], dt_a[bb]


def mean_surface_distance_mm(a: np.ndarray, b: np.ndarray,
                             spacing_mm: Optional[Sequence[float]]) -> Optional[float]:
    """Average symmetric surface distance (ASSD) in mm. None if either empty."""
    res = _surface_distances(a, b, spacing_mm)
    if res is None:
        return None
    d_ab, d_ba = res
    both = np.concatenate([d_ab, d_ba])
    return float(both.mean()) if both.size else None


def hausdorff95_mm(a: np.ndarray, b: np.ndarray,
                   spacing_mm: Optional[Sequence[float]]) -> Optional[float]:
    """95th-percentile symmetric Hausdorff distance in mm. None if either empty."""
    res = _surface_distances(a, b, spacing_mm)
    if res is None:
        return None
    d_ab, d_ba = res
    both = np.concatenate([d_ab, d_ba])
    return float(np.percentile(both, 95)) if both.size else None


def compare(user: np.ndarray, expert: np.ndarray,
            spacing_mm: Optional[Sequence[float]],
            with_surface: bool = True) -> dict:
    """Bundle the comparison metrics between a user contour and an expert mask."""
    out = {
        "dice": round(dice(user, expert), 3),
        "iou": round(iou(user, expert), 3),
        "volume_ratio": (round(volume_ratio(user, expert), 2)
                         if volume_ratio(user, expert) is not None else None),
        "centroid_distance_mm": (round(centroid_distance_mm(user, expert, spacing_mm), 1)
                                 if centroid_distance_mm(user, expert, spacing_mm) is not None
                                 else None),
    }
    if with_surface:
        try:
            msd = mean_surface_distance_mm(user, expert, spacing_mm)
            h95 = hausdorff95_mm(user, expert, spacing_mm)
            out["mean_surface_distance_mm"] = round(msd, 1) if msd is not None else None
            out["hausdorff95_mm"] = round(h95, 1) if h95 is not None else None
        except Exception:
            # scipy missing or degenerate geometry — overlap metrics still returned.
            pass
    return out
