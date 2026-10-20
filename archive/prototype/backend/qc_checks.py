"""Deterministic contour-QC checks, as an extensible registry.

Add a check = write a function taking a CheckContext and returning a Finding
(or None if it passes / is not applicable), then decorate it with @register.
scipy is imported lazily inside the checks that need it so the module and the
pure-numpy checks stay usable without scipy.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

SEVERITY_ORDER = {"ok": 0, "warn": 1, "error": 2}


@dataclass
class CheckContext:
    organ: str                       # canonical organ name
    mask: np.ndarray                 # 3D binary mask (user contour)
    spacing: Optional[list]          # voxel spacing (x, y, z) in mm, or None
    lr_axis: int                     # spatial axis that runs left<->right
    left_is_high_index: bool         # is a HIGHER index along lr_axis patient-LEFT?
    reference: Optional[dict] = None  # organ_reference entry for this organ
    image: Optional[np.ndarray] = None


def _finding(check: str, severity: str, message: str, metric=None) -> dict:
    return {"check": check, "severity": severity, "message": message, "metric": metric}


CHECKS: List[Callable[[CheckContext], Optional[dict]]] = []


def register(fn: Callable[[CheckContext], Optional[dict]]):
    """Register a QC check. Extensibility hook: decorate a new function."""
    CHECKS.append(fn)
    return fn


# ------------------------------------------------------------------ checks

@register
def check_empty(ctx: CheckContext):
    n = int((ctx.mask > 0).sum())
    if n == 0:
        return _finding("empty", "error", "mask is empty (0 voxels)", {"voxels": 0})
    if n < 10:
        return _finding("empty", "warn", f"near-empty mask ({n} voxels)", {"voxels": n})
    return None


@register
def check_volume(ctx: CheckContext):
    ref = ctx.reference or {}
    rng = ref.get("volume_ml")
    if not rng or ctx.spacing is None or len(ctx.spacing) < 3:
        return None
    n = int((ctx.mask > 0).sum())
    if n == 0:
        return None  # handled by check_empty
    vox_mm3 = abs(float(np.prod(list(ctx.spacing)[:3])))
    ml = n * vox_mm3 / 1000.0
    lo, hi = rng
    if ml < lo:
        sev = "error" if ml < lo * 0.5 else "warn"
        return _finding("volume", sev,
                        f"volume {ml:.1f} mL is below the normal range "
                        f"[{lo}, {hi}] mL — possible under-contour",
                        {"volume_ml": round(ml, 1), "range": rng})
    if ml > hi:
        sev = "error" if ml > hi * 2 else "warn"
        return _finding("volume", sev,
                        f"volume {ml:.1f} mL is above the normal range "
                        f"[{lo}, {hi}] mL — possible over-contour / leakage",
                        {"volume_ml": round(ml, 1), "range": rng})
    return None


@register
def check_components(ctx: CheckContext):
    ref = ctx.reference or {}
    expected = ref.get("expected_components")
    if expected is None:
        return None
    m = ctx.mask > 0
    if m.sum() == 0:
        return None
    try:
        from scipy.ndimage import label
    except Exception:
        return None
    lab, n = label(m)
    if n <= 1:
        return None
    sizes = np.bincount(lab.ravel())[1:]
    largest = int(sizes.max())
    signif = int((sizes >= max(largest * 0.05, 10)).sum())  # ignore tiny specks
    specks = int(n - signif)
    if signif > expected:
        return _finding("components", "warn",
                        f"{signif} separate components (expected {expected}) — "
                        f"possible stray region or leakage",
                        {"components": int(n), "significant": signif})
    if specks > 0:
        return _finding("components", "warn",
                        f"{specks} tiny disconnected speck(s) outside the main region",
                        {"components": int(n), "specks": specks})
    return None


@register
def check_laterality(ctx: CheckContext):
    ref = ctx.reference or {}
    side = ref.get("laterality")
    if side not in ("left", "right"):
        return None
    coords = np.argwhere(ctx.mask > 0)
    if coords.size == 0:
        return None
    c = float(coords[:, ctx.lr_axis].mean())
    dim = ctx.mask.shape[ctx.lr_axis]
    mid = dim / 2.0
    frac = (c - mid) / dim
    if abs(frac) < 0.02:          # too near the midline to judge reliably
        return None
    centroid_is_left = (c > mid) == ctx.left_is_high_index
    if centroid_is_left != (side == "left"):
        actual = "left" if centroid_is_left else "right"
        return _finding("laterality", "error",
                        f"centroid is on the patient-{actual} side but this "
                        f"structure should be {side} — possible left/right swap",
                        {"lr_axis": ctx.lr_axis, "offset_frac": round(frac, 3)})
    return None


@register
def check_holes(ctx: CheckContext):
    m = ctx.mask > 0
    total = int(m.sum())
    if total == 0:
        return None
    try:
        from scipy.ndimage import binary_fill_holes
    except Exception:
        return None
    hole = int(binary_fill_holes(m).sum() - total)
    if hole > 0 and hole > 0.03 * total:
        return _finding("holes", "warn",
                        f"interior holes ({hole} voxels, {100 * hole / total:.0f}% of "
                        f"volume) — possible under-fill",
                        {"hole_voxels": hole})
    return None


# --------------------------------------------------------------- aggregate

def run_checks(ctx: CheckContext) -> List[dict]:
    findings = []
    for fn in CHECKS:
        try:
            f = fn(ctx)
        except Exception as e:  # a broken check must not abort QC
            f = _finding(getattr(fn, "__name__", "check"), "warn",
                         f"check error: {e}")
        if f:
            findings.append(f)
    return findings


def overall_status(findings: List[dict]) -> str:
    s = "ok"
    for f in findings:
        if SEVERITY_ORDER.get(f["severity"], 0) > SEVERITY_ORDER[s]:
            s = f["severity"]
    return s
