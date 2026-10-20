"""Expert-model cross-check for contour QC.

Re-segment the present organs with an expert model and compare each user contour
to the expert's mask (Dice / overlap / surface distance). Large disagreement is
flagged — framed as "disagrees with the expert model", not "wrong", since the
expert (VoxTell) is not ground truth.

Experts live in a registry so more can be added (TotalSegmentator, SegVol, ...)
without touching the QC agent. The model is injected, so this module imports no
torch.
"""

from typing import Callable, Dict, List, Optional

import numpy as np

import qc_metrics as M

# Dice thresholds for turning agreement into a severity.
DICE_OK = 0.70
DICE_WARN = 0.50


class VoxTellExpert:
    """Wraps the loaded VoxTellPredictor; re-segments a batch in ONE pass."""

    name = "voxtell"

    def __init__(self, predictor):
        self.predictor = predictor

    def resegment(self, image: np.ndarray, props, organs: List[str]) -> Dict[str, np.ndarray]:
        if not organs:
            return {}
        seg = self.predictor.predict_single_image(image, organs)  # (N, X, Y, Z)
        return {organ: (np.asarray(seg[i]) > 0) for i, organ in enumerate(organs)}


# ---- registry (extensibility hook) ----
EXPERTS: Dict[str, Callable[[object], object]] = {
    "voxtell": VoxTellExpert,
}


def make_expert(name: str, predictor):
    factory = EXPERTS.get(name)
    return factory(predictor) if factory else None


def _finding(severity: str, message: str, metric=None) -> dict:
    return {"check": "expert", "severity": severity, "message": message, "metric": metric}


def _dice_finding(organ: str, metrics: dict, expert_empty: bool) -> Optional[dict]:
    if expert_empty:
        return _finding("ok", "expert model did not segment this organ; "
                              "cross-check unavailable")
    dice = metrics.get("dice")
    if dice is None:
        return None
    msd = metrics.get("mean_surface_distance_mm")
    vr = metrics.get("volume_ratio")
    tail = (f", surface dist {msd} mm" if msd is not None else "")
    tail += (f", vol ratio {vr}" if vr is not None else "")
    if dice >= DICE_OK:
        return _finding("ok", f"agrees with expert (Dice {dice}{tail})",
                        {"dice": dice})
    if dice >= DICE_WARN:
        return _finding("warn", f"moderate disagreement with expert (Dice {dice}"
                                f"{tail}) — review boundaries", {"dice": dice})
    return _finding("error", f"large disagreement with expert (Dice {dice}{tail}) "
                             f"— likely mis-contour or wrong structure",
                    {"dice": dice})


def cross_check(structure_set, expert, with_surface: bool = True) -> Dict[str, dict]:
    """Re-segment all present organs and compare to the user contours.

    Returns ``organ -> {"metrics", "expert_mask", "finding"}``.
    """
    organs = list(structure_set.structures.keys())
    try:
        expert_masks = expert.resegment(
            structure_set.image, structure_set.props, organs)
    except Exception as e:
        return {organ: {"metrics": {}, "expert_mask": None,
                        "finding": _finding("ok", f"expert cross-check failed: {e}")}
                for organ in organs}

    results: Dict[str, dict] = {}
    for organ in organs:
        user = structure_set.structures[organ]["mask"]
        exp = expert_masks.get(organ)
        expert_empty = exp is None or int(np.asarray(exp).sum()) == 0
        metrics = ({} if expert_empty
                   else M.compare(user, exp, structure_set.spacing, with_surface))
        results[organ] = {
            "metrics": metrics,
            "expert_mask": exp,
            "finding": _dice_finding(organ, metrics, expert_empty),
        }
    return results
