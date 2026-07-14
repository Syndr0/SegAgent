"""Ingest a contour structure-set folder for QC.

A folder holds one grayscale scan (image.nii[.gz]) plus one binary mask per
structure, the filename encoding the organ (left_eye.nii -> "left eye").

The heavy reader (nnU-Net's ``NibabelIOWithReorient``) is *injected* so this
module has no hard nibabel/nnunet dependency and stays importable/testable
without them. Using the same reader VoxTell uses guarantees the user masks land
on the same reoriented (RAS) grid as the expert cross-check.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from knowledge_base import _norm

NII_EXT = (".nii", ".nii.gz")
# stems that identify the grayscale scan rather than a structure mask
IMAGE_NAME_HINTS = ("image", "img", "ct", "scan", "volume", "vol", "mr", "mri", "pet")


@dataclass
class StructureSet:
    image: np.ndarray                 # 3D grayscale (RAS)
    spacing: Optional[list]           # (x, y, z) mm
    props: dict                       # reader props (for VoxTell + mask writing)
    lr_axis: int                      # array axis running left<->right
    left_is_high_index: bool          # higher index along lr_axis == patient-left?
    structures: Dict[str, dict]       # organ name -> {"mask", "path", "stem"}
    warnings: List[str] = field(default_factory=list)


def _stem(fname: str) -> str:
    b = os.path.basename(fname)
    for e in (".nii.gz", ".nii"):
        if b.lower().endswith(e):
            return b[: -len(e)]
    return os.path.splitext(b)[0]


def _display_name(stem: str) -> str:
    return stem.replace("_", " ").replace("-", " ").strip()


def _lr_axis_and_dir(props: dict):
    """Derive the left-right array axis and its direction from the affine.

    RAS world-x points to the patient's RIGHT, so a structure on the patient's
    LEFT sits at LOW index when the affine's x-component along that axis is
    positive. Falls back to the RAS default (axis 0, left = low index).
    """
    aff = None
    ns = props.get("nibabel_stuff") if isinstance(props, dict) else None
    if isinstance(ns, dict):
        aff = ns.get("reoriented_affine")
        if aff is None:
            aff = ns.get("original_affine")
    if aff is None and isinstance(props, dict):
        aff = props.get("affine")
    if aff is None:
        return 0, False
    aff = np.asarray(aff, dtype=float)
    lr_axis = int(np.argmax(np.abs(aff[0, :3])))
    left_is_high = bool(aff[0, lr_axis] < 0)
    return lr_axis, left_is_high


def _list_nii(folder: str) -> List[str]:
    out = []
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn.startswith(".") or fn.startswith("._"):
                continue
            if fn.lower().endswith(NII_EXT):
                out.append(os.path.join(root, fn))
    return sorted(out)


def load_structure_set(folder: str, reader) -> StructureSet:
    """Load image + per-organ masks from a folder, co-registered on one grid.

    `reader` must be an nnU-Net ``NibabelIOWithReorient`` (or compatible object
    exposing ``read_images([path]) -> (data, props)``).
    """
    paths = _list_nii(folder)
    if not paths:
        raise ValueError("no .nii/.nii.gz files found in the folder")

    loaded = {}  # path -> (arr3d, props, n_unique)
    for p in paths:
        data, props = reader.read_images([p])
        arr = np.asarray(data[0] if getattr(data, "ndim", 3) == 4 else data)
        sample = arr.ravel()
        if sample.size > 200_000:
            sample = sample[:: sample.size // 200_000 + 1]
        loaded[p] = (arr, props, int(np.unique(sample).size))

    # identify the scan: a name hint first, else the file with the most levels
    image_path = None
    for p in paths:
        toks = _norm(_stem(p)).split()
        if any(h in toks for h in IMAGE_NAME_HINTS):
            image_path = p
            break
    if image_path is None:
        image_path = max(paths, key=lambda p: loaded[p][2])

    image, props, _ = loaded[image_path]
    spacing = props.get("spacing") if isinstance(props, dict) else None
    lr_axis, left_is_high = _lr_axis_and_dir(props)

    warnings: List[str] = []
    structures: Dict[str, dict] = {}
    for p in paths:
        if p == image_path:
            continue
        arr, _mprops, nuniq = loaded[p]
        name = _display_name(_stem(p))
        if arr.shape != image.shape:
            warnings.append(f'"{name}": shape {tuple(arr.shape)} != image '
                            f"{tuple(image.shape)}; skipped")
            continue
        if nuniq > 2:
            warnings.append(f'"{name}": not binary ({nuniq} levels); thresholded at >0')
        structures[name] = {"mask": arr > 0, "path": p, "stem": _stem(p)}

    if not structures:
        warnings.append("no structure masks found besides the image")

    return StructureSet(
        image=np.asarray(image, dtype=np.float32),
        spacing=list(spacing) if spacing is not None else None,
        props=props,
        lr_axis=lr_axis,
        left_is_high_index=left_is_high,
        structures=structures,
        warnings=warnings,
    )
