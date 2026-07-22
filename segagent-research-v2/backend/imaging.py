from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class VolumeData:
    data: np.ndarray
    affine: np.ndarray
    header: nib.Nifti1Header

    @property
    def spacing(self) -> tuple[float, float, float]:
        zooms = self.header.get_zooms()[:3]
        return tuple(float(abs(item)) for item in zooms)


def load_volume(path: Path) -> VolumeData:
    image = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asanyarray(image.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"expected a 3D NIfTI volume, got shape {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError("volume contains NaN or infinite values")
    return VolumeData(
        data=np.asarray(data, dtype=np.float32),
        affine=np.asarray(image.affine, dtype=np.float64),
        header=image.header.copy(),
    )


def load_mask(path: Path, reference: VolumeData) -> np.ndarray:
    image = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asanyarray(image.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.shape != reference.data.shape:
        raise ValueError(
            f"mask shape {data.shape} does not match image {reference.data.shape}"
        )
    if not np.allclose(image.affine, reference.affine, atol=1e-3, rtol=1e-4):
        raise ValueError("mask affine/grid does not match the case image")
    return np.asarray(data > 0, dtype=bool)


def save_mask(mask: np.ndarray, reference: VolumeData, path: Path) -> None:
    array = np.asarray(mask, dtype=np.uint8)
    if array.shape != reference.data.shape:
        raise ValueError("cannot save a mask on a different image grid")
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(array, reference.affine, header), str(path))


def window_bounds(volume: np.ndarray) -> tuple[float, float]:
    nonzero = volume[np.isfinite(volume)]
    if nonzero.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(nonzero, [1.0, 99.0])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def to_uint8(array: np.ndarray, lo: float, hi: float) -> np.ndarray:
    scaled = np.clip((array - lo) / (hi - lo), 0.0, 1.0)
    return np.asarray(scaled * 255.0, dtype=np.uint8)


def grounding_views(volume: np.ndarray, n_axial: int) -> list[tuple[str, Image.Image]]:
    lo, hi = window_bounds(volume)
    depth = volume.shape[2]
    indices = np.linspace(depth * 0.1, depth * 0.9, n_axial).round().astype(int)
    indices = np.clip(indices, 0, depth - 1)
    views: list[tuple[str, Image.Image]] = []
    for z in indices:
        array = to_uint8(volume[..., int(z)], lo, hi).T[::-1]
        views.append((f"axial z={int(z)}", Image.fromarray(array).convert("RGB")))
    coronal = to_uint8(volume[:, volume.shape[1] // 2, :], lo, hi).T[::-1]
    sagittal = to_uint8(volume[volume.shape[0] // 2, :, :], lo, hi).T[::-1]
    views.append(("coronal mid-plane", Image.fromarray(coronal).convert("RGB")))
    views.append(("sagittal mid-plane", Image.fromarray(sagittal).convert("RGB")))
    return views


def render_overlays(
    volume: np.ndarray, mask: np.ndarray, count: int
) -> list[tuple[int, Image.Image]]:
    binary = np.asarray(mask, dtype=bool)
    areas = binary.sum(axis=(0, 1))
    z_values = [int(z) for z in np.argsort(areas)[::-1] if areas[z] > 0][:count]
    z_values.sort()
    lo, hi = window_bounds(volume)
    output: list[tuple[int, Image.Image]] = []
    for z in z_values:
        base = to_uint8(volume[..., z], lo, hi).astype(np.float32)
        rgb = np.stack([base, base, base], axis=-1)
        rgb[binary[..., z]] = 0.45 * rgb[binary[..., z]] + 0.55 * np.array(
            [255.0, 40.0, 40.0], dtype=np.float32
        )
        oriented = np.transpose(rgb.astype(np.uint8), (1, 0, 2))[::-1]
        output.append((z, Image.fromarray(oriented)))
    return output


def mask_geometry(mask: np.ndarray) -> tuple[int, list[list[int]] | None, list[float] | None]:
    coordinates = np.argwhere(np.asarray(mask) > 0)
    if coordinates.size == 0:
        return 0, None, None
    lower = coordinates.min(axis=0).astype(int).tolist()
    upper = coordinates.max(axis=0).astype(int).tolist()
    centroid = coordinates.mean(axis=0).round(2).tolist()
    return int(coordinates.shape[0]), [lower, upper], centroid

