from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Protocol

import numpy as np

from .imaging import VolumeData, load_volume


class SegmentationBackend(Protocol):
    name: str
    version: str

    def segment(self, image_path: Path, structures: list[str]) -> tuple[VolumeData, np.ndarray]:
        """Return canonical image data and masks shaped (N, X, Y, Z)."""

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return one embedding per input string."""


class VoxTellBackend:
    """Lazy, lock-protected adapter around the VoxTell predictor."""

    name = "VoxTell"

    def __init__(self, model_dir: Path | None, device: str = "auto"):
        self.model_dir = Path(model_dir) if model_dir else None
        self.device_name = device
        self.version = self.model_dir.name if self.model_dir else "unconfigured"
        self._predictor = None
        self._lock = threading.RLock()

    def _ensure(self):
        if self._predictor is not None:
            return self._predictor
        if self.model_dir is None:
            raise RuntimeError("SEGAGENT_VOXTELL_MODEL is not configured")
        with self._lock:
            if self._predictor is not None:
                return self._predictor
            import torch
            from voxtell.inference.predictor import VoxTellPredictor

            if self.device_name == "auto":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device(self.device_name)
            predictor = VoxTellPredictor(model_dir=str(self.model_dir), device=device)
            predictor.perform_everything_on_device = False
            self._predictor = predictor
            return predictor

    def segment(self, image_path: Path, structures: list[str]) -> tuple[VolumeData, np.ndarray]:
        # VoxTell was trained on images read by nnU-Net's NibabelIOWithReorient,
        # which reorients to RAS AND transposes the spatial axes to (z, y, x).
        # Reading with a plain as_closest_canonical (x, y, z) feeds the model
        # axis-transposed data and produces poor masks, so use the same reader
        # and transpose the result back to the (x, y, z) frame the rest of the
        # pipeline (geometry, overlays, save_mask, QC) works in.
        from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

        volume = load_volume(image_path)
        predictor = self._ensure()
        with self._lock:
            image, _ = NibabelIOWithReorient().read_images([str(image_path)])
            masks = predictor.predict_single_image(image, structures)
        array = np.asarray(masks, dtype=np.uint8)
        if array.ndim == 4:
            array = array.transpose(0, 3, 2, 1)  # (N, z, y, x) -> (N, x, y, z)
        if array.shape != (len(structures), *volume.data.shape):
            raise RuntimeError(f"unexpected VoxTell output shape: {array.shape}")
        return volume, array

    def embed(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            tensor = self._ensure().embed_text_prompts(texts)
        return tensor[0].float().cpu().numpy()


class DeterministicFakeBackend:
    """Small deterministic backend used by tests and trajectory evaluations."""

    name = "deterministic-fake"
    version = "1"

    def segment(self, image_path: Path, structures: list[str]) -> tuple[VolumeData, np.ndarray]:
        volume = load_volume(image_path)
        masks = np.zeros((len(structures), *volume.data.shape), dtype=np.uint8)
        center = np.asarray(volume.data.shape) // 2
        grid = np.indices(volume.data.shape)
        for index, structure in enumerate(structures):
            radius = 2 + (sum(structure.encode("utf-8")) % max(min(volume.data.shape) // 5, 2))
            offset = index - len(structures) // 2
            distance = (
                (grid[0] - center[0] - offset) ** 2
                + (grid[1] - center[1]) ** 2
                + (grid[2] - center[2]) ** 2
            )
            masks[index] = distance <= radius**2
        return volume, masks

    def embed(self, texts: list[str]) -> np.ndarray:
        output = np.zeros((len(texts), 32), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.casefold().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "big") % output.shape[1]
                output[row, bucket] += 1.0
        return output
