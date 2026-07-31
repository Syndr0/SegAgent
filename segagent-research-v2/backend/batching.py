from __future__ import annotations

from typing import Any, Callable

import numpy as np


def predict_structure_chunks(
    predict: Callable[[Any, list[str]], Any],
    image: Any,
    structures: list[str],
    batch_size: int,
) -> np.ndarray:
    """Run an ordered structure list in bounded predictor batches."""

    if not structures:
        raise ValueError("at least one structure is required")
    size = max(1, int(batch_size))
    chunks: list[np.ndarray] = []
    spatial_shape: tuple[int, ...] | None = None
    for start in range(0, len(structures), size):
        prompts = structures[start : start + size]
        predicted = np.asarray(predict(image, prompts), dtype=np.uint8)
        if predicted.ndim == 3 and len(prompts) == 1:
            predicted = predicted[None, ...]
        if predicted.ndim != 4 or predicted.shape[0] != len(prompts):
            raise RuntimeError(
                "unexpected VoxTell chunk output shape: "
                f"{predicted.shape} for {len(prompts)} structures"
            )
        if spatial_shape is None:
            spatial_shape = predicted.shape[1:]
        elif predicted.shape[1:] != spatial_shape:
            raise RuntimeError(
                "VoxTell returned inconsistent spatial shapes across chunks: "
                f"{spatial_shape} and {predicted.shape[1:]}"
            )
        chunks.append(predicted)
    return np.concatenate(chunks, axis=0)
