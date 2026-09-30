"""Save reproducible Step 4 visual evidence from full-image evaluation tensors.

The caller must compute metrics on the unquantized float tensors. These PNGs
are display artifacts only; rounding to 8-bit is never used for scoring.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _rgb_float(image: torch.Tensor | np.ndarray, name: str) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3 or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be a numeric [H, W, 3] RGB image")
    if array.shape[0] < 1 or array.shape[1] < 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must have nonempty, finite pixels")
    # A float32 renderer can overshoot a sigmoid/white composite by a few
    # ULPs. Match the evaluation wrapper's 1e-6 tolerance and clamp once.
    if np.any((array < -1e-6) | (array > 1 + 1e-6)):
        raise ValueError(f"{name} must use the [0, 1] RGB convention")
    return np.clip(array, 0, 1)


def render_image_paths(run_directory: Path, iteration: int, split: str,
                       view_index: int) -> dict[str, Path]:
    """Return names that unambiguously retain checkpoint, split, and view."""
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("iteration must be a nonnegative integer")
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val, or test")
    if isinstance(view_index, bool) or not isinstance(view_index, int) or view_index < 0:
        raise ValueError("view_index must be a nonnegative integer")
    folder = Path(run_directory) / "evaluation" / "renders" / f"iter_{iteration:06d}"
    stem = f"{split}_view_{view_index:03d}"
    return {key: folder / f"{stem}_{key}.png" for key in
            ("ground_truth", "prediction", "absolute_difference")}


def save_evaluation_view(run_directory: Path, iteration: int, split: str,
                         view_index: int, prediction: torch.Tensor | np.ndarray,
                         ground_truth: torch.Tensor | np.ndarray) -> dict[str, Path]:
    """Save a rendered full view and its aligned source image without resizing."""
    paths = render_image_paths(run_directory, iteration, split, view_index)
    pred_float = _rgb_float(prediction, "prediction")
    truth_float = _rgb_float(ground_truth, "ground_truth")
    if pred_float.shape != truth_float.shape:
        raise ValueError("prediction and ground_truth must have identical full-image shape")
    paths["prediction"].parent.mkdir(parents=True, exist_ok=True)
    pred = np.rint(pred_float * 255).astype(np.uint8)
    truth = np.rint(truth_float * 255).astype(np.uint8)
    # The display difference uses clipped float RGB before 8-bit rounding,
    # matching the evaluator's image convention without scoring on PNGs.
    difference = np.rint(np.abs(pred_float - truth_float) * 255).astype(np.uint8)
    for key, pixels in (("prediction", pred), ("ground_truth", truth),
                        ("absolute_difference", difference)):
        path = paths[key]
        if path.exists():
            with Image.open(path) as existing:
                if existing.mode != "RGB" or not np.array_equal(np.asarray(existing), pixels):
                    raise ValueError(f"existing evaluation render differs from current tensors: {path}")
            continue
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".png", dir=path.parent)
        os.close(descriptor)
        try:
            Image.fromarray(pixels).save(temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return paths
