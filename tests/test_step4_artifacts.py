"""Recovery and identity checks for full-view Step 4 PNG evidence."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from step4_artifacts import render_image_paths, save_evaluation_view  # noqa: E402


def test_full_view_render_save_is_idempotent_and_recovers_partial_write(tmp_path: Path) -> None:
    truth = torch.zeros((3, 4, 3), dtype=torch.float32)
    prediction = torch.full_like(truth, 0.5)
    paths = render_image_paths(tmp_path, 500, "val", 2)
    saved = save_evaluation_view(tmp_path, 500, "val", 2, prediction, truth)
    assert saved == paths
    assert all(path.is_file() for path in paths.values())
    assert Image.open(paths["prediction"]).size == (4, 3)
    assert np.all(np.asarray(Image.open(paths["absolute_difference"])) == 128)

    # A process may stop after only one PNG is safely written, before the CSV
    # is exported. The next evaluation must repair missing files and keep the
    # already verified evidence unchanged.
    paths["ground_truth"].unlink()
    paths["absolute_difference"].unlink()
    save_evaluation_view(tmp_path, 500, "val", 2, prediction, truth)
    assert all(path.is_file() for path in paths.values())
    save_evaluation_view(tmp_path, 500, "val", 2, prediction, truth)

    with pytest.raises(ValueError, match="differs"):
        save_evaluation_view(tmp_path, 500, "val", 2, torch.zeros_like(prediction), truth)


def test_full_view_render_rejects_mismatched_or_invalid_inputs(tmp_path: Path) -> None:
    image = torch.zeros((2, 3, 3), dtype=torch.float32)
    tiny_overshoot = image.clone()
    tiny_overshoot[0, 0, 0] = 1.00000024
    paths = save_evaluation_view(tmp_path, 10, "test", 1, tiny_overshoot, image)
    assert np.asarray(Image.open(paths["prediction"]))[0, 0, 0] == 255
    with pytest.raises(ValueError, match="identical full-image shape"):
        save_evaluation_view(tmp_path, 10, "test", 0, image, image[:1])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        save_evaluation_view(tmp_path, 10, "test", 0, image + 2, image)
    with pytest.raises(ValueError, match="split"):
        render_image_paths(tmp_path, 10, "validation", 0)
