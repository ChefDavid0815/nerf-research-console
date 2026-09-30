"""Chunked preview rendering must preserve source pixel alignment."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nerf_step1.dataset import CameraIntrinsics  # noqa: E402
from nerf_step1.pipeline import CoarseFineNeRF  # noqa: E402
from nerf_step1.preview import aligned_ground_truth, preview_pixel_grid, render_image, save_preview  # noqa: E402
from test_pipeline import tiny_config  # noqa: E402


def test_preview_grid_and_target_alignment() -> None:
    intrinsics = CameraIntrinsics(8, 8, 4.0, 4.0, 4.0, 4.0)
    image = torch.arange(8 * 8 * 3, dtype=torch.float32).reshape(8, 8, 3) / (8 * 8 * 3)
    grid = preview_pixel_grid(intrinsics, (3, 3))
    target = aligned_ground_truth(image, intrinsics, (3, 3))
    assert tuple(grid[0, 0].tolist()) == (0, 0)
    assert tuple(grid[-1, -1].tolist()) == (7, 7)
    torch.testing.assert_close(target[1, 1], image[4, 4])


def test_render_image_chunks_and_save_iteration_files(tmp_path: Path) -> None:
    config = tiny_config()
    model = CoarseFineNeRF(config)
    intrinsics = CameraIntrinsics(4, 4, 4.0, 4.0, 2.0, 2.0)
    pose = torch.eye(4, dtype=torch.float64)
    calls: list[int] = []
    original_forward = model.forward

    def counted_forward(origins, directions, **kwargs):
        calls.append(len(origins))
        return original_forward(origins, directions, **kwargs)

    model.forward = counted_forward
    predicted = render_image(model, pose, intrinsics, (4, 4), render_chunk_size=3)
    assert predicted.shape == (4, 4, 3)
    assert len(calls) == 6 and max(calls) <= 3
    assert bool(torch.isfinite(predicted).all())
    assert model.training  # render_image restores the caller's mode
    target = torch.ones_like(predicted)
    paths = save_preview(tmp_path, 5, predicted, target)
    assert all(path.is_file() for path in paths.values())
    assert all("iter_000005" in path.name for path in paths.values())
    with pytest.raises(FileExistsError):
        save_preview(tmp_path, 5, predicted, target)
