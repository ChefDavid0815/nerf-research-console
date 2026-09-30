"""Chunked deterministic preview rendering and aligned image export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .dataset import CameraIntrinsics
from .pipeline import CoarseFineNeRF
from .rays import generate_rays


def preview_pixel_grid(
    intrinsics: CameraIntrinsics, resolution: tuple[int, int]
) -> torch.Tensor:
    """Select integer source pixels so prediction and GT use identical rays."""
    width, height = resolution
    if not (isinstance(width, int) and isinstance(height, int)) or width <= 0 or height <= 0:
        raise ValueError("resolution must be positive integer (width, height)")
    if width > intrinsics.width or height > intrinsics.height:
        raise ValueError("preview resolution cannot exceed the source image resolution")
    xs = torch.linspace(0, intrinsics.width - 1, width).round().long()
    ys = torch.linspace(0, intrinsics.height - 1, height).round().long()
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1)


@torch.no_grad()
def render_image(
    model: CoarseFineNeRF,
    camera_to_world: torch.Tensor,
    intrinsics: CameraIntrinsics,
    resolution: tuple[int, int],
    *,
    render_chunk_size: int,
) -> torch.Tensor:
    """Render final fine RGB on a source-aligned grid, returning CPU [H,W,3]."""
    if not isinstance(render_chunk_size, int) or render_chunk_size <= 0:
        raise ValueError("render_chunk_size must be a positive integer")
    grid = preview_pixel_grid(intrinsics, resolution)
    pixels = grid.reshape(-1, 2)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    predictions: list[torch.Tensor] = []
    try:
        for start in range(0, len(pixels), render_chunk_size):
            origins, directions = generate_rays(
                camera_to_world, intrinsics, pixels[start:start + render_chunk_size],
                device=device, dtype=torch.float32,
                normalize_directions=bool(model.config["normalize_ray_directions"]),
            )
            result = model(origins, directions, randomized=False)
            predictions.append(result.fine.rgb_map.cpu())
    finally:
        model.train(was_training)
    width, height = resolution
    return torch.cat(predictions).reshape(height, width, 3)


def aligned_ground_truth(
    source_rgb: torch.Tensor, intrinsics: CameraIntrinsics, resolution: tuple[int, int]
) -> torch.Tensor:
    if source_rgb.shape != (intrinsics.height, intrinsics.width, 3):
        raise ValueError("ground truth dimensions do not match intrinsics")
    grid = preview_pixel_grid(intrinsics, resolution)
    return source_rgb[grid[..., 1], grid[..., 0]]


def _rgb_image(rgb: torch.Tensor) -> Image.Image:
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or not bool(torch.isfinite(rgb).all()):
        raise ValueError("preview image must be finite [H, W, 3]")
    pixels = (rgb.detach().cpu().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(pixels)


def save_preview(
    run_directory: Path,
    iteration: int,
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
) -> dict[str, Path]:
    """Write iteration-addressed prediction, target, absolute error and contact sheet."""
    if not isinstance(iteration, int) or iteration < 0:
        raise ValueError("iteration must be a nonnegative integer")
    if prediction.shape != ground_truth.shape:
        raise ValueError("prediction and ground truth must have the same shape")
    render_directory = Path(run_directory) / "renders"
    render_directory.mkdir(parents=True, exist_ok=True)
    suffix = f"iter_{iteration:06d}.png"
    paths = {
        "prediction": render_directory / f"prediction_{suffix}",
        "ground_truth": render_directory / f"ground_truth_{suffix}",
        "absolute_difference": render_directory / f"absolute_difference_{suffix}",
        "contact_sheet": render_directory / f"comparison_{suffix}",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"preview for iteration {iteration} already exists")
    images = {
        "prediction": _rgb_image(prediction),
        "ground_truth": _rgb_image(ground_truth),
        "absolute_difference": _rgb_image((prediction - ground_truth).abs()),
    }
    for key, image in images.items():
        image.save(paths[key])
    width, height = images["prediction"].size
    sheet = Image.new("RGB", (width * 3, height + 40), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    draw.text((5, 4), f"Iteration {iteration}", fill=(15, 15, 15))
    for index, (key, title) in enumerate(
        (("prediction", "Prediction"), ("ground_truth", "Ground truth"),
         ("absolute_difference", "Abs. difference"))
    ):
        sheet.paste(images[key], (index * width, 40))
        draw.text((index * width + 5, 21), title, fill=(15, 15, 15))
    sheet.save(paths["contact_sheet"])
    return paths
