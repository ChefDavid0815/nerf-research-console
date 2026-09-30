"""Differentiable coarse-to-fine Vanilla NeRF ray pipeline.

All positions use Step 1's ray parameterization. The sampling module converts
parameter gaps into world distances for the density integral; viewing
directions alone are normalized before the unchanged Step 2 MLP sees them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import nn
from torch.nn import functional as F

from .hierarchical import merge_and_sort_depths, sample_pdf
from .model import VanillaNeRF
from .sampling import sample_along_rays, sample_distances_from_depths
from .volume_rendering import RenderOutput, volume_render


REQUIRED_TRAINING_KEYS = (
    "scene", "dataset_root", "image_resolution", "white_background",
    "normalize_ray_directions", "position_encoding_L", "direction_encoding_L",
    "network_depth", "network_width", "skip_connection_layer", "view_width",
    "num_coarse_samples", "num_fine_samples", "batch_size", "learning_rate",
    "training_iterations", "random_seed", "density_initial_bias", "near", "far",
    "learning_rate_decay_steps", "learning_rate_decay_factor", "ray_chunk_size",
    "render_chunk_size", "checkpoint_interval", "preview_interval",
    "preview_width", "preview_height", "progress_interval",
)


def validate_training_config(config: Mapping[str, object]) -> None:
    """Reject missing or physically impossible scientific and run settings."""
    missing = [key for key in REQUIRED_TRAINING_KEYS if key not in config]
    if missing:
        raise ValueError(f"training config missing keys: {', '.join(missing)}")
    positive_ints = (
        "num_coarse_samples", "num_fine_samples", "batch_size", "training_iterations",
        "ray_chunk_size", "render_chunk_size", "checkpoint_interval",
        "preview_interval", "preview_width", "preview_height", "progress_interval",
        "learning_rate_decay_steps",
    )
    for key in positive_ints:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["num_coarse_samples"] < 4:
        raise ValueError("num_coarse_samples must be at least four for interior PDF bins")
    if isinstance(config["random_seed"], bool) or not isinstance(config["random_seed"], int):
        raise ValueError("random_seed must be an integer")
    for key in ("near", "far", "learning_rate", "learning_rate_decay_factor", "density_initial_bias"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
    if not 0 < config["near"] < config["far"]:
        raise ValueError("near and far must satisfy 0 < near < far")
    if config["learning_rate"] <= 0 or not 0 < config["learning_rate_decay_factor"] <= 1:
        raise ValueError("learning rate must be positive and decay factor in (0, 1]")
    if config["density_initial_bias"] < 0:
        raise ValueError("density_initial_bias must be nonnegative")
    if config["scene"] != "lego" or config["image_resolution"] != "original":
        raise ValueError("Step 3 supports the original-resolution Lego scene only")
    if config["white_background"] is not True:
        raise ValueError("Step 3 Lego targets and renderer require white_background: true")
    if not isinstance(config["normalize_ray_directions"], bool):
        raise ValueError("normalize_ray_directions must be boolean")
    VanillaNeRF.from_config(config)  # Validates the unchanged Step 2 architecture.


@dataclass(frozen=True)
class RayRenderResult:
    coarse: RenderOutput
    fine: RenderOutput
    coarse_depths: torch.Tensor
    fine_depths: torch.Tensor
    combined_depths: torch.Tensor


class CoarseFineNeRF(nn.Module):
    """Two independent instances of the verified Step 2 VanillaNeRF."""

    def __init__(self, config: Mapping[str, object]) -> None:
        super().__init__()
        validate_training_config(config)
        self.config = MappingProxyType(dict(config))
        self.coarse_model = VanillaNeRF.from_config(config)
        self.fine_model = VanillaNeRF.from_config(config)
        # A fully negative ReLU density head produces transparent rays and no
        # density gradient. The explicit run setting prevents a dead start
        # while retaining the verified Step 2 architecture and output rule.
        for network in (self.coarse_model, self.fine_model):
            nn.init.constant_(network.density_layer.bias, float(config["density_initial_bias"]))

    @staticmethod
    def _query(model: VanillaNeRF, points: torch.Tensor, rays_d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_rays, n_samples, _ = points.shape
        unit_viewdirs = F.normalize(rays_d, dim=-1)
        if bool((torch.linalg.vector_norm(rays_d, dim=-1) == 0).any()):
            raise ValueError("ray directions must be nonzero")
        expanded_dirs = unit_viewdirs[:, None, :].expand(-1, n_samples, -1)
        rgb, sigma = model(points.reshape(-1, 3), expanded_dirs.reshape(-1, 3))
        return rgb.reshape(n_rays, n_samples, 3), sigma.reshape(n_rays, n_samples)

    def forward(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        *,
        randomized: bool = False,
        generator: torch.Generator | None = None,
    ) -> RayRenderResult:
        if ray_origins.ndim != 2 or ray_origins.shape[1] != 3 or ray_directions.shape != ray_origins.shape:
            raise ValueError("ray_origins and ray_directions must have shape [N, 3]")
        if ray_origins.device != ray_directions.device or ray_origins.dtype != ray_directions.dtype:
            raise ValueError("ray origins and directions must share dtype and device")
        cfg = self.config
        coarse_samples = sample_along_rays(
            ray_origins, ray_directions, float(cfg["near"]), float(cfg["far"]),
            int(cfg["num_coarse_samples"]), randomized=randomized, generator=generator,
        )
        coarse_rgb, coarse_sigma = self._query(
            self.coarse_model, coarse_samples.points, ray_directions
        )
        coarse = volume_render(
            coarse_rgb, coarse_sigma, coarse_samples.depths, coarse_samples.distances,
            white_background=True,
        )

        # Original NeRF samples from midpoint edges with interior coarse
        # weights, then stops gradients through the PDF draw.
        midpoint_edges = 0.5 * (coarse_samples.depths[..., 1:] + coarse_samples.depths[..., :-1])
        fine_depths = sample_pdf(
            midpoint_edges.detach(), coarse.weights[..., 1:-1].detach(),
            int(cfg["num_fine_samples"]), randomized=randomized, generator=generator,
        ).detach()
        combined_depths = merge_and_sort_depths(coarse_samples.depths, fine_depths)
        fine_points = ray_origins[:, None, :] + combined_depths[..., None] * ray_directions[:, None, :]
        fine_distances = sample_distances_from_depths(combined_depths, ray_directions)
        fine_rgb, fine_sigma = self._query(self.fine_model, fine_points, ray_directions)
        fine = volume_render(
            fine_rgb, fine_sigma, combined_depths, fine_distances, white_background=True
        )
        return RayRenderResult(
            coarse=coarse, fine=fine, coarse_depths=coarse_samples.depths,
            fine_depths=fine_depths, combined_depths=combined_depths,
        )


def nerf_loss(result: RayRenderResult, ground_truth_rgb: torch.Tensor) -> dict[str, torch.Tensor]:
    """Original NeRF RGB MSE at both levels; PSNR uses final fine MSE."""
    if ground_truth_rgb.shape != result.fine.rgb_map.shape:
        raise ValueError("ground_truth_rgb must have shape [N, 3]")
    coarse_loss = F.mse_loss(result.coarse.rgb_map, ground_truth_rgb)
    fine_loss = F.mse_loss(result.fine.rgb_map, ground_truth_rgb)
    total_loss = coarse_loss + fine_loss
    psnr = -10.0 * torch.log10(fine_loss.clamp_min(1e-12))
    return {"total_loss": total_loss, "coarse_loss": coarse_loss, "fine_loss": fine_loss, "PSNR": psnr}


def train_step(
    model: CoarseFineNeRF,
    optimizer: torch.optim.Optimizer,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    ground_truth_rgb: torch.Tensor,
    *,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, float]:
    """One optimizer update over a full ray batch, with bounded graph memory."""
    if ray_origins.shape != ray_directions.shape or ray_origins.ndim != 2 or ray_origins.shape[1] != 3:
        raise ValueError("ray origins and directions must have shape [N, 3]")
    if ground_truth_rgb.shape != ray_origins.shape or ray_origins.shape[0] == 0:
        raise ValueError("ground truth must have shape [N, 3] for a nonempty batch")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {key: 0.0 for key in ("total_loss", "coarse_loss", "fine_loss", "PSNR")}
    batch_size = ray_origins.shape[0]
    chunk_size = int(model.config["ray_chunk_size"])
    for start in range(0, batch_size, chunk_size):
        end = min(start + chunk_size, batch_size)
        result = model(ray_origins[start:end], ray_directions[start:end], randomized=True)
        loss = nerf_loss(result, ground_truth_rgb[start:end])
        fraction = (end - start) / batch_size
        (loss["total_loss"] * fraction).backward()
        for key in ("total_loss", "coarse_loss", "fine_loss"):
            totals[key] += float(loss[key].detach()) * fraction
    # Averaged MSE across chunks is the actual global batch MSE.
    totals["PSNR"] = -10.0 * math.log10(max(totals["fine_loss"], 1e-12))
    for network_name in ("coarse_model", "fine_model"):
        network = getattr(model, network_name)
        gradients = [p.grad for p in network.parameters() if p.grad is not None]
        if not gradients or any(not bool(torch.isfinite(grad).all()) for grad in gradients):
            raise FloatingPointError(f"{network_name} has missing or nonfinite gradients")
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return totals


def make_scheduler(
    optimizer: torch.optim.Optimizer, config: Mapping[str, object]
) -> torch.optim.lr_scheduler.LambdaLR:
    """Exponential LR decay, parameterized entirely by the run snapshot."""
    steps = int(config["learning_rate_decay_steps"])
    factor = float(config["learning_rate_decay_factor"])
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda iteration: factor ** (iteration / steps)
    )
