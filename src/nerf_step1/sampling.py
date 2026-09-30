"""Stratified coarse samples along Step 1's (possibly non-unit) camera rays.

Depths are ray parameters in ``o + t*d``. Optical distances are physical
distances: each adjacent depth gap is multiplied by ``||d||``. The terminal
gap defaults to the original NeRF implementation's ``1e10`` ray-parameter
units, so the last sampled density can absorb any remaining transmittance.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import NamedTuple

import torch


class RaySamples(NamedTuple):
    points: torch.Tensor
    depths: torch.Tensor
    distances: torch.Tensor

    @property
    def sample_points(self) -> torch.Tensor:
        return self.points

    @property
    def sample_depths(self) -> torch.Tensor:
        return self.depths

    @property
    def sample_distances(self) -> torch.Tensor:
        return self.distances


def _finite_scalar(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_rays(ray_origins: torch.Tensor, ray_directions: torch.Tensor) -> None:
    for name, value in (("ray_origins", ray_origins), ("ray_directions", ray_directions)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 2 or value.shape[-1] != 3:
            raise ValueError(f"{name} must have shape [N_rays, 3]")
        if value.dtype not in (torch.float32, torch.float64):
            raise TypeError(f"{name} must have dtype torch.float32 or torch.float64")
    if ray_origins.shape != ray_directions.shape:
        raise ValueError("ray origins and directions must have the same shape")
    if ray_origins.device != ray_directions.device or ray_origins.dtype != ray_directions.dtype:
        raise ValueError("ray origins and directions must share a device and dtype")
    if not bool(torch.isfinite(ray_origins).all()) or not bool(torch.isfinite(ray_directions).all()):
        raise ValueError("ray origins and directions must be finite")
    if bool((torch.linalg.vector_norm(ray_directions, dim=-1) == 0).any()):
        raise ValueError("ray directions must be nonzero")


def sample_distances_from_depths(
    depths: torch.Tensor,
    ray_directions: torch.Tensor,
    far: Real | None = None,
) -> torch.Tensor:
    """Return physical intervals for sorted samples on each ray.

    The terminal interval is ``1e10 * ||d||`` by default, as in the released
    NeRF renderer. Supplying ``far`` instead uses ``(far - last_depth)*||d||``
    for a finite final interval. The leading interval from ``near`` to the
    first sample is omitted, following NeRF's left-endpoint quadrature.
    """
    if not isinstance(depths, torch.Tensor):
        raise TypeError("depths must be a torch.Tensor")
    if depths.ndim != 2 or depths.shape[1] < 1:
        raise ValueError("depths must have shape [N_rays, N_samples] with N_samples >= 1")
    if depths.dtype not in (torch.float32, torch.float64):
        raise TypeError("depths must have dtype torch.float32 or torch.float64")
    if not isinstance(ray_directions, torch.Tensor):
        raise TypeError("ray_directions must be a torch.Tensor")
    if ray_directions.shape != (depths.shape[0], 3):
        raise ValueError("ray_directions must have shape [N_rays, 3]")
    if depths.device != ray_directions.device or depths.dtype != ray_directions.dtype:
        raise ValueError("depths and ray_directions must share a device and dtype")
    if not bool(torch.isfinite(depths).all()) or not bool(torch.isfinite(ray_directions).all()):
        raise ValueError("depths and ray_directions must be finite")

    ray_lengths = torch.linalg.vector_norm(ray_directions, dim=-1, keepdim=True)
    if bool((ray_lengths == 0).any()):
        raise ValueError("ray directions must be nonzero")
    gaps = depths[:, 1:] - depths[:, :-1]
    if bool((gaps < 0).any()):
        raise ValueError("depths must be sorted in ascending order")
    if far is None:
        final_gap = torch.full_like(depths[:, :1], 1e10)
    else:
        far_value = _finite_scalar("far", far)
        final_gap = far_value - depths[:, -1:]
        if bool((final_gap < 0).any()):
            raise ValueError("far must be at or beyond the final sample depth")
    return torch.cat((gaps, final_gap), dim=-1) * ray_lengths


def sample_along_rays(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near: Real,
    far: Real,
    num_samples: int = 64,
    randomized: bool = False,
    generator: torch.Generator | None = None,
) -> RaySamples:
    """Take one point per uniform bin, randomly or at each fixed midpoint.

    ``randomized=False`` is the evaluation path. Pass a seeded generator to
    make the stratified training path reproducible on a given device.
    """
    _validate_rays(ray_origins, ray_directions)
    near_value = _finite_scalar("near", near)
    far_value = _finite_scalar("far", far)
    if far_value <= near_value:
        raise ValueError("far must be greater than near")
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError("num_samples must be a positive integer")
    if not isinstance(randomized, bool):
        raise TypeError("randomized must be a bool")
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator or None")

    edges = torch.linspace(
        near_value, far_value, num_samples + 1,
        device=ray_origins.device, dtype=ray_origins.dtype,
    )
    lower = edges[:-1]
    widths = edges[1:] - lower
    if randomized:
        offsets = torch.rand(
            (ray_origins.shape[0], num_samples),
            device=ray_origins.device, dtype=ray_origins.dtype, generator=generator,
        )
    else:
        offsets = torch.full(
            (ray_origins.shape[0], num_samples), 0.5,
            device=ray_origins.device, dtype=ray_origins.dtype,
        )
    depths = lower + offsets * widths
    points = ray_origins[:, None, :] + depths[..., None] * ray_directions[:, None, :]
    distances = sample_distances_from_depths(depths, ray_directions)
    return RaySamples(points=points, depths=depths, distances=distances)
