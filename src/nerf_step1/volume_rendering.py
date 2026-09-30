"""Numerically stable NeRF alpha compositing with an explicit white background."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch


class RenderOutput(NamedTuple):
    rgb_map: torch.Tensor
    weights: torch.Tensor
    alpha: torch.Tensor
    transmittance: torch.Tensor
    depth_map: torch.Tensor
    accumulated_opacity: torch.Tensor


def volume_render(
    rgb: torch.Tensor,
    sigma: torch.Tensor,
    depths: torch.Tensor,
    sample_distances: torch.Tensor,
    white_background: bool = True,
) -> RenderOutput:
    """Integrate RGB and density from ordered points on each ray.

    ``rgb`` is ``[N_rays, N_samples, 3]``. ``sigma`` is either
    ``[N_rays, N_samples]`` or ``[N_rays, N_samples, 1]``. ``depths`` and
    physical ``sample_distances`` are ``[N_rays, N_samples]``. Optical
    thickness ``tau=sigma*distance`` gives ``alpha=-expm1(-tau)`` and
    exclusive transmittance ``T_i=exp(-sum_{j<i} tau_j)``. This avoids a
    cumulative product of rounded ``1-alpha`` values. The depth map is the
    original NeRF-style unnormalized weighted depth sum.
    """
    for name, value in (
        ("rgb", rgb), ("sigma", sigma), ("depths", depths),
        ("sample_distances", sample_distances),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.dtype not in (torch.float32, torch.float64):
            raise TypeError(f"{name} must have dtype torch.float32 or torch.float64")
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.shape[1] < 1:
        raise ValueError("rgb must have shape [N_rays, N_samples, 3] with N_samples >= 1")
    sample_shape = rgb.shape[:2]
    if sigma.shape == (*sample_shape, 1):
        sigma = sigma.squeeze(-1)
    elif sigma.shape != sample_shape:
        raise ValueError("sigma must have shape [N_rays, N_samples] or [N_rays, N_samples, 1]")
    if depths.shape != sample_shape or sample_distances.shape != sample_shape:
        raise ValueError("depths and sample_distances must have shape [N_rays, N_samples]")
    if any(t.device != rgb.device or t.dtype != rgb.dtype for t in (sigma, depths, sample_distances)):
        raise ValueError("all rendering tensors must share a device and dtype")
    if not isinstance(white_background, bool):
        raise TypeError("white_background must be a bool")
    if any(not bool(torch.isfinite(t).all()) for t in (rgb, sigma, depths, sample_distances)):
        raise ValueError("rendering tensors must be finite")
    if bool((sigma < 0).any()):
        raise ValueError("sigma must be nonnegative")
    if bool((sample_distances < 0).any()):
        raise ValueError("sample_distances must be nonnegative")

    # Clipping only fully opaque optical thickness avoids inf in products and
    # cumulative sums. At this threshold exp(-tau) is already negligible.
    max_optical_depth = -math.log(torch.finfo(rgb.dtype).tiny)
    optical_depth = torch.clamp(sigma * sample_distances, max=max_optical_depth)
    alpha = -torch.expm1(-optical_depth)
    exclusive_optical_depth = torch.cat(
        (torch.zeros_like(optical_depth[:, :1]), optical_depth[:, :-1]), dim=-1
    ).cumsum(dim=-1)
    transmittance = torch.exp(-exclusive_optical_depth)
    weights = transmittance * alpha

    accumulated_opacity = weights.sum(dim=-1)
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    if white_background:
        rgb_map = rgb_map + (1.0 - accumulated_opacity).clamp_min(0)[..., None]
    depth_map = torch.sum(weights * depths, dim=-1)
    return RenderOutput(
        rgb_map=rgb_map,
        weights=weights,
        alpha=alpha,
        transmittance=transmittance,
        depth_map=depth_map,
        accumulated_opacity=accumulated_opacity,
    )
