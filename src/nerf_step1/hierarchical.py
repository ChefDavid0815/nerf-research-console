"""Inverse-CDF importance sampling for the coarse-to-fine NeRF pass.

The caller supplies one weight per interval between consecutive ``bin_edges``.
For the original NeRF midpoint construction, use the midpoints of adjacent
coarse depths as edges and the interior coarse rendering weights as weights.
The fine pass should detach the returned depths before querying its network,
as in the released NeRF implementation; this utility leaves that choice with
the caller.
"""

from __future__ import annotations

import torch


def sample_pdf(
    bin_edges: torch.Tensor,
    weights: torch.Tensor,
    num_samples: int,
    *,
    randomized: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample depths from a piecewise-constant PDF by inverse CDF.

    ``bin_edges`` has shape ``[N, M+1]`` and strictly increasing depths along
    each ray. ``weights`` has shape ``[N, M]`` and must be finite and
    nonnegative. As in the original NeRF sampler, adding a small weight to
    each bin makes zero and near-zero weight rays well defined. Deterministic
    sampling uses evenly spaced *midpoint* quantiles; stochastic sampling uses
    ``generator`` when provided. The generator must belong to the input
    device type. The returned tensor has shape ``[N, num_samples]``.
    """
    if not isinstance(bin_edges, torch.Tensor) or not isinstance(weights, torch.Tensor):
        raise TypeError("bin_edges and weights must be torch.Tensor instances")
    if bin_edges.ndim != 2 or weights.ndim != 2:
        raise ValueError("bin_edges and weights must both have shape [N, M]")
    if bin_edges.shape[0] != weights.shape[0] or bin_edges.shape[1] != weights.shape[1] + 1:
        raise ValueError("bin_edges must have shape [N, M+1] for weights [N, M]")
    if weights.shape[1] == 0:
        raise ValueError("at least one PDF bin is required")
    if bin_edges.dtype not in (torch.float32, torch.float64) or weights.dtype != bin_edges.dtype:
        raise TypeError("bin_edges and weights must share a float32 or float64 dtype")
    if bin_edges.device != weights.device:
        raise ValueError("bin_edges and weights must be on the same device")
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")
    if not bool(torch.isfinite(bin_edges).all()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("bin_edges and weights must be finite")
    if bool((bin_edges[:, 1:] <= bin_edges[:, :-1]).any()):
        raise ValueError("bin_edges must increase strictly along each ray")
    if bool((weights < 0).any()):
        raise ValueError("weights must be nonnegative")
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator or None")

    # Original NeRF adds 1e-5 per interval before normalizing. This yields an
    # approximately uniform fallback when all rendering weights vanish.
    # Scaling by at least the row maximum preserves the NeRF 1e-5 smoothing
    # ratio while preventing a sum of large but finite weights from overflowing.
    scale = torch.maximum(weights.amax(dim=-1, keepdim=True), torch.ones_like(weights[:, :1]))
    smoothed_weights = (weights / scale) + (1e-5 / scale)
    pdf = smoothed_weights / smoothed_weights.sum(dim=-1, keepdim=True)
    cdf = torch.cat((torch.zeros_like(pdf[:, :1]), pdf.cumsum(dim=-1)), dim=-1)
    # Remove a possible rounding discrepancy at the right endpoint without
    # mutating a tensor needed by autograd.
    cdf = torch.cat((cdf[:, :-1], torch.ones_like(cdf[:, -1:])), dim=-1)

    if randomized:
        quantiles = torch.rand(
            (bin_edges.shape[0], num_samples),
            device=bin_edges.device,
            dtype=bin_edges.dtype,
            generator=generator,
        )
    else:
        quantiles = (
            (torch.arange(num_samples, device=bin_edges.device, dtype=bin_edges.dtype) + 0.5)
            / num_samples
        ).expand(bin_edges.shape[0], -1)

    upper = torch.searchsorted(cdf.contiguous(), quantiles.contiguous(), right=True)
    lower = (upper - 1).clamp(min=0, max=weights.shape[1])
    upper = upper.clamp(max=weights.shape[1])
    cdf_lower = torch.gather(cdf, 1, lower)
    cdf_upper = torch.gather(cdf, 1, upper)
    edge_lower = torch.gather(bin_edges, 1, lower)
    edge_upper = torch.gather(bin_edges, 1, upper)
    denominator = cdf_upper - cdf_lower
    # A very small bin probability may round to zero in the CDF. In that case
    # choose its lower edge; dividing by zero would contaminate the fine pass.
    fraction = (quantiles - cdf_lower) / torch.where(
        denominator > 0, denominator, torch.ones_like(denominator)
    )
    fraction = torch.where(denominator > 0, fraction, torch.zeros_like(fraction))
    return edge_lower + fraction * (edge_upper - edge_lower)


def merge_and_sort_depths(
    coarse_depths: torch.Tensor, fine_depths: torch.Tensor
) -> torch.Tensor:
    """Keep every coarse and fine depth, sorted in ascending order per ray.

    Equal input depths remain equal: preserving ``Nc + Nf`` samples makes a
    strict inequality impossible when duplicates are present.
    """
    if not isinstance(coarse_depths, torch.Tensor) or not isinstance(fine_depths, torch.Tensor):
        raise TypeError("coarse_depths and fine_depths must be torch.Tensor instances")
    if coarse_depths.ndim != 2 or fine_depths.ndim != 2:
        raise ValueError("coarse_depths and fine_depths must both have shape [N, S]")
    if coarse_depths.shape[0] != fine_depths.shape[0]:
        raise ValueError("coarse_depths and fine_depths must have the same ray count")
    if coarse_depths.dtype != fine_depths.dtype or coarse_depths.device != fine_depths.device:
        raise ValueError("coarse_depths and fine_depths must share device and dtype")
    if coarse_depths.dtype not in (torch.float32, torch.float64):
        raise TypeError("depths must have a float32 or float64 dtype")
    if not bool(torch.isfinite(coarse_depths).all()) or not bool(torch.isfinite(fine_depths).all()):
        raise ValueError("depths must be finite")
    return torch.sort(torch.cat((coarse_depths, fine_depths), dim=-1), dim=-1).values
