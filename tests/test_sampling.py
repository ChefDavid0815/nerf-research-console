"""Geometry, stratification and reproducibility checks for coarse ray samples."""

import pytest
import torch

from nerf_step1.sampling import sample_along_rays, sample_distances_from_depths


def test_midpoint_samples_follow_non_unit_step1_rays_and_physical_distances():
    origins = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    directions = torch.tensor([[0.0, 0.0, -2.0]], dtype=torch.float64)

    samples = sample_along_rays(origins, directions, near=2.0, far=6.0, num_samples=4)

    torch.testing.assert_close(samples.depths, torch.tensor([[2.5, 3.5, 4.5, 5.5]], dtype=torch.float64))
    torch.testing.assert_close(samples.points[0, :, 2], torch.tensor([-4.0, -6.0, -8.0, -10.0], dtype=torch.float64))
    torch.testing.assert_close(samples.distances, torch.tensor([[2.0, 2.0, 2.0, 2e10]], dtype=torch.float64))
    assert samples.sample_points is samples.points
    assert samples.sample_depths is samples.depths
    assert samples.sample_distances is samples.distances


def test_stratified_sample_lies_in_each_bin_and_seed_repeats():
    origins = torch.zeros((7, 3))
    directions = torch.tensor([[0.0, 0.0, -1.0]]).expand_as(origins)
    first = sample_along_rays(
        origins, directions, 2.0, 6.0, 8, randomized=True,
        generator=torch.Generator().manual_seed(213),
    )
    second = sample_along_rays(
        origins, directions, 2.0, 6.0, 8, randomized=True,
        generator=torch.Generator().manual_seed(213),
    )
    assert torch.equal(first.depths, second.depths)
    assert torch.equal(first.points, second.points)
    lower = torch.linspace(2.0, 6.0, 9)[:-1]
    upper = torch.linspace(2.0, 6.0, 9)[1:]
    assert bool((first.depths >= lower).all())
    assert bool((first.depths < upper).all())
    assert bool((first.depths[:, 1:] > first.depths[:, :-1]).all())
    assert not torch.equal(first.depths[0], first.depths[1])


def test_finite_terminal_interval_is_available_for_diagnostics():
    depths = torch.tensor([[2.5, 3.5]], dtype=torch.float64)
    directions = torch.tensor([[0.0, 0.0, -2.0]], dtype=torch.float64)
    torch.testing.assert_close(
        sample_distances_from_depths(depths, directions, far=4.0),
        torch.tensor([[2.0, 1.0]], dtype=torch.float64),
    )


def test_single_sample_and_empty_batch_have_consistent_shapes():
    origins = torch.zeros((1, 3))
    directions = torch.tensor([[0.0, 0.0, -1.0]])
    single = sample_along_rays(origins, directions, 2.0, 6.0, 1)
    assert single.points.shape == (1, 1, 3)
    assert single.depths.item() == 4.0
    assert single.distances.item() == 1e10

    empty = sample_along_rays(origins[:0], directions[:0], 2.0, 6.0, 4)
    assert empty.points.shape == (0, 4, 3)
    assert empty.depths.shape == empty.distances.shape == (0, 4)


@pytest.mark.parametrize("near,far,count", [(2.0, 2.0, 4), (5.0, 2.0, 4), (2.0, 6.0, 0), (2.0, 6.0, True)])
def test_invalid_bounds_and_count_are_rejected(near, far, count):
    with pytest.raises(ValueError):
        sample_along_rays(torch.zeros((1, 3)), torch.tensor([[0.0, 0.0, -1.0]]), near, far, count)


def test_unsorted_depths_and_zero_direction_are_rejected():
    with pytest.raises(ValueError, match="sorted"):
        sample_distances_from_depths(torch.tensor([[3.0, 2.0]]), torch.tensor([[0.0, 0.0, -1.0]]))
    with pytest.raises(ValueError, match="nonzero"):
        sample_along_rays(torch.zeros((1, 3)), torch.zeros((1, 3)), 2.0, 6.0, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_midpoints_match_cpu_and_randomized_seed_is_reproducible():
    origins = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    directions = torch.tensor([[0.0, 0.0, -2.0], [0.0, 1.0, -1.0]])
    cpu = sample_along_rays(origins, directions, 2.0, 6.0, 8)
    gpu = sample_along_rays(origins.cuda(), directions.cuda(), 2.0, 6.0, 8)
    for cpu_tensor, gpu_tensor in zip(cpu, gpu):
        torch.testing.assert_close(gpu_tensor.cpu(), cpu_tensor, rtol=1e-6, atol=1e-6)

    first = sample_along_rays(
        origins.cuda(), directions.cuda(), 2.0, 6.0, 8, randomized=True,
        generator=torch.Generator(device="cuda").manual_seed(47),
    )
    second = sample_along_rays(
        origins.cuda(), directions.cuda(), 2.0, 6.0, 8, randomized=True,
        generator=torch.Generator(device="cuda").manual_seed(47),
    )
    assert torch.equal(first.depths, second.depths)
    assert bool(torch.isfinite(first.distances).all())
