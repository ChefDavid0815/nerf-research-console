"""Mathematical and edge-case checks for NeRF hierarchical depth sampling."""

import pytest
import torch

from nerf_step1.hierarchical import merge_and_sort_depths, sample_pdf


def test_uniform_pdf_returns_evenly_spaced_midpoint_quantiles():
    edges = torch.arange(5, dtype=torch.float64).unsqueeze(0)
    weights = torch.ones((1, 4), dtype=torch.float64)

    samples = sample_pdf(edges, weights, 8)

    torch.testing.assert_close(
        samples,
        torch.tensor([[0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75]], dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )


def test_inverse_cdf_matches_manual_piecewise_interpolation():
    edges = torch.tensor([[0.0, 1.0, 3.0]], dtype=torch.float64)
    weights = torch.tensor([[1.0, 3.0]], dtype=torch.float64)

    actual = sample_pdf(edges, weights, 4)

    # The smoothed PDF is approximately [1/4, 3/4]. The first quantile
    # 1/8 lies halfway through [0, 1]; the remaining three interpolate in
    # [1, 3] at probabilities (3/8, 5/8, 7/8).
    expected = torch.tensor([[0.5, 4.0 / 3.0, 2.0, 8.0 / 3.0]], dtype=torch.float64)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)


def test_fine_samples_concentrate_in_high_weight_interval():
    edges = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])
    weights = torch.tensor([[0.0, 0.0, 20.0, 0.0]])

    samples = sample_pdf(edges, weights, 128)

    assert samples.shape == (1, 128)
    assert bool(((samples >= 2.0) & (samples <= 3.0)).float().mean() > 0.95)
    assert bool((samples[:, 1:] > samples[:, :-1]).all())


@pytest.mark.parametrize("weight_value", [0.0, 1e-12])
def test_zero_and_near_zero_weights_are_finite_and_bounded(weight_value):
    edges = torch.tensor([[2.0, 3.0, 4.0, 5.0]], dtype=torch.float64)
    weights = torch.full((1, 3), weight_value, dtype=torch.float64)

    samples = sample_pdf(edges, weights, 101)

    assert bool(torch.isfinite(samples).all())
    assert bool(((samples >= 2.0) & (samples <= 5.0)).all())
    torch.testing.assert_close(samples.mean(), torch.tensor(3.5, dtype=torch.float64), atol=1e-12, rtol=0)


def test_each_ray_stays_within_its_own_bounds_and_retains_dtype():
    edges = torch.tensor(
        [[1.0, 2.0, 5.0], [10.0, 11.0, 20.0]], dtype=torch.float32
    )
    weights = torch.tensor([[2.0, 1.0], [0.0, 1.0]], dtype=torch.float32)

    samples = sample_pdf(edges, weights, 17)

    assert samples.shape == (2, 17)
    assert samples.dtype == torch.float32
    assert samples.device == edges.device
    assert bool(torch.isfinite(samples).all())
    assert bool((samples >= edges[:, :1]).all())
    assert bool((samples <= edges[:, -1:]).all())


def test_large_finite_weights_do_not_overflow_the_pdf():
    edges = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32)
    weights = torch.tensor([[1e38, 1e38, 0.0]], dtype=torch.float32)

    samples = sample_pdf(edges, weights, 101)

    assert bool(torch.isfinite(samples).all())
    assert bool(((samples >= 0.0) & (samples <= 3.0)).all())


def test_randomized_sampling_is_reproducible_with_an_explicit_generator():
    edges = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    weights = torch.tensor([[0.2, 0.5, 0.3]])
    first = sample_pdf(edges, weights, 100, randomized=True, generator=torch.Generator().manual_seed(19))
    second = sample_pdf(edges, weights, 100, randomized=True, generator=torch.Generator().manual_seed(19))
    other = sample_pdf(edges, weights, 100, randomized=True, generator=torch.Generator().manual_seed(20))

    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert bool(((first >= 0.0) & (first <= 3.0)).all())


def test_merge_preserves_all_depths_and_sorts_each_ray():
    coarse = torch.tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
    fine = torch.tensor([[4.0, 2.0], [5.0, 3.0]])

    merged = merge_and_sort_depths(coarse, fine)

    assert merged.shape == (2, 5)
    torch.testing.assert_close(
        merged, torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]])
    )
    assert bool((merged[:, 1:] > merged[:, :-1]).all())


def test_merge_retains_duplicates_without_changing_shape():
    merged = merge_and_sort_depths(torch.tensor([[1.0, 2.0]]), torch.tensor([[2.0, 3.0]]))
    torch.testing.assert_close(merged, torch.tensor([[1.0, 2.0, 2.0, 3.0]]))


@pytest.mark.parametrize(
    "edges, weights, error, match",
    [
        (torch.tensor([[0.0, 1.0]]), torch.tensor([[1.0, 2.0]]), ValueError, "shape"),
        (torch.tensor([[0.0, 0.0, 2.0]]), torch.ones(1, 2), ValueError, "increase"),
        (torch.tensor([[0.0, 1.0]]), torch.tensor([[-1.0]]), ValueError, "nonnegative"),
        (torch.tensor([[0.0, float("nan")]]), torch.ones(1, 1), ValueError, "finite"),
        (torch.tensor([[0.0, 1.0]]), torch.tensor([[float("inf")]]), ValueError, "finite"),
    ],
)
def test_invalid_pdf_inputs_fail_clearly(edges, weights, error, match):
    with pytest.raises(error, match=match):
        sample_pdf(edges, weights, 4)


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_invalid_sample_count_is_rejected(count):
    with pytest.raises(ValueError, match="positive integer"):
        sample_pdf(torch.tensor([[0.0, 1.0]]), torch.ones(1, 1), count)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_deterministic_matches_cpu_and_seeded_random_mode_is_reproducible():
    edges = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32)
    weights = torch.tensor([[0.5, 4.0, 1.0]], dtype=torch.float32)
    cpu = sample_pdf(edges, weights, 128)
    gpu = sample_pdf(edges.cuda(), weights.cuda(), 128)

    assert gpu.device.type == "cuda"
    torch.testing.assert_close(gpu.cpu(), cpu, rtol=1e-5, atol=1e-6)
    generator1 = torch.Generator(device="cuda").manual_seed(83)
    generator2 = torch.Generator(device="cuda").manual_seed(83)
    first = sample_pdf(edges.cuda(), weights.cuda(), 128, randomized=True, generator=generator1)
    second = sample_pdf(edges.cuda(), weights.cuda(), 128, randomized=True, generator=generator2)
    assert torch.equal(first, second)
    merged = merge_and_sort_depths(edges[:, :3].cuda(), first)
    assert merged.device.type == "cuda"
    assert bool((merged[:, 1:] >= merged[:, :-1]).all())
