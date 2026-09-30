"""Analytic alpha compositing, stability, gradient and device checks."""

import math

import pytest
import torch

from nerf_step1.volume_rendering import volume_render


def test_zero_density_returns_white_or_black_background_exactly():
    rgb = torch.tensor([[[0.2, 0.3, 0.4], [0.8, 0.7, 0.6]]])
    sigma = torch.zeros((1, 2))
    depths = torch.tensor([[2.0, 3.0]])
    distances = torch.tensor([[1.0, 1e10]])
    white = volume_render(rgb, sigma, depths, distances)
    black = volume_render(rgb, sigma, depths, distances, white_background=False)

    torch.testing.assert_close(white.rgb_map, torch.ones((1, 3)))
    torch.testing.assert_close(black.rgb_map, torch.zeros((1, 3)))
    torch.testing.assert_close(white.alpha, torch.zeros((1, 2)))
    torch.testing.assert_close(white.weights, torch.zeros((1, 2)))
    torch.testing.assert_close(white.transmittance, torch.ones((1, 2)))
    torch.testing.assert_close(white.depth_map, torch.zeros(1))
    torch.testing.assert_close(white.accumulated_opacity, torch.zeros(1))


def test_first_extreme_density_dominates_without_nan_or_inf():
    rgb = torch.tensor([[[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]]], requires_grad=True)
    sigma = torch.tensor([[1e30, 1.0]], requires_grad=True)
    result = volume_render(
        rgb, sigma,
        torch.tensor([[2.0, 3.0]]), torch.tensor([[1e30, 1e10]]),
    )
    assert result.weights[0, 0] > 1 - 1e-6
    assert result.weights[0, 1] < 1e-30
    torch.testing.assert_close(result.rgb_map[0], rgb[0, 0], atol=1e-6, rtol=0)
    for tensor in result:
        assert bool(torch.isfinite(tensor).all())
    result.rgb_map.sum().backward()
    assert sigma.grad is not None and rgb.grad is not None
    assert bool(torch.isfinite(sigma.grad).all())
    assert bool(torch.isfinite(rgb.grad).all())


def test_single_opaque_red_sample_renders_red():
    result = volume_render(
        torch.tensor([[[1.0, 0.0, 0.0]]]),
        torch.tensor([[100.0]]),
        torch.tensor([[2.0]]),
        torch.tensor([[1.0]]),
    )
    torch.testing.assert_close(result.rgb_map, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(result.accumulated_opacity, torch.ones(1), atol=1e-6, rtol=0)


def test_two_colour_ray_matches_hand_calculated_alpha_compositing():
    # sigma*delta = ln(2) for both samples: alpha=(1/2,1/2),
    # T=(1,1/2), w=(1/2,1/4), residual white=(1/4,1/4,1/4).
    result = volume_render(
        torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float64),
        torch.tensor([[math.log(2.0), math.log(2.0)]], dtype=torch.float64),
        torch.tensor([[1.0, 2.0]], dtype=torch.float64),
        torch.ones((1, 2), dtype=torch.float64),
    )
    torch.testing.assert_close(result.alpha, torch.tensor([[0.5, 0.5]], dtype=torch.float64), rtol=0, atol=1e-15)
    torch.testing.assert_close(result.transmittance, torch.tensor([[1.0, 0.5]], dtype=torch.float64), rtol=0, atol=1e-15)
    torch.testing.assert_close(result.weights, torch.tensor([[0.5, 0.25]], dtype=torch.float64), rtol=0, atol=1e-15)
    torch.testing.assert_close(result.rgb_map, torch.tensor([[0.75, 0.25, 0.5]], dtype=torch.float64), rtol=0, atol=1e-15)
    torch.testing.assert_close(result.depth_map, torch.tensor([1.0], dtype=torch.float64), rtol=0, atol=1e-15)
    torch.testing.assert_close(result.accumulated_opacity, torch.tensor([0.75], dtype=torch.float64), rtol=0, atol=1e-15)


def test_shapes_nonnegative_weights_and_opacity_bound_for_many_rays():
    generator = torch.Generator().manual_seed(88)
    rgb = torch.rand((11, 32, 3), generator=generator)
    sigma = torch.rand((11, 32, 1), generator=generator) * 2
    depths = torch.linspace(2.0, 6.0, 32).expand(11, 32)
    distances = torch.cat((depths[:, 1:] - depths[:, :-1], torch.full((11, 1), 1e10)), dim=-1)
    result = volume_render(rgb, sigma, depths, distances)
    assert result.rgb_map.shape == (11, 3)
    assert result.weights.shape == result.alpha.shape == result.transmittance.shape == (11, 32)
    assert result.depth_map.shape == result.accumulated_opacity.shape == (11,)
    assert bool((result.weights >= 0).all())
    assert bool((result.weights.sum(dim=-1) <= 1 + 1e-6).all())
    assert bool(torch.isfinite(result.rgb_map).all())


def test_rgb_and_density_gradients_match_single_sample_analytic_derivatives():
    rgb = torch.tensor([[[0.2, 0.3, 0.4]]], dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor([[0.7]], dtype=torch.float64, requires_grad=True)
    result = volume_render(
        rgb, sigma, torch.tensor([[2.0]], dtype=torch.float64),
        torch.tensor([[1.0]], dtype=torch.float64),
    )
    result.rgb_map[0, 0].backward()
    expected_alpha = 1.0 - math.exp(-0.7)
    assert rgb.grad is not None and sigma.grad is not None
    torch.testing.assert_close(rgb.grad[0, 0], torch.tensor([expected_alpha, 0.0, 0.0], dtype=torch.float64), rtol=0, atol=1e-15)
    torch.testing.assert_close(sigma.grad, torch.tensor([[(0.2 - 1.0) * math.exp(-0.7)]], dtype=torch.float64), rtol=0, atol=1e-15)


def test_invalid_negative_density_or_distance_is_rejected():
    rgb = torch.zeros((1, 1, 3))
    depth = torch.ones((1, 1))
    with pytest.raises(ValueError, match="sigma"):
        volume_render(rgb, torch.tensor([[-1.0]]), depth, depth)
    with pytest.raises(ValueError, match="sample_distances"):
        volume_render(rgb, depth, depth, torch.tensor([[-1.0]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_matches_cpu_and_backpropagates():
    rgb = torch.tensor([[[0.2, 0.4, 0.6], [0.8, 0.3, 0.1]]], dtype=torch.float32)
    sigma = torch.tensor([[0.5, 1.0]], dtype=torch.float32)
    depths = torch.tensor([[2.0, 3.0]], dtype=torch.float32)
    distances = torch.tensor([[1.0, 1e10]], dtype=torch.float32)
    cpu = volume_render(rgb, sigma, depths, distances)
    rgb_gpu = rgb.cuda().requires_grad_()
    sigma_gpu = sigma.cuda().requires_grad_()
    gpu = volume_render(rgb_gpu, sigma_gpu, depths.cuda(), distances.cuda())
    for cpu_tensor, gpu_tensor in zip(cpu, gpu):
        torch.testing.assert_close(gpu_tensor.cpu(), cpu_tensor, rtol=1e-5, atol=1e-6)
    gpu.rgb_map.sum().backward()
    assert rgb_gpu.grad is not None and sigma_gpu.grad is not None
    assert bool(torch.isfinite(rgb_gpu.grad).all())
    assert bool(torch.isfinite(sigma_gpu.grad).all())
