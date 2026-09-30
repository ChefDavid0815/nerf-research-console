"""Mathematical and device checks for explicit NeRF Fourier features."""

import math

import pytest
import torch

from nerf_step1.positional_encoding import SinusoidalPositionalEncoding


@pytest.mark.parametrize(
    ("bandwidth", "expected_dim"),
    [(0, 3), (1, 9), (2, 15), (4, 27), (10, 63), (15, 93)],
)
def test_bandwidth_sets_output_dimension_and_batch_shape(bandwidth, expected_dim):
    encoder = SinusoidalPositionalEncoding(3, bandwidth)
    points = torch.zeros((2, 4, 3), dtype=torch.float32)

    result = encoder(points)

    assert encoder.output_dim == expected_dim
    assert result.shape == (2, 4, expected_dim)
    assert result.dtype == points.dtype
    assert result.device == points.device


def test_zero_bandwidth_is_exact_identity_and_preserves_gradients():
    points = torch.tensor([[0.0, -1.0, 1.0]], requires_grad=True)
    result = SinusoidalPositionalEncoding(3, 0)(points)

    assert torch.equal(result, points)
    result.sum().backward()
    torch.testing.assert_close(points.grad, torch.ones_like(points))


def test_scalar_half_has_hand_calculated_frequency_values():
    # At p = 1/2, the first three phases are pi/2, pi, and 2*pi.
    encoder = SinusoidalPositionalEncoding(1, 3)
    result = encoder(torch.tensor([0.5], dtype=torch.float64))

    expected = torch.tensor([0.5, 1.0, 0.0, 0.0, -1.0, 0.0, 1.0], dtype=torch.float64)
    torch.testing.assert_close(result, expected, rtol=0, atol=1e-14)


def test_vector_component_order_and_frequency_formula():
    encoder = SinusoidalPositionalEncoding(3, 2)
    point = torch.tensor([0.5, 0.0, -0.5], dtype=torch.float64)

    # Raw xyz; sin(pi * xyz), cos(pi * xyz); sin(2*pi * xyz), cos(...).
    expected = torch.tensor(
        [0.5, 0.0, -0.5, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, -1.0, 1.0, -1.0],
        dtype=torch.float64,
    )
    torch.testing.assert_close(encoder(point), expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_determinism_and_finite_boundary_values(dtype):
    encoder = SinusoidalPositionalEncoding(3, 15)
    boundaries = torch.tensor(
        [[-1.0, 0.0, 1.0], [0.5, -0.5, 0.25]], dtype=dtype
    )
    generator = torch.Generator().manual_seed(2026)
    random_points = torch.rand((128, 3), generator=generator, dtype=dtype) * 2 - 1
    points = torch.cat((boundaries, random_points))

    first = encoder(points)
    second = encoder(points)

    assert first.dtype == dtype
    assert first.shape == (130, 93)
    assert torch.equal(first, second)
    assert bool(torch.isfinite(first).all())
    torch.testing.assert_close(first[:, :3], points, rtol=0, atol=0)
    assert bool((first[:, 3:].abs() <= 1).all())


def test_unbatched_and_empty_batch_shapes():
    encoder = SinusoidalPositionalEncoding(3, 4)
    assert encoder(torch.zeros(3)).shape == (27,)
    assert encoder(torch.empty((0, 3))).shape == (0, 27)


def test_trigonometric_gradient_matches_analytic_derivative():
    points = torch.tensor([[0.1, 0.2, -0.3]], dtype=torch.float64, requires_grad=True)
    encoded = SinusoidalPositionalEncoding(3, 2)(points)
    # Index 3 is sin(pi*x), since indices 0:3 contain raw xyz.
    encoded[:, 3].sum().backward()

    expected = torch.tensor(
        [[math.pi * math.cos(math.pi * 0.1), 0.0, 0.0]], dtype=torch.float64
    )
    torch.testing.assert_close(points.grad, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize(
    ("input_dim", "num_frequencies", "include_input", "exception"),
    [
        (0, 2, True, ValueError),
        (3, -1, True, ValueError),
        (3, 0, False, ValueError),
        (3, 2, False, ValueError),
        (3, 1.5, True, ValueError),
        (3, 2, "yes", TypeError),
    ],
)
def test_invalid_configuration_is_rejected(
    input_dim, num_frequencies, include_input, exception
):
    with pytest.raises(exception):
        SinusoidalPositionalEncoding(input_dim, num_frequencies, include_input)


def test_invalid_input_shape_and_dtype_are_rejected():
    encoder = SinusoidalPositionalEncoding(3, 4)
    with pytest.raises(ValueError, match="final dimension"):
        encoder(torch.zeros((2, 2)))
    with pytest.raises(ValueError, match="final dimension"):
        encoder(torch.tensor(1.0))
    with pytest.raises(TypeError, match="floating-point"):
        encoder(torch.zeros((2, 3), dtype=torch.int64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_matches_cpu_and_backpropagates():
    encoder = SinusoidalPositionalEncoding(3, 4)
    cpu_points = torch.tensor(
        [[0.125, -0.25, 0.5], [-0.75, 0.0, 1.0]], dtype=torch.float32
    )
    cuda_points = cpu_points.to("cuda").requires_grad_()

    cpu_result = encoder(cpu_points)
    cuda_result = encoder(cuda_points)

    assert cuda_result.device.type == "cuda"
    assert cuda_result.dtype == cpu_result.dtype
    torch.testing.assert_close(cuda_result.cpu(), cpu_result, rtol=1e-5, atol=1e-5)
    cuda_result.sum().backward()
    assert cuda_points.grad is not None
    assert bool(torch.isfinite(cuda_points.grad).all())
