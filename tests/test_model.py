"""Architecture and numerical contract checks for the untrained vanilla NeRF."""

from __future__ import annotations

from io import BytesIO

import pytest
import torch
import yaml

from nerf_step1.dataset import CameraIntrinsics
from nerf_step1.model import VanillaNeRF
from nerf_step1.rays import generate_rays


SMALL_ARCHITECTURE_CONFIG = {
    "position_encoding_L": 2,
    "direction_encoding_L": 1,
    "network_depth": 4,
    "network_width": 16,
    "skip_connection_layer": 1,
    "view_width": 8,
}


def _small_model(**overrides: int) -> VanillaNeRF:
    config = dict(SMALL_ARCHITECTURE_CONFIG)
    config.update(overrides)
    return VanillaNeRF(**config)


@pytest.mark.parametrize("batch_size", [0, 1, 7])
def test_forward_shapes_ranges_and_finiteness(batch_size: int) -> None:
    model = _small_model()
    positions = torch.randn(batch_size, 3)
    directions = torch.randn(batch_size, 3)

    rgb, sigma = model(positions, directions)

    assert rgb.shape == (batch_size, 3)
    assert sigma.shape == (batch_size, 1)
    assert rgb.dtype == sigma.dtype == torch.float32
    assert bool(torch.isfinite(rgb).all())
    assert bool(torch.isfinite(sigma).all())
    assert bool(((rgb >= 0) & (rgb <= 1)).all())
    assert bool((sigma >= 0).all())


@pytest.mark.parametrize("bandwidth", [0, 2, 4, 10, 15])
def test_position_bandwidth_changes_network_input(bandwidth: int) -> None:
    model = _small_model(position_encoding_L=bandwidth)
    position_dim = 3 + 6 * bandwidth

    assert model.position_encoder.output_dim == position_dim
    assert model.position_layers[0].in_features == position_dim
    assert model.position_layers[2].in_features == 16 + position_dim
    assert model.view_layer.in_features == 16 + 9  # direction L=1 stays fixed
    rgb, sigma = model(torch.randn(2, 3), torch.randn(2, 3))
    assert rgb.shape == (2, 3)
    assert sigma.shape == (2, 1)


def test_default_architecture_matches_released_view_dependent_nerf() -> None:
    model = VanillaNeRF()

    assert model.position_encoder.output_dim == 63
    assert model.direction_encoder.output_dim == 27
    assert len(model.position_layers) == 8
    assert [(layer.in_features, layer.out_features) for layer in model.position_layers] == [
        (63, 256),
        (256, 256),
        (256, 256),
        (256, 256),
        (256, 256),
        (319, 256),  # concatenation follows zero-based layer index 4
        (256, 256),
        (256, 256),
    ]
    assert (model.density_layer.in_features, model.density_layer.out_features) == (256, 1)
    assert (model.feature_layer.in_features, model.feature_layer.out_features) == (256, 256)
    assert (model.view_layer.in_features, model.view_layer.out_features) == (283, 128)
    assert (model.rgb_layer.in_features, model.rgb_layer.out_features) == (128, 3)
    assert sum(parameter.numel() for parameter in model.parameters()) == 595_844


def test_bandwidth_parameter_count_is_disclosed_by_architecture() -> None:
    no_frequency = VanillaNeRF(position_encoding_L=0)
    baseline = VanillaNeRF(position_encoding_L=10)
    assert sum(parameter.numel() for parameter in no_frequency.parameters()) == 565_124
    assert sum(parameter.numel() for parameter in baseline.parameters()) == 595_844


def test_configurable_depth_width_skip_and_view_width() -> None:
    model = _small_model(network_depth=6, network_width=20, skip_connection_layer=3, view_width=11)

    assert len(model.position_layers) == 6
    assert model.position_layers[4].in_features == 20 + 15
    assert model.position_layers[4].out_features == 20
    assert model.view_layer.in_features == 20 + 9
    assert model.view_layer.out_features == 11
    assert model.rgb_layer.in_features == 11
    assert model.architecture_config == {
        "position_encoding_L": 2,
        "direction_encoding_L": 1,
        "network_depth": 6,
        "network_width": 20,
        "skip_connection_layer": 3,
        "view_width": 11,
    }


def test_from_config_accepts_full_baseline_config() -> None:
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    model = VanillaNeRF.from_config(config)
    assert model.position_encoding_L == config["position_encoding_L"]
    assert model.direction_encoding_L == config["direction_encoding_L"]
    assert model.network_depth == config["network_depth"]
    assert model.network_width == config["network_width"]
    assert model.skip_connection_layer == config["skip_connection_layer"]
    assert model.view_width == config["view_width"]


@pytest.mark.parametrize(
    "bad_config",
    [
        {"position_encoding_L": -1},
        {"direction_encoding_L": 2.5},
        {"network_depth": 0},
        {"network_width": True},
        {"view_width": 0},
        {"skip_connection_layer": -1},
        {"network_depth": 4, "skip_connection_layer": 3},
    ],
)
def test_invalid_architecture_config_rejected(bad_config: dict[str, object]) -> None:
    config = {**SMALL_ARCHITECTURE_CONFIG, **bad_config}
    with pytest.raises(ValueError):
        VanillaNeRF.from_config(config)


@pytest.mark.parametrize("missing_key", list(SMALL_ARCHITECTURE_CONFIG))
def test_from_config_rejects_each_missing_architecture_key(missing_key: str) -> None:
    config = dict(SMALL_ARCHITECTURE_CONFIG)
    del config[missing_key]
    with pytest.raises(ValueError, match=f"missing required architecture keys: {missing_key}"):
        VanillaNeRF.from_config(config)


def test_from_config_requires_mapping() -> None:
    with pytest.raises(TypeError, match="mapping"):
        VanillaNeRF.from_config(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "positions,directions,error_type,match",
    [
        (torch.randn(3), torch.randn(1, 3), ValueError, "positions"),
        (torch.randn(1, 2), torch.randn(1, 3), ValueError, "positions"),
        (torch.randn(1, 3), torch.randn(2, 3), ValueError, "batch size"),
        (torch.randn(1, 3), torch.randn(3), ValueError, "directions"),
        (torch.ones(1, 3, dtype=torch.int64), torch.randn(1, 3), TypeError, "positions"),
        (torch.randn(1, 3), torch.randn(1, 3, dtype=torch.float64), ValueError, "dtype"),
    ],
)
def test_forward_rejects_invalid_inputs(
    positions: torch.Tensor,
    directions: torch.Tensor,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        _small_model()(positions, directions)


def test_forward_requires_model_dtype_to_match_inputs() -> None:
    model = _small_model().double()
    with pytest.raises(ValueError, match="model parameters"):
        model(torch.randn(1, 3), torch.randn(1, 3))
    rgb, sigma = model(torch.randn(1, 3, dtype=torch.float64), torch.randn(1, 3, dtype=torch.float64))
    assert rgb.dtype == sigma.dtype == torch.float64


def test_density_does_not_depend_on_viewing_direction() -> None:
    model = _small_model()
    positions = torch.randn(5, 3)
    rgb_a, sigma_a = model(positions, torch.zeros(5, 3))
    rgb_b, sigma_b = model(positions, torch.ones(5, 3))

    torch.testing.assert_close(sigma_a, sigma_b, rtol=0, atol=0)
    assert not torch.equal(rgb_a, rgb_b)


def test_backward_reaches_every_parameter_and_raw_input() -> None:
    model = _small_model()
    # Keep every ReLU branch active so this tests connectivity, not a random
    # draw where an untrained density head happens to output only negatives.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.fill_(0.01)
                module.bias.fill_(0.5)
    positions = torch.rand(4, 3, requires_grad=True) / 10
    directions = torch.rand(4, 3, requires_grad=True) / 10
    positions.retain_grad()
    directions.retain_grad()

    rgb, sigma = model(positions, directions)
    (rgb.sum() + sigma.sum()).backward()

    assert positions.grad is not None and bool(torch.isfinite(positions.grad).all())
    assert directions.grad is not None and bool(torch.isfinite(directions.grad).all())
    assert bool((positions.grad.abs().sum() > 0).item())
    assert bool((directions.grad.abs().sum() > 0).item())
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
        assert bool((parameter.grad.abs().sum() > 0).item()), name


def test_fixed_seed_initialization_is_reproducible() -> None:
    torch.manual_seed(9871)
    first = _small_model()
    torch.manual_seed(9871)
    second = _small_model()

    for first_parameter, second_parameter in zip(first.parameters(), second.parameters()):
        assert torch.equal(first_parameter, second_parameter)
    positions = torch.randn(2, 3)
    directions = torch.randn(2, 3)
    for first_output, second_output in zip(first(positions, directions), second(positions, directions)):
        assert torch.equal(first_output, second_output)


def test_state_dict_round_trip_preserves_output() -> None:
    model = _small_model()
    positions, directions = torch.randn(3, 3), torch.randn(3, 3)
    expected = model(positions, directions)
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)

    restored = VanillaNeRF.from_config(model.architecture_config)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    for actual, reference in zip(restored(positions, directions), expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_step1_rays_to_raw_position_and_direction_to_mlp() -> None:
    intrinsics = CameraIntrinsics(width=4, height=4, fx=2.0, fy=2.0, cx=2.0, cy=2.0)
    origins, raw_directions = generate_rays(
        torch.eye(4), intrinsics, torch.tensor([[2, 2], [3, 1]])
    )
    positions = origins + 2.0 * raw_directions
    view_directions = torch.nn.functional.normalize(raw_directions, dim=-1)

    rgb, sigma = _small_model()(positions, view_directions)

    assert rgb.shape == (2, 3)
    assert sigma.shape == (2, 1)
    assert bool(torch.isfinite(rgb).all())
    assert bool(torch.isfinite(sigma).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_forward_and_backward() -> None:
    model = _small_model().cuda()
    positions = torch.randn(4, 3, device="cuda", requires_grad=True)
    directions = torch.randn(4, 3, device="cuda", requires_grad=True)
    rgb, sigma = model(positions, directions)
    (rgb.sum() + sigma.sum()).backward()

    assert rgb.device.type == sigma.device.type == "cuda"
    assert rgb.shape == (4, 3) and sigma.shape == (4, 1)
    assert positions.grad is not None and bool(torch.isfinite(positions.grad).all())
    assert directions.grad is not None and bool(torch.isfinite(directions.grad).all())
    assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_device_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="device and dtype"):
        _small_model().cuda()(torch.randn(1, 3, device="cuda"), torch.randn(1, 3))
