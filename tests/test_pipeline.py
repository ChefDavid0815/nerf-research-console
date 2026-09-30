"""End-to-end gradient and small-data optimization checks for Step 3."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nerf_step1.pipeline import CoarseFineNeRF, make_scheduler, nerf_loss, train_step  # noqa: E402
from nerf_step1.training_io import create_training_run, load_checkpoint, save_checkpoint  # noqa: E402


def tiny_config() -> dict:
    path = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config.update(
        position_encoding_L=2, direction_encoding_L=1,
        network_depth=3, network_width=24, skip_connection_layer=1, view_width=12,
        num_coarse_samples=6, num_fine_samples=6, batch_size=4,
        ray_chunk_size=2, learning_rate=0.005,
    )
    return config


def tiny_rays(device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    origins = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.1, 0.1, 0.0]],
        device=device,
    )
    directions = torch.tensor(
        [[0.0, 0.0, -1.0], [0.05, 0.0, -1.0],
         [0.0, 0.05, -1.0], [0.05, 0.05, -1.0]], device=device,
    )
    return origins, directions


def test_coarse_fine_forward_shapes_sorted_depths_and_gradients() -> None:
    model = CoarseFineNeRF(tiny_config())
    origins, directions = tiny_rays()
    result = model(origins, directions, randomized=False)
    assert result.coarse.rgb_map.shape == (4, 3)
    assert result.fine.rgb_map.shape == (4, 3)
    assert result.coarse_depths.shape == (4, 6)
    assert result.fine_depths.shape == (4, 6)
    assert result.combined_depths.shape == (4, 12)
    assert torch.all(result.combined_depths[:, 1:] >= result.combined_depths[:, :-1])
    assert torch.all((result.combined_depths >= model.config["near"]) &
                     (result.combined_depths <= model.config["far"]))
    loss = nerf_loss(result, torch.tensor([[0.8, 0.2, 0.1]] * 4))
    torch.testing.assert_close(loss["total_loss"], loss["coarse_loss"] + loss["fine_loss"])
    loss["total_loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.coarse_model.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.fine_model.parameters())


def test_one_training_step_updates_both_networks_and_scheduler() -> None:
    torch.manual_seed(7)
    config = tiny_config()
    model = CoarseFineNeRF(config)
    for network in (model.coarse_model, model.fine_model):
        network.density_layer.bias.data.fill_(0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = make_scheduler(optimizer, config)
    origins, directions = tiny_rays()
    targets = torch.tensor([[0.8, 0.2, 0.1]] * 4)
    before_coarse = {name: p.detach().clone() for name, p in model.coarse_model.named_parameters()}
    before_fine = {name: p.detach().clone() for name, p in model.fine_model.named_parameters()}
    metrics = train_step(model, optimizer, origins, directions, targets, scheduler=scheduler)
    assert metrics["total_loss"] == pytest.approx(metrics["coarse_loss"] + metrics["fine_loss"])
    assert metrics["PSNR"] > 0
    assert any(not torch.equal(before_coarse[name], p) for name, p in model.coarse_model.named_parameters())
    assert any(not torch.equal(before_fine[name], p) for name, p in model.fine_model.named_parameters())
    assert optimizer.param_groups[0]["lr"] < config["learning_rate"]


def test_tiny_end_to_end_overfit() -> None:
    """A few rays with nonidentical colors must actually become easier to fit."""
    torch.manual_seed(14)
    config = tiny_config()
    model = CoarseFineNeRF(config)
    for network in (model.coarse_model, model.fine_model):
        network.density_layer.bias.data.fill_(0.35)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    origins, directions = tiny_rays()
    targets = torch.tensor(
        [[0.78, 0.22, 0.12], [0.72, 0.28, 0.18],
         [0.18, 0.72, 0.24], [0.24, 0.78, 0.30]]
    )
    with torch.no_grad():
        initial = float(nerf_loss(model(origins, directions), targets)["total_loss"])
    for _ in range(90):
        train_step(model, optimizer, origins, directions, targets)
    with torch.no_grad():
        final = float(nerf_loss(model(origins, directions), targets)["total_loss"])
    assert final < 0.25 * initial, (initial, final)


def test_checkpoint_resume_repeats_next_stochastic_training_step(tmp_path: Path) -> None:
    """The ray sampler and PDF must continue from the saved RNG stream."""
    torch.manual_seed(31)
    config = tiny_config()
    model = CoarseFineNeRF(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = make_scheduler(optimizer, config)
    origins, directions = tiny_rays()
    targets = torch.tensor([[0.8, 0.2, 0.1]] * 4)
    train_step(model, optimizer, origins, directions, targets, scheduler=scheduler)
    run = create_training_run(tmp_path, "resume_test", config)
    checkpoint = run / "checkpoints" / "iter_000001.pt"
    save_checkpoint(
        checkpoint, coarse_model=model.coarse_model, fine_model=model.fine_model,
        optimizer=optimizer, scheduler=scheduler, iteration=1, config=config,
    )
    expected_metrics = train_step(model, optimizer, origins, directions, targets, scheduler=scheduler)
    expected_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    resumed = CoarseFineNeRF(config)
    resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=config["learning_rate"])
    resumed_scheduler = make_scheduler(resumed_optimizer, config)
    assert load_checkpoint(
        checkpoint, coarse_model=resumed.coarse_model, fine_model=resumed.fine_model,
        optimizer=resumed_optimizer, scheduler=resumed_scheduler, expected_config=config,
    ) == 1
    actual_metrics = train_step(
        resumed, resumed_optimizer, origins, directions, targets, scheduler=resumed_scheduler
    )
    assert actual_metrics == pytest.approx(expected_metrics, rel=0, abs=0)
    for key, value in expected_state.items():
        torch.testing.assert_close(resumed.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_pipeline_matches_cpu_deterministic_forward() -> None:
    torch.manual_seed(3)
    config = tiny_config()
    cpu_model = CoarseFineNeRF(config)
    cuda_model = CoarseFineNeRF(config).cuda()
    cuda_model.load_state_dict(cpu_model.state_dict())
    origins, directions = tiny_rays()
    with torch.no_grad():
        cpu = cpu_model(origins, directions)
        cuda = cuda_model(origins.cuda(), directions.cuda())
    torch.testing.assert_close(cpu.fine.rgb_map, cuda.fine.rgb_map.cpu(), atol=2e-5, rtol=2e-5)
