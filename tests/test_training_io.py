"""Persistence checks for Step 3 run records, metrics and exact resume state."""

from __future__ import annotations

import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from nerf_step1.model import VanillaNeRF
from nerf_step1.training_io import (
    METRIC_FIELDS,
    append_metrics,
    assert_run_config_compatible,
    create_training_run,
    load_checkpoint,
    reconcile_metrics,
    save_checkpoint,
)


@pytest.fixture
def config() -> dict[str, object]:
    return {
        "scene": "lego",
        "dataset_root": "data/nerf_synthetic",
        "position_encoding_L": 2,
        "direction_encoding_L": 1,
        "network_depth": 4,
        "network_width": 16,
        "skip_connection_layer": 1,
        "view_width": 8,
        "num_coarse_samples": 4,
        "num_fine_samples": 4,
        "batch_size": 8,
        "learning_rate": 0.0005,
        "training_iterations": 12,
        "random_seed": 71,
        "near": 2.0,
        "far": 6.0,
    }


def _run(tmp_path: Path, config: dict[str, object]) -> Path:
    return create_training_run(tmp_path, "test_training", config)


def _trainable_pair() -> tuple[torch.nn.Module, torch.nn.Module, torch.optim.Adam, object]:
    coarse = torch.nn.Linear(3, 3)
    fine = torch.nn.Linear(3, 3)
    optimizer = torch.optim.Adam(
        list(coarse.parameters()) + list(fine.parameters()), lr=5e-4
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.99 ** step)
    return coarse, fine, optimizer, scheduler


def test_run_record_freezes_config_and_records_environment(
    tmp_path: Path, config: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "nerf_step1.training_io.environment_snapshot",
        lambda: {"gpu_name": "Test GPU", "pytorch": "2.test", "pytorch_cuda_runtime": "12.test"},
    )
    monkeypatch.setattr("nerf_step1.training_io._git_commit", lambda: None)
    original = copy.deepcopy(config)
    first = _run(tmp_path, config)
    second = _run(tmp_path, config)
    assert first != second

    snapshot = yaml.safe_load((first / "config_snapshot.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((first / "run_manifest.json").read_text(encoding="utf-8"))
    environment = json.loads((first / "environment_snapshot.json").read_text(encoding="utf-8"))
    assert snapshot == original
    assert manifest["scene"] == "lego"
    assert manifest["position_encoding_L"] == 2
    assert manifest["seed"] == 71
    assert manifest["gpu"] == "Test GPU"
    assert manifest["pytorch_version"] == "2.test"
    assert manifest["cuda_runtime"] == "12.test"
    assert manifest["git_commit"] is None
    assert manifest["training_settings"] == original
    assert environment["gpu_name"] == "Test GPU"
    with (first / "metrics.csv").open(newline="", encoding="utf-8") as file:
        assert tuple(next(csv.reader(file))) == METRIC_FIELDS

    config["position_encoding_L"] = 4
    assert yaml.safe_load((first / "config_snapshot.yaml").read_text(encoding="utf-8")) == original
    with pytest.raises(ValueError, match="configuration differs"):
        assert_run_config_compatible(first, config)
    assert_run_config_compatible(first, original)

    with (first / "config_snapshot.yaml").open("a", encoding="utf-8") as file:
        file.write("# changed after run creation\n")
    with pytest.raises(ValueError, match="manifest digest"):
        assert_run_config_compatible(first, original)


def test_checkpoint_restores_models_optimizer_scheduler_and_all_cpu_rng(
    tmp_path: Path, config: dict[str, object]
) -> None:
    torch.manual_seed(71)
    np.random.seed(71)
    random.seed(71)
    run = _run(tmp_path, config)
    coarse, fine, optimizer, scheduler = _trainable_pair()
    x = torch.randn(4, 3)
    optimizer.zero_grad()
    (coarse(x).square().mean() + fine(x).square().mean()).backward()
    optimizer.step()
    scheduler.step()
    expected_coarse = copy.deepcopy(coarse.state_dict())
    expected_fine = copy.deepcopy(fine.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_scheduler = copy.deepcopy(scheduler.state_dict())
    checkpoint = run / "checkpoints" / "latest.pt"
    save_checkpoint(
        checkpoint, coarse_model=coarse, fine_model=fine,
        optimizer=optimizer, scheduler=scheduler, iteration=1, config=config,
    )
    expected_random = (torch.rand(3), np.random.rand(3), random.random())

    with torch.no_grad():
        for model in (coarse, fine):
            for parameter in model.parameters():
                parameter.add_(10)
    optimizer.param_groups[0]["lr"] = 0.7
    torch.rand(10)
    np.random.rand(10)
    random.random()
    restored_iteration = load_checkpoint(
        checkpoint, coarse_model=coarse, fine_model=fine,
        optimizer=optimizer, scheduler=scheduler, expected_config=config,
    )

    assert restored_iteration == 1
    for name, value in expected_coarse.items():
        torch.testing.assert_close(coarse.state_dict()[name], value, rtol=0, atol=0)
    for name, value in expected_fine.items():
        torch.testing.assert_close(fine.state_dict()[name], value, rtol=0, atol=0)
    assert optimizer.param_groups[0]["lr"] == expected_optimizer["param_groups"][0]["lr"]
    assert scheduler.state_dict() == expected_scheduler
    current_optimizer = optimizer.state_dict()
    for parameter_index, state in expected_optimizer["state"].items():
        for name, value in state.items():
            actual = current_optimizer["state"][parameter_index][name]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual, value, rtol=0, atol=0)
            else:
                assert actual == value
    torch.testing.assert_close(torch.rand(3), expected_random[0], rtol=0, atol=0)
    np.testing.assert_array_equal(np.random.rand(3), expected_random[1])
    assert random.random() == expected_random[2]


def test_checkpoint_rejects_incompatible_config_without_mutating_model(
    tmp_path: Path, config: dict[str, object]
) -> None:
    run = _run(tmp_path, config)
    coarse, fine, optimizer, scheduler = _trainable_pair()
    checkpoint = run / "checkpoints" / "latest.pt"
    save_checkpoint(
        checkpoint, coarse_model=coarse, fine_model=fine,
        optimizer=optimizer, scheduler=scheduler, iteration=0, config=config,
    )
    before = copy.deepcopy(coarse.state_dict())
    changed = {**config, "position_encoding_L": 4}
    with pytest.raises(ValueError, match="configuration differs"):
        load_checkpoint(
            checkpoint, coarse_model=coarse, fine_model=fine,
            optimizer=optimizer, scheduler=scheduler, expected_config=changed,
        )
    for key, value in before.items():
        torch.testing.assert_close(coarse.state_dict()[key], value, rtol=0, atol=0)
    with pytest.raises(ValueError, match="scheduler presence"):
        load_checkpoint(
            checkpoint, coarse_model=coarse, fine_model=fine,
            optimizer=optimizer, expected_config=config,
        )


def test_checkpoint_rejects_model_architecture_drift(
    tmp_path: Path, config: dict[str, object]
) -> None:
    run = _run(tmp_path, config)
    matching = VanillaNeRF.from_config(config)
    different = VanillaNeRF.from_config({**config, "position_encoding_L": 3})
    optimizer = torch.optim.Adam(list(matching.parameters()) + list(different.parameters()))
    with pytest.raises(ValueError, match="fine model architecture"):
        save_checkpoint(
            run / "checkpoint.pt", coarse_model=matching, fine_model=different,
            optimizer=optimizer, iteration=0, config=config,
        )


def test_metrics_append_rows_without_rewriting_header(
    tmp_path: Path, config: dict[str, object]
) -> None:
    run = _run(tmp_path, config)
    first = {
        "iteration": 1, "total_loss": 0.8, "coarse_loss": 0.5,
        "fine_loss": 0.3, "psnr": 5.2, "learning_rate": 0.0005,
        "elapsed_time": 1.25, "rays_per_second": 400.0,
        "gpu_memory_allocated_bytes": 1024, "gpu_memory_reserved_bytes": 2048,
    }
    append_metrics(run, first)
    append_metrics(run, {**first, "iteration": 2, "total_loss": 0.4})
    with (run / "metrics.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert rows[0]["iteration"] == "1"
    assert rows[0]["gpu_memory_allocated_bytes"] == "1024"
    assert rows[1]["total_loss"] == "0.4"
    assert rows[1]["psnr"] == "5.2"

    with pytest.raises(ValueError, match="finite number"):
        append_metrics(run, {**first, "iteration": 3, "total_loss": float("nan")})
    with pytest.raises(ValueError, match="missing"):
        append_metrics(run, {"iteration": 3})
    with (run / "metrics.csv").open(newline="", encoding="utf-8") as file:
        assert len(list(csv.reader(file))) == 3  # header and two valid rows


def test_resume_archives_metrics_beyond_checkpoint_and_rejects_nonmonotonic_rows(
    tmp_path: Path, config: dict[str, object]
) -> None:
    run = _run(tmp_path, config)
    base_row = {
        "total_loss": 0.8, "coarse_loss": 0.5, "fine_loss": 0.3,
        "psnr": 5.2, "learning_rate": 0.0005,
        "elapsed_time": 1.25, "rays_per_second": 400.0,
    }
    for iteration in (1, 2, 3):
        append_metrics(run, {"iteration": iteration, **base_row})
    original = (run / "metrics.csv").read_bytes()
    archive = reconcile_metrics(run, 1)
    assert archive is not None
    assert archive.read_bytes() == original
    with (run / "metrics.csv").open(newline="", encoding="utf-8") as file:
        assert [row["iteration"] for row in csv.DictReader(file)] == ["1"]
    append_metrics(run, {"iteration": 2, **base_row})
    assert reconcile_metrics(run, 2) is None

    # Refuse to erase or silently reorder an already ambiguous history.
    append_metrics(run, {"iteration": 2, **base_row})
    ambiguous = (run / "metrics.csv").read_bytes()
    with pytest.raises(ValueError, match="strictly increasing"):
        reconcile_metrics(run, 1)
    assert (run / "metrics.csv").read_bytes() == ambiguous
