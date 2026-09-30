"""Step 4 run control must extend a frozen Step 3 checkpoint explicitly."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from scripts import train_nerf as training
from scripts.train_nerf import (
    _write_extension_record,
    output_due,
    parse_stage_iterations,
)
from test_dataset import IDENTITY_POSE, TRANSLATED_POSE, _write_split
from test_pipeline import tiny_config


def test_stage_milestones_replace_smoke_output_cadence() -> None:
    milestones = parse_stage_iterations("1000,2000,5000,10000,20000")
    assert milestones == (1000, 2000, 5000, 10000, 20000)
    assert output_due(1000, 2000, 250, milestones)
    assert output_due(2000, 2000, 250, milestones)
    assert not output_due(750, 2000, 250, milestones)
    assert not output_due(1250, 2000, 250, milestones)
    assert output_due(750, 2000, 250, None)  # Original Step 3 cadence.
    assert output_due(1200, 1200, 250, ())  # Boundary saved without milestones.
    assert not output_due(750, 1200, 250, ())


@pytest.mark.parametrize("bad", ["", "1000,", "0,1000", "1000,1000", "2000,1000", "x"])
def test_stage_milestones_reject_ambiguous_input(bad: str) -> None:
    with pytest.raises(ValueError, match="stage-checkpoints"):
        parse_stage_iterations(bad)


def test_extension_record_keeps_original_budget_and_source(tmp_path: Path) -> None:
    milestones = (1000, 2000, 5000)
    _write_extension_record(
        tmp_path, original_budget=500, source_iteration=500,
        target_iteration=1000, stage_iterations=milestones,
    )
    record_path = tmp_path / "baseline_extension.json"
    first = json.loads(record_path.read_text(encoding="utf-8"))
    assert first["source_checkpoint_iteration"] == 500
    assert first["original_training_iterations"] == 500
    assert first["latest_requested_target_iteration"] == 1000
    assert first["training_config_unchanged"] is True

    _write_extension_record(
        tmp_path, original_budget=500, source_iteration=1000,
        target_iteration=2000, stage_iterations=milestones,
    )
    second = json.loads(record_path.read_text(encoding="utf-8"))
    assert second["source_checkpoint_iteration"] == 500
    assert second["latest_requested_target_iteration"] == 2000

    _write_extension_record(
        tmp_path, original_budget=500, source_iteration=2000,
        target_iteration=5000, stage_iterations=(*milestones, 10000),
    )
    extended = json.loads(record_path.read_text(encoding="utf-8"))
    assert extended["stage_iterations"] == [1000, 2000, 5000, 10000]

    with pytest.raises(ValueError, match="schedule differs"):
        _write_extension_record(
            tmp_path, original_budget=500, source_iteration=2000,
            target_iteration=5000, stage_iterations=(1000, 5000),
        )
    assert json.loads(record_path.read_text(encoding="utf-8")) == extended


def test_one_step_continuation_preserves_original_checkpoint_and_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tiny_blender_root = tmp_path / "data"
    scene_root = tiny_blender_root / "lego"
    scene_root.mkdir(parents=True)
    _write_split(scene_root, "train", [IDENTITY_POSE, TRANSLATED_POSE])
    _write_split(scene_root, "val", [IDENTITY_POSE])
    _write_split(scene_root, "test", [TRANSLATED_POSE])
    config = tiny_config()
    config.update(
        dataset_root=str(tiny_blender_root), training_iterations=1,
        preview_width=4, preview_height=4,
        checkpoint_interval=1, preview_interval=1, progress_interval=1,
    )
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(training, "PROJECT_ROOT", tmp_path)
    run = training.train(argparse.Namespace(
        config=config_path, resume=None, stop_after=None, extend_to=None,
        stage_checkpoints=None, device="cpu", label="tiny_step4",
    ))
    source_path = run / "checkpoints" / "iter_000001.pt"
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    source = torch.load(source_path, weights_only=True)

    resumed = training.train(argparse.Namespace(
        config=None, resume=run, stop_after=None, extend_to=2,
        stage_checkpoints="2", device="cpu", label="unused",
    ))
    assert resumed == run
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_sha
    assert yaml.safe_load((run / "config_snapshot.yaml").read_text(encoding="utf-8")) == config
    final = torch.load(run / "checkpoints" / "iter_000002.pt", weights_only=True)
    assert final["iteration"] == 2
    assert final["config"] == source["config"]
    assert final["scheduler"]["last_epoch"] == source["scheduler"]["last_epoch"] + 1
    assert any(
        not torch.equal(final["fine_model"][key], value)
        for key, value in source["fine_model"].items()
    )
    with (run / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        assert [int(row["iteration"]) for row in csv.DictReader(stream)] == [1, 2]
    assert (run / "renders" / "comparison_iter_000002.png").is_file()
    telemetry = json.loads(
        (run / "checkpoints" / "iter_000002_telemetry.json").read_text(encoding="utf-8")
    )
    assert telemetry["iteration"] == 2
    assert telemetry["gpu_peak_allocated_since_stage_start_bytes"] is None
