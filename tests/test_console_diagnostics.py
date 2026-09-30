"""Console adapters must preserve scientific provenance and path boundaries."""

import csv
import hashlib
import json
import math

import pytest
import torch
import yaml
from PIL import Image

from nerf_step1.positional_encoding import SinusoidalPositionalEncoding
from nerf_console_diagnostics import (
    camera_geometry,
    hierarchical_sampling_artifacts,
    list_reconstructions,
    positional_encoding_info,
    read_evaluations,
    read_run_inventory,
    read_run_metadata,
    resolve_run_directory,
    write_baseline_migration_manifest,
)


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_legacy_staged_run_metadata_preserves_original_budget(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _json(run / "run_manifest.json", {"training_settings": {
        "training_iterations": 500, "direction_encoding_L": 4,
        "density_initial_bias": 0.1}})
    _json(run / "status.json", {"status": "paused_after_checkpoint", "last_completed_iteration": 50000})
    _json(run / "baseline_extension.json", {"latest_requested_target_iteration": 50000})
    checkpoint = run / "checkpoints" / "iter_050000.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"not loaded by the reader")

    record = read_run_metadata(run)
    assert record["raw_status"] == "paused_after_checkpoint"
    assert record["completed_target"] is True
    assert record["migration"]["fields"]["original_training_iterations"] == {
        "value": 500, "source": "manifest.training_settings"}
    assert record["migration"]["fields"]["requested_stop_iteration"] == {
        "value": 50000, "source": "baseline_extension"}
    assert not (run / "console_migration_manifest.json").exists()


def test_missing_checkpoint_does_not_mark_target_complete(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _json(run / "run_manifest.json", {"training_settings": {"training_iterations": 500}})
    _json(run / "status.json", {"status": "paused_after_checkpoint", "last_completed_iteration": 500})
    assert read_run_metadata(run)["completed_target"] is False


def test_config_snapshot_digest_mismatch_is_visible(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    original = b"scene: lego\n"
    (run / "config_snapshot.yaml").write_bytes(original + b"changed: true\n")
    _json(run / "run_manifest.json", {"config_sha256": hashlib.sha256(original).hexdigest(),
                                      "training_settings": {"training_iterations": 500}})
    assert read_run_metadata(run)["config_digest_matches_manifest"] is False


def test_resolve_run_directory_rejects_traversal_and_symlink(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    (root / "safe").mkdir()
    assert resolve_run_directory(root, "safe") == (root / "safe").resolve()
    with pytest.raises(ValueError):
        resolve_run_directory(root, "../outside")
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (root / "escape").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError):
        resolve_run_directory(root, "escape")


def test_reconstructions_join_only_exact_measured_view(tmp_path):
    run = tmp_path / "run"
    render = run / "evaluation" / "renders" / "iter_000500"
    render.mkdir(parents=True)
    for name in ("val_view_000_prediction.png", "val_view_000_ground_truth.png",
                 "val_view_001_prediction.png"):
        (render / name).write_bytes(b"png fixture")
    _csv(run / "evaluation_results.csv",
         ("iteration", "split", "view_index", "psnr", "ssim", "lpips"),
         [{"iteration": 500, "split": "val", "view_index": 0,
           "psnr": 23.25, "ssim": 0.8, "lpips": 0.2}])
    rows = list_reconstructions(run)
    assert len(rows) == 2
    assert rows[0]["images"] == {
        "ground_truth": "evaluation/renders/iter_000500/val_view_000_ground_truth.png",
        "prediction": "evaluation/renders/iter_000500/val_view_000_prediction.png"}
    assert rows[0]["metrics"]["psnr"] == 23.25
    assert rows[1]["metrics"] is None
    assert read_evaluations(run)["summary_available"] is False
    assert read_run_inventory(run)["reconstruction_image_count"] == 3


def test_sampling_is_unavailable_without_persisted_ray_trace(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    figure = tmp_path / "artifacts" / "hierarchical_sampling_visualization.png"
    figure.parent.mkdir()
    figure.write_bytes(b"diagnostic image")
    result = hierarchical_sampling_artifacts(run, tmp_path)
    assert result["available"] is False
    assert result["weights"] is None
    assert result["static_figures"][0]["measured_ray_trace"] is False


def test_position_encoding_reports_validated_zero_and_last_band():
    zero = positional_encoding_info(0)
    assert zero["encoded_dim"] == 3
    assert len(zero["x"]) == 180
    assert zero["x"][0] == 0.0 and zero["x"][-1] == 1.0
    assert zero["curves"] == []
    info = positional_encoding_info(10)
    assert info["encoded_dim"] == 63
    assert info["highest_frequency_band"] == 512
    assert info["bands"][-1]["k"] == 9
    assert len(info["curves"]) == 10
    assert all(len(curve["sin"]) == len(curve["cos"]) == 180 for curve in info["curves"])
    x = torch.tensor(info["x"], dtype=torch.float64)
    inputs = torch.stack((x, torch.zeros_like(x), torch.zeros_like(x)), dim=-1)
    encoded = SinusoidalPositionalEncoding(3, 10)(inputs)
    assert info["curves"][0]["sin"] == pytest.approx(encoded[:, 3].tolist())
    assert info["curves"][0]["cos"] == pytest.approx(encoded[:, 6].tolist())
    assert info["curves"][9]["sin"] == pytest.approx(encoded[:, 57].tolist())
    assert info["curves"][9]["cos"] == pytest.approx(encoded[:, 60].tolist())
    with pytest.raises(ValueError):
        positional_encoding_info(True)


def test_camera_geometry_matches_validated_opengl_ray_convention(tmp_path):
    scene = tmp_path / "lego"
    scene.mkdir()
    pose = [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]
    for split in ("train", "val", "test"):
        Image.new("RGB", (4, 4)).save(scene / f"{split}.png")
        _json(scene / f"transforms_{split}.json", {
            "camera_angle_x": math.pi / 2,
            "frames": [{"file_path": f"./{split}", "transform_matrix": pose}]})
    result = camera_geometry(tmp_path, "lego", split="train", pixel_xy=(0, 0))
    camera = result["cameras"][0]
    assert result["total"] == 1
    assert camera["position"] == [1.0, 2.0, 3.0]
    assert camera["forward"] == [0.0, 0.0, -1.0]
    assert camera["center_ray"] == [0.0, 0.0, -1.0]
    assert camera["selected_ray"]["direction"] == pytest.approx([-1.0, 1.0, -1.0])


def test_baseline_migration_sidecar_is_independent_and_idempotent(tmp_path):
    run = tmp_path / "step3_smoke_20260924T165838Z_f3bbd1f5"
    run.mkdir()
    settings = {"scene": "lego", "position_encoding_L": 10,
                "direction_encoding_L": 4, "random_seed": 0,
                "batch_size": 256, "training_iterations": 500}
    snapshot = yaml.safe_dump(settings, sort_keys=True).encode("utf-8")
    (run / "config_snapshot.yaml").write_bytes(snapshot)
    _json(run / "run_manifest.json", {
        "scene": "lego", "training_settings": settings,
        "config_sha256": hashlib.sha256(snapshot).hexdigest()})
    _json(run / "environment_snapshot.json", {"gpu_name": "fixture GPU"})
    _json(run / "status.json", {"status": "paused_after_checkpoint", "last_completed_iteration": 50000})
    _json(run / "baseline_extension.json", {"latest_requested_target_iteration": 50000})
    checkpoint = run / "checkpoints" / "iter_050000.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"unloaded checkpoint fixture")
    _csv(run / "metrics.csv", ("iteration", "total_loss"),
         [{"iteration": 50000, "total_loss": 0.01}])
    _csv(run / "evaluation_results.csv",
         ("iteration", "split", "view_index", "psnr", "ssim", "lpips"), [])
    _csv(run / "evaluation_summary.csv",
         ("iteration", "split", "view_count", "psnr_mean", "ssim_mean", "lpips_mean"), [])
    original_manifest = (run / "run_manifest.json").read_bytes()
    original_snapshot = (run / "config_snapshot.yaml").read_bytes()

    created = write_baseline_migration_manifest(run)
    sidecar = run / "console_migration_manifest.json"
    sidecar_bytes = sidecar.read_bytes()
    assert created["fields"]["original_frozen_training_budget"]["value"] == 500
    assert created["fields"]["staged_target_iteration"]["value"] == 50000
    assert created["fields"]["batch_size"]["value"] == 256
    assert created["unavailable"]["hierarchical_sampling_per_ray_values"]["available"] is False
    assert write_baseline_migration_manifest(run) == created
    assert sidecar.read_bytes() == sidecar_bytes
    assert (run / "run_manifest.json").read_bytes() == original_manifest
    assert (run / "config_snapshot.yaml").read_bytes() == original_snapshot

    _json(sidecar, {"wrong": "content"})
    with pytest.raises(ValueError, match="existing migration manifest"):
        write_baseline_migration_manifest(run)
    assert json.loads(sidecar.read_text(encoding="utf-8")) == {"wrong": "content"}
