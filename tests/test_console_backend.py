"""Console contract checks that never launch training or evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from PIL import Image
from fastapi.testclient import TestClient

from nerf_console.api import create_app
from nerf_console.manager import ExperimentManager, PROJECT_ROOT, _pid_is_running
from nerf_step1.training_io import assert_run_config_compatible


BASELINE_ID = "step3_smoke_20260924T165838Z_f3bbd1f5"


@pytest.mark.skipif(not (PROJECT_ROOT / "runs" / BASELINE_ID / "checkpoints").exists(), reason="Requires the optional local historical 50k run, which is not distributed")
def test_existing_baseline_is_read_only_and_truthful():
    client = TestClient(create_app(PROJECT_ROOT))
    run = client.get(f"/api/runs/{BASELINE_ID}").json()
    assert run["iteration"] == 50000
    assert run["training_iterations"] == 500
    assert run["effective_target_iteration"] == 50000
    assert run["research_status"]["status"] == "paused_after_checkpoint"
    assert run["imported"] is True
    assert run["latest_evaluation"]["split"] == "val"
    assert run["latest_evaluation"]["psnr_mean"] > 0
    assert client.get(f"/api/runs/{BASELINE_ID}/sampling").json()["available"] is False


@pytest.mark.skipif(not (PROJECT_ROOT / "runs" / BASELINE_ID / "checkpoints").exists(), reason="Requires the optional local historical 50k run, which is not distributed")
def test_historical_metrics_checkpoints_and_reconstruction_files():
    client = TestClient(create_app(PROJECT_ROOT))
    metrics = client.get(f"/api/runs/{BASELINE_ID}/metrics?start=49998&limit=2").json()
    assert metrics["total"] == 50000
    assert [row["iteration"] for row in metrics["rows"]] == [49999, 50000]
    checkpoints = client.get(f"/api/runs/{BASELINE_ID}/checkpoints").json()["items"]
    assert checkpoints[-1]["iteration"] == 50000
    images = client.get(f"/api/runs/{BASELINE_ID}/reconstructions").json()["items"]
    full_view = next(item for item in images if item["split"] == "test" and
                     item["view_index"] == 0 and item["kind"] == "prediction")
    assert full_view["metrics"]["psnr"] > 0
    assert client.get(full_view["url"]).status_code == 200
    assert client.get(f"/api/runs/{BASELINE_ID}/files/../config_snapshot.yaml").status_code in (403, 404)


@pytest.fixture
def blank_project(tmp_path: Path) -> Path:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "baseline.yaml").write_bytes((PROJECT_ROOT / "configs" / "baseline.yaml").read_bytes())
    scene = tmp_path / "data" / "nerf_synthetic" / "lego"
    scene.mkdir(parents=True)
    (scene / "transforms_train.json").write_text("{}", encoding="utf-8")
    return tmp_path


def test_create_freezes_config_and_rejects_mutation(blank_project: Path):
    manager = ExperimentManager(blank_project)
    config = {"training_iterations": 2, "batch_size": 16, "checkpoint_interval": 1}
    run = manager.create_run(config, auto_start=False)
    run_path = manager.run_path(run["id"])
    frozen = yaml.safe_load((run_path / "config_snapshot.yaml").read_text(encoding="utf-8"))
    assert run["status"] == "created"
    assert frozen["training_iterations"] == 2
    assert_run_config_compatible(run_path, frozen)
    frozen["random_seed"] += 1
    with pytest.raises(ValueError, match="immutable|differs"):
        assert_run_config_compatible(run_path, frozen)
    manifest = json.loads((run_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_settings"]["random_seed"] == 0
    provenance = manifest["console_research_provenance"]
    assert provenance["optimizer"] == "torch.optim.Adam"
    assert provenance["direction_encoding_L"] == frozen["direction_encoding_L"]
    assert provenance["full_view_validation_interval"] is None
    assert (run_path / "config.yaml").is_file()
    assert (run_path / "environment.json").is_file()
    assert all((run_path / folder).is_dir() for folder in
               ("checkpoints", "renders", "evaluations", "artifacts", "logs"))


def test_validation_resume_and_stop_without_process_launch(blank_project: Path):
    manager = ExperimentManager(blank_project)
    with pytest.raises(ValueError, match="dataset_root"):
        manager.validate_config({"dataset_root": "other"})
    with pytest.raises(ValueError, match="positive integer"):
        manager.validate_config({"batch_size": 0})
    with pytest.raises(ValueError, match="unsupported"):
        manager.validate_config({"validation_interval": 500})
    run = manager.create_run({"training_iterations": 3}, auto_start=False)
    run_id = run["id"]
    with pytest.raises(ValueError, match="no checkpoint"):
        manager.start(run_id, resume=True)
    checkpoints = manager.run_path(run_id) / "checkpoints"
    (checkpoints / "iter_000001.pt").write_bytes(b"invalid checkpoint")
    with pytest.raises(ValueError, match="checkpoint resume validation failed"):
        manager.start(run_id, resume=True)
    with pytest.raises(ValueError, match="not actively training"):
        manager.stop(run_id)
    process = Mock()
    process.poll.return_value = None
    process.pid = 12345
    manager._jobs[run_id] = process
    manager._job_kind[run_id] = "train"
    stopped = manager.stop(run_id)
    assert stopped["status"] == "stopping"
    assert (manager.run_path(run_id) / "console_stop.request").read_text(encoding="utf-8").strip() == "user_requested"


@pytest.fixture
def api_fixture_run(blank_project):
    """Isolated software fixture, never a research observation or training run."""
    scene = blank_project / "data" / "nerf_synthetic" / "lego"
    pose = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 4], [0, 0, 0, 1]]
    for split in ("train", "val", "test"):
        (scene / split).mkdir()
        Image.new("RGBA", (4, 4), (0, 0, 0, 255)).save(scene / split / "r_0.png")
        metadata = {"camera_angle_x": 0.6911112070083618,
                    "frames": [{"file_path": f"./{split}/r_0", "transform_matrix": pose}]}
        (scene / f"transforms_{split}.json").write_text(json.dumps(metadata), encoding="utf-8")
    app = create_app(blank_project)
    run = app.state.manager.create_run({"training_iterations": 500, "batch_size": 16},
                                       auto_start=False)
    path = app.state.manager.run_path(run["id"]) / "checkpoints" / "iter_000500.pt"
    path.parent.mkdir(exist_ok=True)
    # Negative-input checks reject before checkpoint deserialization.
    path.write_bytes(b"isolated-software-fixture-not-a-model")
    return app, run["id"]


def test_api_rejects_evaluation_without_checkpoint(api_fixture_run):
    app, run_id = api_fixture_run
    client = TestClient(app)
    response = client.post(f"/api/runs/{run_id}/evaluate",
                           json={"checkpoint": 999999, "split": "val", "views": [0]})
    assert response.status_code == 422
    response = client.post(f"/api/runs/{run_id}/evaluate",
                           json={"checkpoint": 500, "split": "val", "views": [999999]})
    assert response.status_code == 422


def test_websocket_delivers_structured_run_event(api_fixture_run):
    app, run_id = api_fixture_run
    app.state.manager._publish(run_id, "metric_update", metrics={"iteration": 42, "psnr": 12.5})
    with TestClient(app).websocket_connect(f"/ws/runs/{run_id}") as socket:
        event = socket.receive_json()
    assert event["type"] == "metric_update"
    assert event["run_id"] == run_id
    assert event["metrics"]["iteration"] == 42


@pytest.mark.skipif(not (PROJECT_ROOT / "runs" / BASELINE_ID / "checkpoints").exists(),
                    reason="Requires the optional local historical 50k checkpoint and dataset")
def test_real_baseline_single_ray_is_read_only_and_consistent():
    """One CPU ray from the completed 50k checkpoint; no training or rendering sweep."""
    import math
    from nerf_step1.evaluation_io import file_sha256

    checkpoint = PROJECT_ROOT / "runs" / BASELINE_ID / "checkpoints" / "iter_050000.pt"
    digest_before = file_sha256(checkpoint)
    modified_before = checkpoint.stat().st_mtime_ns
    client = TestClient(create_app(PROJECT_ROOT))
    response = client.get(
        f"/api/runs/{BASELINE_ID}/sampling/ray?checkpoint=50000&split=val&view_index=0&x=400&y=400"
    )
    assert response.status_code == 200, response.text
    ray = response.json()
    assert ray["available"] is True
    assert ray["source"]["stored_artifact"] is False
    assert ray["source"]["checkpoint_sha256"] == digest_before
    assert len(ray["coarse_depths"]) == len(ray["coarse"]["density"]) == 64
    assert len(ray["fine_depths"]) == 128
    assert len(ray["combined_depths"]) == len(ray["fine"]["density"]) == 192
    assert all(0 <= alpha <= 1 for alpha in ray["coarse"]["alpha"])
    assert all(density >= 0 for density in ray["fine"]["density"])
    for stage in (ray["coarse"], ray["fine"]):
        assert all(math.isclose(weight, alpha * transmittance, rel_tol=1e-5, abs_tol=1e-6)
                   for weight, alpha, transmittance in zip(
                       stage["weights"], stage["alpha"], stage["transmittance"]))
    assert checkpoint.stat().st_mtime_ns == modified_before
    assert file_sha256(checkpoint) == digest_before


def test_ray_endpoint_rejects_invalid_pixel_and_missing_checkpoint(api_fixture_run):
    app, run_id = api_fixture_run
    client = TestClient(app)
    root = f"/api/runs/{run_id}/sampling/ray"
    response = client.get(root + "?checkpoint=500&x=9999&y=0")
    assert response.status_code == 422
    assert "pixel" in response.json()["detail"]
    response = client.get(root + "?checkpoint=999999&x=0&y=0")
    assert response.status_code == 404
    assert "checkpoint" in response.json()["detail"]


def test_train_script_stop_saves_at_complete_iteration(tmp_path: Path, monkeypatch):
    """The optional stop hook persists a checkpoint without touching train math."""
    import argparse
    import sys
    from types import SimpleNamespace

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import train_nerf

    config = {"dataset_root": "data/nerf_synthetic", "scene": "lego", "random_seed": 0,
              "learning_rate": 0.001, "batch_size": 4, "normalize_ray_directions": False,
              "training_iterations": 10, "progress_interval": 100,
              "checkpoint_interval": 10, "preview_interval": 10,
              "render_chunk_size": 1, "preview_width": 1, "preview_height": 1}
    stop_file = tmp_path / "stop.request"
    (tmp_path / "metrics.csv").write_text("iteration,elapsed_time\n", encoding="utf-8")
    class Model:
        def to(self, _device):
            return self
        def parameters(self):
            return []
    class Optimizer:
        def __init__(self, *_args, **kwargs):
            self.param_groups = [{"lr": kwargs["lr"]}]
    monkeypatch.setattr(train_nerf, "read_training_config", lambda _path: config)
    monkeypatch.setattr(train_nerf, "seed_all", lambda _seed: None)
    monkeypatch.setattr(train_nerf, "create_training_run", lambda *_args: tmp_path)
    monkeypatch.setattr(train_nerf, "load_blender_scene",
                        lambda *_args, **_kwargs: SimpleNamespace(train=object(), val=object()))
    monkeypatch.setattr(train_nerf, "CoarseFineNeRF", lambda _config: Model())
    monkeypatch.setattr(train_nerf.torch.optim, "Adam", Optimizer)
    monkeypatch.setattr(train_nerf, "make_scheduler", lambda *_args: object())
    monkeypatch.setattr(train_nerf, "sample_training_batch", lambda *_args, **_kwargs: (None, None, None))
    def one_step(*_args, **_kwargs):
        stop_file.write_text("user_requested", encoding="utf-8")
        return {"total_loss": 1.0, "coarse_loss": 0.5, "fine_loss": 0.5, "PSNR": 10.0}
    monkeypatch.setattr(train_nerf, "train_step", one_step)
    monkeypatch.setattr(train_nerf, "append_metrics", lambda *_args: None)
    monkeypatch.setattr(train_nerf, "_render_preview", lambda *_args: {"contact_sheet": tmp_path / "preview.png"})
    saved = []
    def save(_directory, iteration, *_args):
        saved.append(iteration)
        return tmp_path / f"iter_{iteration:06d}.pt"
    monkeypatch.setattr(train_nerf, "_save_checkpoint", save)
    args = argparse.Namespace(config=tmp_path / "config.yaml", resume=None, run_directory=None,
                              stop_file=stop_file, stop_after=None, extend_to=None,
                              stage_checkpoints=None, device="cpu", label="test")
    assert train_nerf.train(args) == tmp_path
    assert saved == [1]
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "stopped"
    assert status["last_completed_iteration"] == 1
    assert status["termination_reason"] == "user_requested"


def test_recovered_pid_requires_matching_identity(tmp_path: Path, monkeypatch):
    import psutil
    class Process:
        def __init__(self, _pid):
            pass
        def cmdline(self):
            return ["python", "-m", "nerf_console.worker", "--run", str(tmp_path)]
        def is_running(self):
            return True
        def status(self):
            return "running"
        def create_time(self):
            return 1234.5
    monkeypatch.setattr(psutil, "Process", Process)
    assert _pid_is_running(99, create_time=1234.5, run_path=tmp_path)
    assert not _pid_is_running(99, create_time=1235.0, run_path=tmp_path)
    assert not _pid_is_running(99, create_time=1234.5, run_path=tmp_path / "another")
