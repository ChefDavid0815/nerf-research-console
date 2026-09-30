"""Process lifecycle and immutable run management for the local console."""

from __future__ import annotations

import csv
from functools import lru_cache
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
import psutil
import torch

from nerf_step1.pipeline import CoarseFineNeRF, make_scheduler, validate_training_config
from nerf_step1.training_io import assert_run_config_compatible, create_training_run, load_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = re.compile(r"^[A-Za-z0-9_-]+$")
CHECKPOINT = re.compile(r"iter_(\d+)\.pt$")


def _pid_is_running(pid: int, *, create_time: float, run_path: Path) -> bool:
    if pid <= 0:
        return False
    try:
        process = psutil.Process(pid)
        command = process.cmdline()
        return (process.is_running() and process.status() != psutil.STATUS_ZOMBIE
                and abs(process.create_time() - create_time) < 0.01
                and "nerf_console.worker" in command
                and str(run_path.resolve()) in command)
    except (psutil.Error, OSError):
        return False


class _RecoveredProcess:
    """Observe a training child that outlived a restarted API process."""

    def __init__(self, pid: int, run_path: Path, create_time: float):
        self.pid = pid
        self.run_path = run_path
        self.create_time = create_time
        self.returncode: int | None = None

    def poll(self) -> int | None:
        if _pid_is_running(self.pid, create_time=self.create_time, run_path=self.run_path):
            return None
        if self.returncode is None:
            try:
                status = json.loads((self.run_path / "status.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                status = {}
            self.returncode = 0 if status.get("status") in ("completed", "paused_after_checkpoint", "stopped") else 1
        return self.returncode


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if value is None or value == "":
                    parsed[key] = None
                    continue
                if key in ("split", "view_indices"):
                    parsed[key] = value
                    continue
                try:
                    number = float(value)
                    parsed[key] = int(number) if number.is_integer() else number
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
        return rows


@lru_cache(maxsize=128)
def _parameter_count(config_sha256: str, config_json: str) -> int:
    """Count real coarse and fine model tensors from the frozen architecture."""
    del config_sha256  # Names the cache identity; config_json is the actual input.
    model = CoarseFineNeRF(json.loads(config_json))
    return sum(parameter.numel() for parameter in model.parameters())


class ExperimentManager:
    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.project_root = Path(project_root).resolve()
        self.runs_root = self.project_root / "runs"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: dict[str, subprocess.Popen | _RecoveredProcess] = {}
        self._job_kind: dict[str, str] = {}
        self._events: dict[str, list[dict]] = {}
        self._replays: set[str] = set()
        self._recover_training_jobs()

    def _recover_training_jobs(self) -> None:
        for path in self.runs_root.iterdir():
            if not path.is_dir() or not (path / "run_manifest.json").is_file():
                continue
            state = self._json(path / "console_state.json")
            pid = state.get("pid")
            started = state.get("process_create_time")
            if state.get("status") in ("running", "stopping") and isinstance(pid, int) and isinstance(started, (int, float)):
                observer = _RecoveredProcess(pid, path, float(started))
                self._jobs[path.name] = observer
                self._job_kind[path.name] = "train"
                threading.Thread(target=self._monitor, args=(path.name, observer, "train"), daemon=True).start()

    def run_path(self, run_id: str) -> Path:
        if not RUN_ID.fullmatch(run_id):
            raise ValueError("invalid run ID")
        path = self.runs_root / run_id
        if not path.is_dir() or not (path / "run_manifest.json").is_file():
            raise FileNotFoundError(run_id)
        return path

    def defaults(self) -> dict:
        with (self.project_root / "configs" / "baseline.yaml").open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        return {"config": config, "validated_baseline": True,
                "reference_config": "configs/baseline.yaml",
                "training_budget_verified": False,
                "training_budget_note": "The 500000-iteration reference budget has not been run; the imported observed Lego baseline reached 50000 iterations by a recorded extension.",
                "position_encoding": {"include_input": True, "dimension_formula": "3 + 6L"},
                "capabilities": {
                    "periodic_full_view_validation": False,
                    "validation_interval": None,
                    "validation_note": (
                        "The validated training script saves diagnostic previews at preview_interval; "
                        "full-view PSNR, SSIM and LPIPS are produced by an explicit checkpoint evaluation."
                    ),
                }}

    def validate_config(self, proposed: dict) -> dict:
        if not isinstance(proposed, dict):
            raise ValueError("config must be an object")
        default = self.defaults()["config"]
        unknown = set(proposed) - set(default)
        if unknown:
            if "validation_interval" in unknown:
                raise ValueError("validation_interval is unsupported by the validated trainer; use preview_interval and explicit checkpoint evaluation")
            raise ValueError(f"unknown training config fields: {', '.join(sorted(unknown))}")
        config = {**default, **proposed}
        if config["dataset_root"] != default["dataset_root"]:
            raise ValueError("dataset_root must remain the validated local Lego dataset")
        validate_training_config(config)
        data_path = self.project_root / config["dataset_root"] / config["scene"]
        if not (data_path / "transforms_train.json").is_file():
            raise ValueError("Lego training dataset is unavailable")
        return config

    def create_run(self, config: dict, *, auto_start: bool = False, label: str | None = None) -> dict:
        frozen = self.validate_config(config)
        label = label or f"{frozen['scene']}_L{frozen['position_encoding_L']}_seed{frozen['random_seed']}"
        if not RUN_ID.fullmatch(label):
            raise ValueError("label must contain letters, digits, underscores or hyphens")
        path = create_training_run(self.runs_root, label, frozen)
        manifest_path = path / "run_manifest.json"
        manifest = self._json(manifest_path)
        manifest["console_research_provenance"] = {
            "source": "validated scripts/train_nerf.py + nerf_step1 pipeline",
            "scene": frozen["scene"],
            "dataset_root": frozen["dataset_root"],
            "position_encoding_L": frozen["position_encoding_L"],
            "direction_encoding_L": frozen["direction_encoding_L"],
            "position_encoding_convention": "include_input=true; sin/cos(2^k*pi*x), k=0..L-1",
            "architecture": {key: frozen[key] for key in (
                "network_depth", "network_width", "skip_connection_layer", "view_width")},
            "sampling": {key: frozen[key] for key in (
                "num_coarse_samples", "num_fine_samples", "near", "far",
                "normalize_ray_directions", "white_background")},
            "optimizer": "torch.optim.Adam",
            "learning_rate": frozen["learning_rate"],
            "batch_size": frozen["batch_size"],
            "seed": frozen["random_seed"],
            "training_budget_iterations": frozen["training_iterations"],
            "checkpoint_interval": frozen["checkpoint_interval"],
            "diagnostic_preview_interval": frozen["preview_interval"],
            "full_view_validation_interval": None,
            "density_initialization_bias": frozen["density_initial_bias"],
            "software_versions": {
                "python": self._json(path / "environment_snapshot.json").get("python"),
                "pytorch": manifest.get("pytorch_version"),
                "cuda_runtime": manifest.get("cuda_runtime"),
            },
            "gpu": manifest.get("gpu"),
            "timestamp_utc": manifest.get("timestamp_utc"),
            "git_commit": manifest.get("git_commit"),
        }
        temporary_manifest = manifest_path.with_suffix(".tmp")
        temporary_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary_manifest, manifest_path)
        for directory in ("checkpoints", "renders", "evaluations", "artifacts", "logs"):
            (path / directory).mkdir(exist_ok=True)
        (path / "evaluations" / "index.json").write_text(json.dumps({
            "canonical_evaluator": "scripts/evaluate_nerf.py",
            "metadata_directory": "evaluation/",
            "per_view_table": "evaluation_results.csv",
            "summary_table": "evaluation_summary.csv",
            "note": "The validated Step 4 evaluator writes to these existing paths; no results exist until evaluation runs.",
        }, indent=2) + "\n", encoding="utf-8")
        # The research script reads config_snapshot.yaml; these are stable console aliases.
        (path / "config.yaml").write_bytes((path / "config_snapshot.yaml").read_bytes())
        (path / "environment.json").write_bytes((path / "environment_snapshot.json").read_bytes())
        (path / "console_origin.json").write_text(json.dumps({"created_by": "nerf_console"}), encoding="utf-8")
        self._write_state(path, "created")
        if auto_start:
            self.start(path.name)
        return self.get_run(path.name)

    @staticmethod
    def _write_state(path: Path, status: str, **extra: Any) -> None:
        target = path / "console_state.json"
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps({"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(), **extra},
                                   indent=2), encoding="utf-8")
        os.replace(temp, target)

    @staticmethod
    def _json(path: Path) -> dict:
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _publish(self, run_id: str, event_type: str, **data: Any) -> None:
        with self._lock:
            events = self._events.setdefault(run_id, [])
            events.append({"seq": events[-1]["seq"] + 1 if events else 1,
                           "type": event_type, "run_id": run_id,
                           "timestamp_utc": datetime.now(timezone.utc).isoformat(), **data})
            if len(events) > 500:
                del events[:-500]

    def events_after(self, run_id: str, seq: int) -> list[dict]:
        self.run_path(run_id)
        with self._lock:
            return [item for item in self._events.get(run_id, []) if item["seq"] > seq]

    def _active_training(self) -> bool:
        return any(self._job_kind.get(key) == "train" and job.poll() is None
                   for key, job in self._jobs.items())

    def _spawn(self, run_id: str, command: list[str], kind: str,
               *, env: dict[str, str] | None = None) -> None:
        path = self.run_path(run_id)
        with self._lock:
            prior = self._jobs.get(run_id)
            if prior and prior.poll() is None:
                raise ValueError("run already has an active process")
            if kind == "train" and self._active_training():
                raise ValueError("another training run is active")
            (path / "logs").mkdir(exist_ok=True)
            output = (path / "logs" / f"console_{kind}.log").open("ab", buffering=0)
            try:
                proc = subprocess.Popen(command, cwd=self.project_root, env=env, stdout=output,
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0))
            finally:
                output.close()
            self._jobs[run_id] = proc
            self._job_kind[run_id] = kind
            try:
                process_create_time = psutil.Process(proc.pid).create_time()
            except psutil.Error:
                process_create_time = None
            self._write_state(path, "running" if kind == "train" else "evaluating",
                              pid=proc.pid, process_create_time=process_create_time,
                              stop_request=str(path / "console_stop.request") if kind == "train" else None)
            self._publish(run_id, "training_started" if kind == "train" else "evaluation_started", pid=proc.pid)
            threading.Thread(target=self._monitor, args=(run_id, proc, kind), daemon=True).start()

    def _monitor(self, run_id: str, proc: subprocess.Popen | _RecoveredProcess, kind: str) -> None:
        path = self.run_path(run_id)
        previous_iteration = -1
        seen_checkpoints: set[str] = set()
        seen_renders: set[str] = set()
        while proc.poll() is None:
            if kind == "train":
                # Read only the last complete CSV record for live updates.
                latest = self._last_metric(path / "metrics.csv")
                if latest and int(latest["iteration"]) != previous_iteration:
                    previous_iteration = int(latest["iteration"])
                    self._publish(run_id, "iteration_update", iteration=previous_iteration)
                    self._publish(run_id, "metric_update", metrics=latest)
                for checkpoint in (path / "checkpoints").glob("iter_*.pt"):
                    match = CHECKPOINT.fullmatch(checkpoint.name)
                    if match and checkpoint.name not in seen_checkpoints:
                        seen_checkpoints.add(checkpoint.name)
                        self._publish(run_id, "checkpoint_saved", iteration=int(match.group(1)))
                for render in (path / "renders").glob("comparison_iter_*.png"):
                    if render.name not in seen_renders:
                        seen_renders.add(render.name)
                        self._publish(run_id, "preview_ready", url=self.artifact_url(run_id, render.relative_to(path)))
            time.sleep(0.5)
        if kind == "train":
            latest = self._last_metric(path / "metrics.csv")
            if latest and int(latest["iteration"]) != previous_iteration:
                self._publish(run_id, "iteration_update", iteration=int(latest["iteration"]))
                self._publish(run_id, "metric_update", metrics=latest)
            for checkpoint in (path / "checkpoints").glob("iter_*.pt"):
                match = CHECKPOINT.fullmatch(checkpoint.name)
                if match and checkpoint.name not in seen_checkpoints:
                    self._publish(run_id, "checkpoint_saved", iteration=int(match.group(1)))
            for render in (path / "renders").glob("comparison_iter_*.png"):
                if render.name not in seen_renders:
                    self._publish(run_id, "preview_ready", url=self.artifact_url(run_id, render.relative_to(path)))
        status = self._json(path / "status.json")
        if kind == "train":
            final = "stopped" if status.get("termination_reason") == "user_requested" else (
                "completed" if proc.returncode == 0 else "failed")
            self._write_state(path, final, exit_code=proc.returncode)
            self._publish(run_id, "training_completed" if final != "failed" else "training_failed",
                          status=final, exit_code=proc.returncode,
                          error=status.get("error") if final == "failed" else None)
        else:
            final = "completed" if proc.returncode == 0 else "failed"
            self._write_state(path, self._json(path / "status.json").get("status", final),
                              evaluation_status=final, exit_code=proc.returncode)
            self._publish(run_id, "evaluation_completed" if final == "completed" else "training_failed",
                          status=final, exit_code=proc.returncode)

    @staticmethod
    def _last_metric(path: Path) -> dict | None:
        # The CSV is append-only. Seek from the end to avoid reparsing long histories.
        if not path.is_file():
            return None
        with path.open("rb") as stream:
            header = stream.readline().decode("utf-8").strip().split(",")
            stream.seek(0, os.SEEK_END)
            end = stream.tell()
            if end == 0:
                return None
            stream.seek(-1, os.SEEK_END)
            complete = stream.read(1) == b"\n"
            stream.seek(max(0, end - 4096))
            lines = stream.read().decode("utf-8", errors="replace").splitlines()
        if len(lines) < 2:
            return None
        last = lines[-1] if complete else lines[-2]
        values = next(csv.reader([last]))
        if len(values) != len(header) or not values[0].isdigit():
            return None
        result = {}
        for key, value in zip(header, values):
            result[key] = None if value == "" else float(value)
        result["iteration"] = int(result["iteration"])
        return result

    @staticmethod
    def _validate_resume_checkpoint(checkpoint_path: Path, config: dict,
                                    expected_iteration: int) -> None:
        try:
            model = CoarseFineNeRF(config)
            optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
            scheduler = make_scheduler(optimizer, config)
            loaded_iteration = load_checkpoint(
                checkpoint_path, coarse_model=model.coarse_model,
                fine_model=model.fine_model, optimizer=optimizer,
                scheduler=scheduler, expected_config=config,
                map_location="cpu", restore_rng=False,
            )
            if loaded_iteration != expected_iteration:
                raise ValueError("checkpoint filename and payload iteration differ")
        except Exception as exc:
            raise ValueError(f"checkpoint resume validation failed: {exc}") from exc

    def start(self, run_id: str, *, resume: bool = False, device: str = "auto") -> dict:
        path = self.run_path(run_id)
        config = yaml.safe_load((path / "config_snapshot.yaml").read_text(encoding="utf-8"))
        assert_run_config_compatible(path, config)
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("invalid device")
        if resume:
            checkpoints = self.checkpoints(run_id)["items"]
            if not checkpoints:
                raise ValueError("run has no checkpoint to resume")
            if checkpoints[-1]["iteration"] >= config["training_iterations"]:
                raise ValueError("run reached its frozen training budget")
            checkpoint_path = path / "checkpoints" / checkpoints[-1]["name"]
            self._validate_resume_checkpoint(checkpoint_path, config, checkpoints[-1]["iteration"])
        elif self._json(path / "console_state.json").get("status") != "created":
            raise ValueError("only a new frozen run can be started; use resume for stopped runs")
        (path / "console_stop.request").unlink(missing_ok=True)
        command = [sys.executable, "-m", "nerf_console.worker", "--run", str(path), "--device", device]
        if resume:
            command.append("--resume")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.project_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        # _spawn receives an explicit environment through an override below.
        self._spawn(run_id, command, "train", env=env)
        return self.get_run(run_id)

    def stop(self, run_id: str) -> dict:
        path = self.run_path(run_id)
        with self._lock:
            job = self._jobs.get(run_id)
            if not job or job.poll() is not None or self._job_kind.get(run_id) != "train":
                raise ValueError("run is not actively training in this console process")
            (path / "console_stop.request").write_text("user_requested\n", encoding="utf-8")
            state = self._json(path / "console_state.json")
            self._write_state(path, "stopping", pid=job.pid,
                              process_create_time=state.get("process_create_time"),
                              stop_request=str(path / "console_stop.request"))
        return self.get_run(run_id)

    def checkpoints(self, run_id: str) -> dict:
        path = self.run_path(run_id)
        items = []
        for file in (path / "checkpoints").glob("iter_*.pt"):
            match = CHECKPOINT.fullmatch(file.name)
            if match:
                items.append({"iteration": int(match.group(1)), "name": file.name,
                              "size_bytes": file.stat().st_size,
                              "url": self.artifact_url(run_id, file.relative_to(path))})
        items.sort(key=lambda value: value["iteration"])
        return {"items": items}

    @staticmethod
    def artifact_url(run_id: str, relative: Path) -> str:
        return f"/api/runs/{run_id}/files/{relative.as_posix()}"

    def evaluate(self, run_id: str, *, checkpoint: int, split: str, views: list[int],
                 device: str = "auto", render_chunk_size: int | None = None) -> dict:
        path = self.run_path(run_id)
        if split not in ("train", "val", "test") or not views or any(v < 0 for v in views):
            raise ValueError("split and nonnegative view indices are required")
        if len(set(views)) != len(views):
            raise ValueError("view indices must be unique")
        if render_chunk_size is not None and (isinstance(render_chunk_size, bool) or render_chunk_size <= 0):
            raise ValueError("render_chunk_size must be a positive integer")
        config = yaml.safe_load((path / "config_snapshot.yaml").read_text(encoding="utf-8"))
        transforms = self.project_root / config["dataset_root"] / config["scene"] / f"transforms_{split}.json"
        if not transforms.is_file():
            raise ValueError(f"{split} split metadata is unavailable")
        view_count = len(json.loads(transforms.read_text(encoding="utf-8")).get("frames", []))
        if any(view >= view_count for view in views):
            raise ValueError(f"{split} view index exceeds {view_count - 1}")
        available = {item["iteration"] for item in self.checkpoints(run_id)["items"]}
        if checkpoint not in available:
            raise ValueError("checkpoint is unavailable")
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in ("cpu", "cuda"):
            raise ValueError("invalid device")
        command = [sys.executable, str(self.project_root / "scripts" / "evaluate_nerf.py"),
                   "--run", str(path), "--checkpoint", str(checkpoint),
                   "--split", split, "--views", *[str(v) for v in views], "--device", device]
        if render_chunk_size is not None:
            command.extend(["--render-chunk-size", str(render_chunk_size)])
        self._spawn(run_id, command, "evaluate")
        return {"status": "evaluating", "run_id": run_id, "checkpoint": checkpoint, "split": split, "views": views}

    def get_run(self, run_id: str) -> dict:
        path = self.run_path(run_id)
        manifest = self._json(path / "run_manifest.json")
        config = yaml.safe_load((path / "config_snapshot.yaml").read_text(encoding="utf-8"))
        environment = self._json(path / "environment_snapshot.json")
        migration = self._json(path / "console_migration_manifest.json")
        state = self._json(path / "console_state.json")
        research_status = self._json(path / "status.json")
        extension = self._json(path / "baseline_extension.json")
        metrics = self._last_metric(path / "metrics.csv")
        summaries = read_csv(path / "evaluation_summary.csv")
        latest_eval = max((row for row in summaries if row.get("split") == "val"),
                          key=lambda row: row.get("iteration", -1), default=None)
        if latest_eval:
            latest_eval = {**latest_eval, "psnr": latest_eval.get("psnr_mean"),
                           "ssim": latest_eval.get("ssim_mean"),
                           "lpips": latest_eval.get("lpips_mean")}
        status = state.get("status") or research_status.get("status") or "unknown"
        parameter_count = _parameter_count(manifest.get("config_sha256", ""),
                                           json.dumps(config, sort_keys=True))
        effective_target = (extension.get("latest_requested_target_iteration")
                            or research_status.get("requested_stop_iteration")
                            or config.get("training_iterations"))
        if status == "paused_after_checkpoint":
            reached = bool(metrics and effective_target and metrics["iteration"] >= effective_target)
            status = "completed" if reached else "stopped"
        traceback = None
        if status == "failed":
            log = path / "train.log"
            if log.is_file():
                content = log.read_text(encoding="utf-8", errors="replace")
                marker = "Traceback (most recent call last):"
                if marker in content:
                    traceback = marker + content.rsplit(marker, 1)[-1]
        return {"id": run_id, "scene": config.get("scene"), "status": status,
                "iteration": metrics.get("iteration") if metrics else research_status.get("last_completed_iteration", 0),
                "duration_seconds": metrics.get("elapsed_time") if metrics else None,
                "training_iterations": config.get("training_iterations"),
                "effective_target_iteration": effective_target,
                "position_encoding_L": config.get("position_encoding_L"),
                "direction_encoding_L": config.get("direction_encoding_L"),
                "parameter_count": parameter_count,
                "seed": config.get("random_seed"), "timestamp_utc": manifest.get("timestamp_utc"),
                "latest_metrics": metrics, "latest_evaluation": latest_eval,
                "imported": not (path / "console_origin.json").exists(),
                "featured_baseline": bool(migration),
                "manifest": manifest, "config": config, "environment": environment,
                "research_status": research_status, "baseline_extension": extension or None,
                "migration": migration or None,
                "console_state": state,
                "termination_reason": research_status.get("termination_reason"),
                "error": research_status.get("error"), "traceback": traceback,
                "availability": {"metrics": (path / "metrics.csv").is_file(),
                                 "evaluations": bool(summaries),
                                 "checkpoints": bool(self.checkpoints(run_id)["items"]),
                                 "sampling": False,
                                 "sampling_on_demand": bool(self.checkpoints(run_id)["items"])}}

    def list_runs(self) -> list[dict]:
        runs = []
        for path in self.runs_root.iterdir():
            if path.is_dir() and (path / "run_manifest.json").is_file() and (path / "config_snapshot.yaml").is_file():
                try:
                    runs.append(self.get_run(path.name))
                except (OSError, ValueError, KeyError):
                    continue
        runs.sort(key=lambda item: item["timestamp_utc"] or "", reverse=True)
        return runs
