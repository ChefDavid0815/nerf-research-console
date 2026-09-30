"""Run records, resumable checkpoints, and scalar metrics for Step 3 training.

One process writes a run directory. The configuration is written once and its
digest is checked before every checkpoint operation; changing an in-memory
configuration or editing the snapshot cannot silently change a resumed run.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from .run_logging import environment_snapshot


METRIC_FIELDS = (
    "iteration",
    "total_loss",
    "coarse_loss",
    "fine_loss",
    "psnr",
    "learning_rate",
    "elapsed_time",
    "rays_per_second",
    "gpu_memory_allocated_bytes",
    "gpu_memory_reserved_bytes",
)
_REQUIRED_METRIC_FIELDS = METRIC_FIELDS[:8]
_FORMAT_VERSION = 1


def _normalized_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if not all(isinstance(key, str) for key in config):
        raise ValueError("config keys must be strings")
    # The YAML round trip limits values to portable safe types and detaches the
    # persisted configuration from a caller's mutable nested mappings.
    normalized = yaml.safe_load(yaml.safe_dump(dict(config), sort_keys=True))
    if not isinstance(normalized, dict):
        raise ValueError("config must serialize to a mapping")
    return normalized


def _snapshot_bytes(config: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump(
        _normalized_config(config), sort_keys=True, allow_unicode=True
    ).encode("utf-8")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", commit) else None


def create_training_run(
    base_directory: Path, label: str, config: Mapping[str, Any]
) -> Path:
    """Create a unique run with one-time inputs, environment, manifest and CSV.

    The manifest's SHA-256 protects the original configuration bytes from
    unnoticed edits. This is a local integrity check, not an authenticity seal.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError("label must contain only letters, digits, underscores or hyphens")
    frozen = _normalized_config(config)
    for key in ("scene", "position_encoding_L", "random_seed"):
        if key not in frozen:
            raise ValueError(f"training config is missing {key}")

    timestamp = datetime.now(timezone.utc)
    run_directory = Path(base_directory) / (
        f"{label}_{timestamp.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    )
    run_directory.mkdir(parents=True, exist_ok=False)
    snapshot = _snapshot_bytes(frozen)
    (run_directory / "config_snapshot.yaml").write_bytes(snapshot)
    environment = environment_snapshot()
    (run_directory / "environment_snapshot.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (run_directory / "metrics.csv").open("x", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(METRIC_FIELDS)
    manifest = {
        "format_version": _FORMAT_VERSION,
        "timestamp_utc": timestamp.isoformat(),
        "scene": frozen["scene"],
        "position_encoding_L": frozen["position_encoding_L"],
        "seed": frozen["random_seed"],
        "gpu": environment["gpu_name"],
        "pytorch_version": str(environment["pytorch"]),
        "cuda_runtime": environment["pytorch_cuda_runtime"],
        "git_commit": _git_commit(),
        "config_sha256": hashlib.sha256(snapshot).hexdigest(),
        "training_settings": frozen,
    }
    # A manifest is written last so its presence indicates a complete run
    # record, even if setup was interrupted partway through.
    (run_directory / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return run_directory


def assert_run_config_compatible(run_directory: Path, config: Mapping[str, Any]) -> None:
    """Reject a changed run snapshot or any change in the full run config."""
    run_directory = Path(run_directory)
    snapshot = (run_directory / "config_snapshot.yaml").read_bytes()
    manifest = json.loads((run_directory / "run_manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(snapshot).hexdigest() != manifest.get("config_sha256"):
        raise ValueError("run configuration snapshot differs from its manifest digest")
    if yaml.safe_load(snapshot) != _normalized_config(config):
        raise ValueError("configuration differs from the immutable run snapshot")


def _run_directory_for(checkpoint_path: Path) -> Path:
    for parent in checkpoint_path.resolve().parents:
        if (parent / "run_manifest.json").is_file():
            return parent
    raise ValueError("checkpoint must be inside a training run directory")


def _validate_model_architecture(model: nn.Module, config: Mapping[str, Any], name: str) -> None:
    architecture = getattr(model, "architecture_config", None)
    if architecture is None:
        return
    mismatches = [key for key, value in architecture.items() if config.get(key) != value]
    if mismatches:
        raise ValueError(f"{name} architecture disagrees with config: {', '.join(mismatches)}")


def _random_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": (
            numpy_state[0], numpy_state[1].tolist(), numpy_state[2],
            numpy_state[3], numpy_state[4],
        ),
        "python": random.getstate(),
    }


def _restore_random_state(state: Mapping[str, Any]) -> None:
    _validate_random_state_compatibility(state)
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_state = state["torch_cuda"]
    if cuda_state:
        torch.cuda.set_rng_state_all([device_state.cpu() for device_state in cuda_state])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
        numpy_state[2], numpy_state[3], numpy_state[4],
    ))
    random.setstate(state["python"])


def _validate_random_state_compatibility(state: Mapping[str, Any]) -> None:
    cuda_state = state["torch_cuda"]
    if cuda_state:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA RNG state cannot be restored without CUDA")
        if len(cuda_state) != torch.cuda.device_count():
            raise RuntimeError("CUDA device count differs from checkpoint RNG state")


def save_checkpoint(
    path: Path,
    *,
    coarse_model: nn.Module,
    fine_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    config: Mapping[str, Any],
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    """Persist all state needed to continue after ``iteration`` updates.

    The final file is replaced only after a complete write to an adjacent
    temporary file. A previous checkpoint survives interrupted serialization.
    """
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("iteration must be a nonnegative integer")
    path = Path(path)
    frozen = _normalized_config(config)
    assert_run_config_compatible(_run_directory_for(path), frozen)
    _validate_model_architecture(coarse_model, frozen, "coarse model")
    _validate_model_architecture(fine_model, frozen, "fine model")
    checkpoint = {
        "format_version": _FORMAT_VERSION,
        "iteration": iteration,
        "config": frozen,
        "coarse_model": coarse_model.state_dict(),
        "fine_model": fine_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "random_state": _random_state(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "wb") as file:
            torch.save(checkpoint, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _validate_state_shapes(model: nn.Module, saved: Mapping[str, Any], name: str) -> None:
    current = model.state_dict()
    if current.keys() != saved.keys():
        raise ValueError(f"{name} checkpoint parameter names do not match the model")
    for key in current:
        if current[key].shape != saved[key].shape:
            raise ValueError(f"{name} checkpoint shape differs for {key}")


def load_checkpoint(
    path: Path,
    *,
    coarse_model: nn.Module,
    fine_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_config: Mapping[str, Any],
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> int:
    """Restore model, optimizer, scheduler and RNG, returning last completed step.

    Only load checkpoints produced by this project. ``weights_only=True`` also
    keeps the file format within PyTorch's restricted unpickler.
    """
    path = Path(path)
    frozen = _normalized_config(expected_config)
    assert_run_config_compatible(_run_directory_for(path), frozen)
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != _FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format")
    if checkpoint.get("config") != frozen:
        raise ValueError("checkpoint configuration differs from requested run")
    iteration = checkpoint.get("iteration")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("checkpoint iteration is invalid")
    if (checkpoint.get("scheduler") is None) != (scheduler is None):
        raise ValueError("scheduler presence differs from checkpoint")
    _validate_model_architecture(coarse_model, frozen, "coarse model")
    _validate_model_architecture(fine_model, frozen, "fine model")
    _validate_state_shapes(coarse_model, checkpoint["coarse_model"], "coarse model")
    _validate_state_shapes(fine_model, checkpoint["fine_model"], "fine model")
    if restore_rng:
        _validate_random_state_compatibility(checkpoint["random_state"])
    coarse_model.load_state_dict(checkpoint["coarse_model"])
    fine_model.load_state_dict(checkpoint["fine_model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if restore_rng:
        _restore_random_state(checkpoint["random_state"])
    return iteration


def append_metrics(run_directory: Path, row: Mapping[str, Any]) -> None:
    """Append one validated scalar row to this run's CSV (elapsed time in s)."""
    missing = set(_REQUIRED_METRIC_FIELDS) - row.keys()
    extra = row.keys() - set(METRIC_FIELDS)
    if missing or extra:
        raise ValueError(f"invalid metrics fields; missing={sorted(missing)}, extra={sorted(extra)}")
    iteration = row["iteration"]
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("metrics iteration must be a nonnegative integer")
    for key in METRIC_FIELDS[1:]:
        value = row.get(key)
        if value is None:
            if key in _REQUIRED_METRIC_FIELDS:
                raise ValueError(f"metrics {key} is required")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"metrics {key} must be a finite number")
    csv_path = Path(run_directory) / "metrics.csv"
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        if tuple(next(csv.reader(file), ())) != METRIC_FIELDS:
            raise ValueError("metrics.csv header differs from training schema")
    buffer = io.StringIO(newline="")
    csv.DictWriter(buffer, fieldnames=METRIC_FIELDS).writerow(row)
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        file.write(buffer.getvalue())
        file.flush()
        os.fsync(file.fileno())


def _iter_metric_rows(csv_path: Path):
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != METRIC_FIELDS:
            raise ValueError("metrics.csv header differs from training schema")
        previous_iteration = -1
        for row in reader:
            if set(row) != set(METRIC_FIELDS) or any(value is None for value in row.values()):
                raise ValueError("metrics.csv contains a malformed row")
            try:
                iteration = int(row["iteration"])
            except ValueError as exc:
                raise ValueError("metrics.csv contains an invalid iteration") from exc
            if iteration <= previous_iteration:
                raise ValueError("metrics.csv iterations must be strictly increasing")
            previous_iteration = iteration
            yield iteration, row


def reconcile_metrics(run_directory: Path, last_iteration: int) -> Path | None:
    """Archive and trim metrics newer than a resumed checkpoint.

    Returns the archive path when trimming was needed, or ``None`` when the
    existing CSV already ends at or before the checkpoint. Call this before
    resuming metric writes; the original CSV remains available for audit.
    """
    if (
        isinstance(last_iteration, bool)
        or not isinstance(last_iteration, int)
        or last_iteration < 0
    ):
        raise ValueError("last_iteration must be a nonnegative integer")
    run_directory = Path(run_directory)
    csv_path = run_directory / "metrics.csv"
    needs_trim = False
    for iteration, _ in _iter_metric_rows(csv_path):
        if iteration > last_iteration:
            needs_trim = True
    if not needs_trim:
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = run_directory / f"metrics_before_resume_{timestamp}_{uuid.uuid4().hex[:8]}.csv"
    with csv_path.open("rb") as source, archive.open("xb") as target:
        shutil.copyfileobj(source, target)
        target.flush()
        os.fsync(target.fileno())

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".metrics.", suffix=".tmp", dir=run_directory
    )
    try:
        with os.fdopen(file_descriptor, "w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=METRIC_FIELDS)
            writer.writeheader()
            for iteration, row in _iter_metric_rows(csv_path):
                if iteration <= last_iteration:
                    writer.writerow(row)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_name, csv_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return archive
