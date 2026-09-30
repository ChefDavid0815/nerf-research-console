"""Safe artifact and geometry readers for the local research console."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from nerf_step1.dataset import load_blender_scene
from nerf_step1.positional_encoding import SinusoidalPositionalEncoding
from nerf_step1.rays import generate_rays


_RUN_ID = re.compile(r"[A-Za-z0-9_-]+\Z")
_CHECKPOINT = re.compile(r"iter_(\d{6})\.pt\Z")
_EVALUATION_RENDER = re.compile(
    r"(train|val|test)_view_(\d{3})_(ground_truth|prediction|absolute_difference)\.png\Z"
)
_PREVIEW = re.compile(
    r"(ground_truth|prediction|absolute_difference|comparison)_iter_(\d{6})\.png\Z"
)
_SPLITS = ("train", "val", "test")
_IMAGE_KINDS = ("ground_truth", "prediction", "absolute_difference")


def resolve_run_directory(runs_root: Path, run_id: str) -> Path:
    """Resolve one run ID without allowing traversal or a symlink escape."""
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError("invalid run ID")
    root = Path(runs_root).resolve(strict=True)
    directory = (root / run_id).resolve(strict=True)
    if directory.parent != root or not directory.is_dir():
        raise ValueError("run ID is outside the runs directory")
    return directory


def _json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    if not path.resolve(strict=True).is_relative_to(path.parent.resolve(strict=True)):
        raise ValueError("JSON artifact path escapes its directory")
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return value


def _run_file(run_directory: Path, path: Path) -> str:
    root = Path(run_directory).resolve(strict=True)
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(root):
        raise ValueError("artifact path escapes the run directory")
    return resolved.relative_to(root).as_posix()


def _csv_rows(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    if not path.resolve(strict=True).is_relative_to(path.parent.resolve(strict=True)):
        raise ValueError("CSV artifact path escapes its directory")
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not set(fields).issubset(reader.fieldnames):
            raise ValueError(f"{path.name} is missing expected columns")
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path.name} contains an incomplete row")
            rows.append(row)
        return rows


def _integer(raw: Any, context: str) -> int:
    try:
        result = int(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{context} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{context} must be nonnegative")
    return result


def _float_or_none(raw: Any, context: str) -> float | None:
    if raw in (None, ""):
        return None
    try:
        result = float(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{context} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _checkpoints(run_directory: Path) -> list[dict[str, Any]]:
    directory = Path(run_directory) / "checkpoints"
    if not directory.is_dir():
        return []
    checkpoints = []
    for path in directory.iterdir():
        match = _CHECKPOINT.fullmatch(path.name)
        if match and path.is_file():
            checkpoints.append({
                "iteration": int(match.group(1)),
                "path": _run_file(run_directory, path),
                "size_bytes": path.stat().st_size,
            })
    return sorted(checkpoints, key=lambda row: row["iteration"])


def read_run_inventory(run_directory: Path) -> dict[str, Any]:
    """Inventory existing files; an absent category is reported as absent."""
    run_directory = Path(run_directory)
    if not run_directory.is_dir():
        raise FileNotFoundError(run_directory)
    checkpoints = _checkpoints(run_directory)
    preview_images = []
    for path in (run_directory / "renders").glob("*.png"):
        match = _PREVIEW.fullmatch(path.name)
        if match and path.is_file():
            preview_images.append({"iteration": int(match.group(2)), "kind": match.group(1),
                                   "path": _run_file(run_directory, path)})
    preview_images.sort(key=lambda row: (row["iteration"], row["kind"]))
    evaluation_renders = list_reconstructions(run_directory)
    files = {}
    for name in ("run_manifest.json", "config_snapshot.yaml", "environment_snapshot.json",
                 "status.json", "baseline_extension.json", "metrics.csv",
                 "evaluation_results.csv", "evaluation_summary.csv", "train.log"):
        path = run_directory / name
        files[name] = _run_file(run_directory, path) if path.is_file() else None
    return {
        "run_id": run_directory.name,
        "files": files,
        "checkpoints": checkpoints,
        "preview_images": preview_images,
        "reconstruction_count": len(evaluation_renders),
        "reconstruction_image_count": sum(len(row["images"]) for row in evaluation_renders),
        "hierarchical_sampling_numeric_available": False,
    }


def read_run_metadata(run_directory: Path) -> dict[str, Any]:
    """Reconcile legacy manifest fields in memory, preserving source labels.

    The original manifest and its configuration hash are never overwritten.
    A Step 4 staged continuation can reach 50k while its original frozen
    training_iterations remains 500; those are separate facts.
    """
    run_directory = Path(run_directory)
    manifest = _json_object(run_directory / "run_manifest.json")
    status = _json_object(run_directory / "status.json")
    extension = _json_object(run_directory / "baseline_extension.json")
    environment = _json_object(run_directory / "environment_snapshot.json")
    if manifest is None:
        return {"run_id": run_directory.name, "manifest": None,
                "status": status, "environment": environment,
                "migration": {"source": "unavailable", "fields": {}},
                "completed_target": False}
    settings = manifest.get("training_settings") or {}
    if not isinstance(settings, dict):
        raise ValueError("manifest training_settings must be an object")
    original_budget = _integer(settings.get("training_iterations", 0), "training_iterations")
    target = _integer(extension["latest_requested_target_iteration"], "target") if extension and "latest_requested_target_iteration" in extension else original_budget
    completed = _integer(status.get("last_completed_iteration", 0), "last_completed_iteration") if status else 0
    checkpoints = _checkpoints(run_directory)
    checkpoint_iterations = {row["iteration"] for row in checkpoints}
    snapshot_path = run_directory / "config_snapshot.yaml"
    config_digest_matches = None
    if snapshot_path.is_file() and isinstance(manifest.get("config_sha256"), str):
        _run_file(run_directory, snapshot_path)
        config_digest_matches = hashlib.sha256(snapshot_path.read_bytes()).hexdigest() == manifest["config_sha256"]
    return {
        "run_id": run_directory.name,
        "manifest": manifest,
        "status": status,
        "environment": environment,
        "migration": {
            "source": "in_memory_read_only",
            "fields": {
                "direction_encoding_L": {"value": settings.get("direction_encoding_L"), "source": "manifest.training_settings"},
                "original_training_iterations": {"value": original_budget, "source": "manifest.training_settings"},
                "requested_stop_iteration": {"value": target, "source": "baseline_extension" if extension else "manifest.training_settings"},
                "last_completed_iteration": {"value": completed if status else None, "source": "status.json" if status else "unavailable"},
                "positional_encoding_convention": {"value": "raw xyz; sin/cos(2^k pi xyz), k=0..L-1", "source": "validated_step2_code"},
                "density_initial_bias": {"value": settings.get("density_initial_bias"), "source": "manifest.training_settings"},
            },
        },
        "completed_target": bool(target > 0 and completed >= target and target in checkpoint_iterations),
        "config_digest_matches_manifest": config_digest_matches,
        "raw_status": status.get("status") if status else None,
    }


def read_evaluations(run_directory: Path) -> dict[str, Any]:
    """Return measured per-view and aggregate rows, with nullable metrics."""
    run_directory = Path(run_directory)
    per_view = []
    for row in _csv_rows(run_directory / "evaluation_results.csv",
                         ("iteration", "split", "view_index", "psnr", "ssim", "lpips")):
        if row["split"] not in _SPLITS:
            raise ValueError("invalid evaluation split")
        converted = {"iteration": _integer(row["iteration"], "iteration"),
                     "split": row["split"], "view_index": _integer(row["view_index"], "view_index")}
        for key in ("psnr", "ssim", "lpips", "render_seconds", "rays_per_second"):
            converted[key] = _float_or_none(row.get(key), key)
        for key in ("peak_vram_bytes", "width", "height", "render_chunk_size"):
            converted[key] = _integer(row[key], key) if row.get(key) not in (None, "") else None
        per_view.append(converted)
    summary = []
    for row in _csv_rows(run_directory / "evaluation_summary.csv",
                         ("iteration", "split", "view_count", "psnr_mean", "ssim_mean", "lpips_mean")):
        if row["split"] not in _SPLITS:
            raise ValueError("invalid evaluation split")
        converted = {"iteration": _integer(row["iteration"], "iteration"), "split": row["split"],
                     "view_count": _integer(row["view_count"], "view_count"),
                     "view_indices": [_integer(i, "view index") for i in row.get("view_indices", "").split(";") if i]}
        for key in ("psnr", "ssim", "lpips"):
            for suffix in ("mean", "median", "std"):
                name = f"{key}_{suffix}"
                converted[name] = _float_or_none(row.get(name), name)
        summary.append(converted)
    return {"per_view": per_view, "summary": summary,
            "per_view_available": bool(per_view), "summary_available": bool(summary)}


def list_reconstructions(run_directory: Path) -> list[dict[str, Any]]:
    """Group existing full-resolution PNGs and join only matching CSV metrics."""
    run_directory = Path(run_directory)
    measured = {(row["iteration"], row["split"], row["view_index"]): row
                for row in read_evaluations(run_directory)["per_view"]}
    groups: dict[tuple[int, str, int], dict[str, Any]] = {}
    render_root = run_directory / "evaluation" / "renders"
    if render_root.is_dir():
        for directory in render_root.iterdir():
            if not directory.is_dir() or not re.fullmatch(r"iter_\d{6}", directory.name):
                continue
            iteration = int(directory.name[5:])
            for path in directory.iterdir():
                match = _EVALUATION_RENDER.fullmatch(path.name)
                if match and path.is_file():
                    split, view_index, kind = match.groups()
                    key = (iteration, split, int(view_index))
                    row = groups.setdefault(key, {"iteration": iteration, "split": split,
                                                  "view_index": int(view_index), "images": {},
                                                  "metrics": measured.get(key)})
                    row["images"][kind] = _run_file(run_directory, path)
    return [groups[key] for key in sorted(groups)]


def camera_geometry(data_root: Path, scene: str, *, split: str | None = None,
                    start: int = 0, limit: int = 100,
                    pixel_xy: tuple[int, int] | None = None) -> dict[str, Any]:
    """Expose Step 1 validated c2w/intrinsics and its own ray convention."""
    if split is not None and split not in _SPLITS:
        raise ValueError("split must be train, val, or test")
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError("start must be a nonnegative integer")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
        raise ValueError("limit must be from 1 to 500")
    loaded = load_blender_scene(data_root, scene)
    all_views = []
    for name in _SPLITS if split is None else (split,):
        source = getattr(loaded, name)
        for index in range(len(source)):
            all_views.append((name, index, source))
    cameras = []
    for name, index, source in all_views[start:start + limit]:
        pose = source.camera_to_world[index]
        center_pixel = (source.intrinsics.width // 2, source.intrinsics.height // 2)
        origin, center_ray = generate_rays(pose, source.intrinsics, center_pixel)
        corners = ((0, 0), (source.intrinsics.width - 1, 0),
                   (source.intrinsics.width - 1, source.intrinsics.height - 1),
                   (0, source.intrinsics.height - 1))
        _, corner_rays = generate_rays(pose, source.intrinsics, corners)
        selected_ray = None
        if pixel_xy is not None:
            ray_origin, ray_direction = generate_rays(pose, source.intrinsics, pixel_xy)
            selected_ray = {"pixel_xy": list(pixel_xy), "origin": ray_origin.tolist(),
                            "direction": ray_direction.tolist()}
        cameras.append({
            "split": name, "index": index, "camera_to_world": pose.tolist(),
            "position": origin.tolist(), "forward": (-pose[:3, 2]).tolist(),
            "right": pose[:3, 0].tolist(), "up": pose[:3, 1].tolist(),
            "center_ray": center_ray.tolist(), "corner_rays": corner_rays.tolist(),
            "selected_ray": selected_ray,
        })
    return {"scene": loaded.name, "coordinate_system": "OpenGL c2w; +X right, +Y up, forward -Z",
            "ray_convention": "integer pixel indices; unnormalized world directions",
            "intrinsics": asdict(loaded.train.intrinsics), "total": len(all_views),
            "start": start, "cameras": cameras}


def positional_encoding_info(L: int) -> dict[str, Any]:
    """Describe exactly the Step 2 encoding used by the model."""
    if isinstance(L, bool) or not isinstance(L, int) or not 0 <= L <= 15:
        raise ValueError("L must be an integer from 0 to 15")
    encoder = SinusoidalPositionalEncoding(input_dim=3, num_frequencies=L, include_input=True)
    x = torch.linspace(0.0, 1.0, 180, dtype=torch.float64)
    positions = torch.stack((x, torch.zeros_like(x), torch.zeros_like(x)), dim=-1)
    with torch.no_grad():
        encoded = encoder(positions)
    # The validated Step 2 encoder orders each band as sin(xyz), cos(xyz).
    # Read its actual output rather than reimplementing the scientific formula.
    curves = [
        {"k": k, "frequency_multiplier": 2 ** k,
         "sin": encoded[:, 3 + 6 * k].tolist(),
         "cos": encoded[:, 3 + 6 * k + 3].tolist()}
        for k in range(L)
    ]
    return {"L": L, "input_dim": encoder.input_dim, "encoded_dim": encoder.output_dim,
            "include_input": encoder.include_input, "highest_frequency_band": 2 ** (L - 1) if L else None,
            "x": x.tolist(), "curves": curves,
            "bands": [{"k": k, "frequency_multiplier": 2 ** k,
                       "phase_formula": f"2^{k} * pi * x"} for k in range(L)],
            "feature_order": "raw xyz, then per-band sin(xyz), cos(xyz)",
            "formula": "sin(2^k*pi*x), cos(2^k*pi*x)"}


def hierarchical_sampling_artifacts(run_directory: Path,
                                    project_root: Path | None = None) -> dict[str, Any]:
    """Report available static validation figures without claiming ray traces."""
    images = []
    if project_root is not None:
        root = Path(project_root).resolve(strict=True)
        for name in ("hierarchical_sampling_visualization.png",
                     "hierarchical_sampling_after_training.png"):
            path = root / "artifacts" / name
            if path.is_file() and path.resolve().is_relative_to(root):
                images.append({"path": path.relative_to(root).as_posix(),
                               "source": "project_validation_figure",
                               "measured_ray_trace": False})
    return {"run_id": Path(run_directory).name, "available": False,
            "reason": "No persisted per-ray coarse/fine depths, density, alpha, weights or transmittance were found.",
            "coarse_depths": None, "fine_depths": None, "density": None,
            "alpha": None, "weights": None, "transmittance": None,
            "static_figures": images}


def _baseline_migration_payload(run_directory: Path) -> dict[str, Any]:
    """Build a source-labelled migration record only for the validated 50k run."""
    metadata = read_run_metadata(run_directory)
    manifest = metadata["manifest"]
    status = metadata["status"]
    if not isinstance(manifest, dict) or not isinstance(status, dict):
        raise ValueError("baseline migration requires an original manifest and status")
    if metadata["config_digest_matches_manifest"] is not True:
        raise ValueError("baseline migration requires an intact frozen config snapshot")
    settings = manifest["training_settings"]
    frozen_settings = yaml.safe_load((run_directory / "config_snapshot.yaml").read_text(encoding="utf-8"))
    if frozen_settings != settings:
        raise ValueError("manifest training settings differ from the frozen config snapshot")
    fields = metadata["migration"]["fields"]
    if (manifest.get("scene") != "lego" or settings.get("position_encoding_L") != 10
            or settings.get("direction_encoding_L") != 4
            or settings.get("random_seed") != 0
            or settings.get("batch_size") != 256
            or fields["original_training_iterations"]["value"] != 500
            or fields["requested_stop_iteration"]["value"] != 50000
            or fields["last_completed_iteration"]["value"] != 50000
            or not metadata["completed_target"]):
        raise ValueError("run does not match the validated 50k Lego baseline")

    source_names = (
        "run_manifest.json", "config_snapshot.yaml", "environment_snapshot.json",
        "status.json", "baseline_extension.json", "metrics.csv",
        "evaluation_results.csv", "evaluation_summary.csv",
    )
    sources: dict[str, dict[str, str]] = {}
    for name in source_names:
        path = run_directory / name
        relative = _run_file(run_directory, path)
        sources[name] = {"path": relative,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    inventory = read_run_inventory(run_directory)
    evaluations = read_evaluations(run_directory)
    metric_rows = _csv_rows(run_directory / "metrics.csv", ("iteration",))
    if not metric_rows or _integer(metric_rows[-1]["iteration"], "last metric iteration") != 50000:
        raise ValueError("baseline training metrics do not reach iteration 50000")

    def sourced(value: Any, source: str) -> dict[str, Any]:
        return {"value": value, "source": source}

    return {
        "schema_version": 1,
        "record_kind": "console_import_migration",
        "run_id": run_directory.name,
        "scientific_artifacts_modified": False,
        "original_run_manifest_preserved": True,
        "source_files": sources,
        "config_digest_matches_original_manifest": True,
        "fields": {
            "scene": sourced(manifest["scene"], "run_manifest.json.scene"),
            "position_encoding_L": sourced(settings["position_encoding_L"], "run_manifest.json.training_settings"),
            "direction_encoding_L": sourced(settings["direction_encoding_L"], "run_manifest.json.training_settings"),
            "seed": sourced(settings["random_seed"], "run_manifest.json.training_settings"),
            "batch_size": sourced(settings["batch_size"], "run_manifest.json.training_settings"),
            "original_frozen_training_budget": sourced(500, "config_snapshot.yaml and run_manifest.json.training_settings"),
            "staged_target_iteration": sourced(50000, "baseline_extension.json.latest_requested_target_iteration"),
            "last_completed_iteration": sourced(50000, "status.json.last_completed_iteration"),
            "raw_status": sourced(status["status"], "status.json.status"),
            "target_reached_with_checkpoint": sourced(True, "status.json plus checkpoints/iter_050000.pt existence"),
            "training_metric_rows": sourced(len(metric_rows), "metrics.csv"),
            "per_view_evaluation_rows": sourced(len(evaluations["per_view"]), "evaluation_results.csv"),
            "evaluation_summary_rows": sourced(len(evaluations["summary"]), "evaluation_summary.csv"),
            "checkpoint_count": sourced(len(inventory["checkpoints"]), "checkpoints/"),
            "preview_image_count": sourced(len(inventory["preview_images"]), "renders/"),
            "full_resolution_reconstruction_images": sourced(inventory["reconstruction_image_count"], "evaluation/renders/"),
            "positional_encoding_convention": sourced(
                "raw xyz; sin/cos(2^k pi xyz), k=0..L-1, include_input=true",
                "src/nerf_step1/positional_encoding.py"),
            "camera_convention": sourced(
                "OpenGL camera-to-world; +X right, +Y up, forward -Z; integer pixel grid; unnormalized rays",
                "src/nerf_step1/rays.py"),
        },
        "unavailable": {
            "hierarchical_sampling_per_ray_values": {
                "available": False,
                "fields": ["coarse_depths", "fine_depths", "density", "alpha", "weights", "transmittance"],
                "reason": "No persisted numeric per-ray trace exists in the baseline run.",
            },
            "live_gpu_telemetry_during_completed_run": {
                "available": False,
                "reason": "Historical GPU utilization, temperature and power were not persisted per iteration.",
            },
        },
        "metric_semantics": {
            "metrics.csv.psnr": "training batch PSNR; not validation PSNR",
            "evaluation_results.csv": "measured full-image per-view PSNR, SSIM and LPIPS",
            "preview_pngs": "visualization only; not metric inputs",
        },
    }


def write_baseline_migration_manifest(run_directory: Path) -> dict[str, Any]:
    """Create an independent baseline sidecar once; validate on every retry.

    Existing sidecars are never replaced. Source fingerprints make a rerun
    reject a stale sidecar if an underlying artifact has changed.
    """
    run_directory = Path(run_directory).resolve(strict=True)
    if not run_directory.is_dir():
        raise ValueError("run_directory must be a directory")
    expected = _baseline_migration_payload(run_directory)
    sidecar = run_directory / "console_migration_manifest.json"

    def validate_existing() -> dict[str, Any]:
        existing = _json_object(sidecar)
        if existing is None:
            raise FileNotFoundError(sidecar)
        imported_at = existing.get("imported_at_utc")
        if not isinstance(imported_at, str):
            raise ValueError("existing migration manifest has no import timestamp")
        try:
            datetime.fromisoformat(imported_at)
        except ValueError as exc:
            raise ValueError("existing migration manifest has an invalid import timestamp") from exc
        without_timestamp = {key: value for key, value in existing.items() if key != "imported_at_utc"}
        if without_timestamp != expected:
            raise ValueError("existing migration manifest differs from current baseline evidence")
        return existing

    if sidecar.exists():
        return validate_existing()
    payload = dict(expected)
    payload["imported_at_utc"] = datetime.now(timezone.utc).isoformat()
    descriptor, temporary_name = tempfile.mkstemp(prefix=".console_migration_", suffix=".json", dir=run_directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_name, sidecar)
        except FileExistsError:
            return validate_existing()
        return payload
    finally:
        Path(temporary_name).unlink(missing_ok=True)
