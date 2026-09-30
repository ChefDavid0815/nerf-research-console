"""Read-only, deterministic single-ray inspection of a frozen checkpoint.

All camera, sampling, rendering and model operations call the verified Step 1–4
Python implementation. No artifact, metric, checkpoint or run status is written.
"""

from __future__ import annotations

import hashlib
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from nerf_step1.dataset import load_blender_scene
from nerf_step1.pipeline import CoarseFineNeRF, make_scheduler
from nerf_step1.rays import generate_rays
from nerf_step1.training_io import assert_run_config_compatible, load_checkpoint
from nerf_step1.evaluation_io import file_sha256


def _values(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().cpu().reshape(-1).tolist()


def inspect_ray(project_root: Path, run_directory: Path, *, checkpoint: int,
                split: str, view_index: int, x: int, y: int) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    run_directory = Path(run_directory).resolve()
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val or test")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in (checkpoint, view_index, x, y)):
        raise ValueError("checkpoint, view_index, x and y must be nonnegative integers")

    config_path = run_directory / "config_snapshot.yaml"
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    if not isinstance(config, dict):
        raise ValueError("frozen run config is invalid")
    try:
        assert_run_config_compatible(run_directory, config)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"frozen run config integrity check failed: {exc}") from exc
    checkpoint_path = run_directory / "checkpoints" / f"iter_{checkpoint:06d}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint {checkpoint} is unavailable")
    checkpoint_digest_before = file_sha256(checkpoint_path)

    dataset_root = Path(config["dataset_root"])
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    scene = load_blender_scene(dataset_root, str(config["scene"]))
    selected_split = getattr(scene, split)
    if view_index >= len(selected_split):
        raise ValueError(f"{split} view_index must be below {len(selected_split)}")
    intrinsics = selected_split.intrinsics
    if x >= intrinsics.width or y >= intrinsics.height:
        raise ValueError(f"pixel must be within 0..{intrinsics.width - 1} × 0..{intrinsics.height - 1}")

    # Match the evaluator's CPU-safe checkpoint restore while keeping the API
    # process RNG untouched. This validates config, architecture, weights and
    # optimizer/scheduler presence before any model inference.
    model = CoarseFineNeRF(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    scheduler = make_scheduler(optimizer, config)
    try:
        loaded_iteration = load_checkpoint(
            checkpoint_path, coarse_model=model.coarse_model,
            fine_model=model.fine_model, optimizer=optimizer,
            scheduler=scheduler, expected_config=config,
            map_location="cpu", restore_rng=False,
        )
    except (OSError, RuntimeError, ValueError, TypeError, KeyError,
            IndexError, EOFError, pickle.UnpicklingError) as exc:
        raise RuntimeError(f"checkpoint cannot be inspected: {exc}") from exc
    if loaded_iteration != checkpoint:
        raise ValueError("checkpoint filename and payload iteration differ")
    del optimizer, scheduler
    model.eval()

    origins, directions = generate_rays(
        selected_split.camera_to_world[view_index], intrinsics, [(x, y)],
        device="cpu", dtype=torch.float32,
        normalize_directions=bool(config["normalize_ray_directions"]),
    )
    with torch.inference_mode():
        result = model(origins, directions, randomized=False)
        coarse_points = origins[:, None, :] + result.coarse_depths[..., None] * directions[:, None, :]
        fine_points = origins[:, None, :] + result.combined_depths[..., None] * directions[:, None, :]
        _, coarse_density = model._query(model.coarse_model, coarse_points, directions)
        _, fine_density = model._query(model.fine_model, fine_points, directions)

    checkpoint_digest_after = file_sha256(checkpoint_path)
    if checkpoint_digest_before != checkpoint_digest_after:
        raise RuntimeError("checkpoint changed during ray inspection; discard result")

    def stage(depths: torch.Tensor, density: torch.Tensor, render) -> dict[str, Any]:
        return {
            "depths": _values(depths),
            "density": _values(density),
            "alpha": _values(render.alpha),
            "weights": _values(render.weights),
            "transmittance": _values(render.transmittance),
            "rgb": _values(render.rgb_map),
            "depth_map": float(render.depth_map.item()),
            "accumulated_opacity": float(render.accumulated_opacity.item()),
        }

    coarse = stage(result.coarse_depths, coarse_density, result.coarse)
    fine = stage(result.combined_depths, fine_density, result.fine)
    return {
        "available": True,
        "run_id": run_directory.name,
        "checkpoint": checkpoint,
        "split": split,
        "view_index": view_index,
        "pixel": {"x": x, "y": y},
        "image_size": {"width": intrinsics.width, "height": intrinsics.height},
        "ray": {"origin": _values(origins), "direction": _values(directions),
                "direction_normalized": bool(config["normalize_ray_directions"])},
        "coarse_depths": coarse["depths"],
        "fine_depths": _values(result.fine_depths),
        "combined_depths": fine["depths"],
        "coarse": coarse,
        "fine": {**fine, "sampled_depths": _values(result.fine_depths)},
        "source": {
            "kind": "on_demand_deterministic_checkpoint_ray",
            "calculated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_snapshot_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "checkpoint_sha256": checkpoint_digest_after,
            "dataset_root": str(dataset_root),
            "camera_metadata": str(dataset_root / scene.name / f"transforms_{split}.json"),
            "checkpoint_path": str(checkpoint_path),
            "engine": "nerf_step1.pipeline.CoarseFineNeRF.forward(randomized=False)",
            "density_query": "nerf_step1.pipeline.CoarseFineNeRF._query",
            "depth_convention": "ray parameter t in origin + t * direction; not physical world distance",
            "fine_density_alignment": "fine network densities align with combined_depths (coarse + newly sampled fine)",
            "device": "cpu",
            "stored_artifact": False,
        },
    }
