"""Train or resume the Vanilla NeRF pipeline on official Blender Lego.

The default config is the Step 3 engineering smoke run. ``--extend-to``
continues its frozen checkpoint/config beyond the original 500-step smoke
budget for Step 4; it does not start an EE bandwidth sweep. Preview images
are diagnostic only; use the separate full-view evaluator for research metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nerf_step1.dataset import BlenderSplit, load_blender_scene  # noqa: E402
from nerf_step1.pipeline import CoarseFineNeRF, make_scheduler, train_step, validate_training_config  # noqa: E402
from nerf_step1.preview import aligned_ground_truth, render_image, save_preview  # noqa: E402
from nerf_step1.rays import generate_rays  # noqa: E402
from nerf_step1.run_logging import write_status  # noqa: E402
from nerf_step1.training_io import (  # noqa: E402
    append_metrics, assert_run_config_compatible, create_training_run,
    load_checkpoint, reconcile_metrics, save_checkpoint,
)


def read_training_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    validate_training_config(config)
    return config


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def sample_training_batch(
    split: BlenderSplit,
    batch_size: int,
    device: torch.device,
    *,
    normalize_ray_directions: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose one train image and uniformly random pixels within that image."""
    image_index = int(torch.randint(len(split), (1,)).item())
    view = split.get_view(image_index)
    x = torch.randint(view.intrinsics.width, (batch_size,))
    y = torch.randint(view.intrinsics.height, (batch_size,))
    pixels = torch.stack((x, y), dim=-1)
    target = view.image[y, x].to(device=device)
    origins, directions = generate_rays(
        view.camera_to_world, view.intrinsics, pixels,
        device=device, dtype=torch.float32,
        normalize_directions=normalize_ray_directions,
    )
    return origins, directions, target


def _last_elapsed(run_directory: Path) -> float:
    with (run_directory / "metrics.csv").open("r", newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        last = None
        for last in rows:
            pass
    return float(last["elapsed_time"]) if last is not None else 0.0


def parse_stage_iterations(raw: str | None) -> tuple[int, ...]:
    """Parse explicit absolute iteration milestones without mutating the run config."""
    if raw is None:
        return ()
    try:
        iterations = tuple(int(part.strip()) for part in raw.split(","))
    except ValueError as exc:
        raise ValueError("--stage-checkpoints must be comma-separated positive integers") from exc
    if not iterations or any(value <= 0 for value in iterations):
        raise ValueError("--stage-checkpoints must be comma-separated positive integers")
    if tuple(sorted(set(iterations))) != iterations:
        raise ValueError("--stage-checkpoints must be strictly increasing without duplicates")
    return iterations


def output_due(iteration: int, stop_iteration: int, interval: int,
               stage_iterations: Sequence[int] | None) -> bool:
    """Always persist the stage boundary; explicit milestones replace cadence."""
    return (iteration == stop_iteration or
            (iteration in stage_iterations if stage_iterations is not None else
             iteration % interval == 0))


def _write_extension_record(run_directory: Path, *, original_budget: int,
                            source_iteration: int, target_iteration: int,
                            stage_iterations: Sequence[int]) -> None:
    """Keep the changed *observation budget* separate from the frozen config.

    This is intentionally not a new training config: model, batches, optimizer,
    scheduler and sampling retain their checkpointed Step 3 definitions.
    """
    path = run_directory / "baseline_extension.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("original_training_iterations") != original_budget:
            raise ValueError("extension record disagrees with frozen training budget")
        if record.get("source_checkpoint_iteration") > source_iteration:
            raise ValueError("extension record source is newer than the resume checkpoint")
        previous_stages = record.get("stage_iterations")
        if not isinstance(previous_stages, list) or list(stage_iterations[:len(previous_stages)]) != previous_stages:
            raise ValueError("stage checkpoint schedule differs from existing extension record")
        # Convergence is assessed from observed data, so future stages may be
        # appended later. Already scheduled milestones cannot be removed.
        record["stage_iterations"] = list(stage_iterations)
        record["latest_requested_target_iteration"] = target_iteration
    else:
        record = {
            "purpose": "Step 4 Lego baseline convergence; no L sweep",
            "source_checkpoint_iteration": source_iteration,
            "original_training_iterations": original_budget,
            "stage_iterations": list(stage_iterations),
            "latest_requested_target_iteration": target_iteration,
            "training_config_unchanged": True,
            "changed_run_control": "absolute stop iteration and checkpoint/preview cadence only",
        }
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_checkpoint_telemetry(run_directory: Path, *, iteration: int,
                                row: dict, checkpoint: Path, preview: Path,
                                device: torch.device) -> Path:
    """Capture the training scalar and stage-local CUDA peak beside a milestone."""
    record = {
        "iteration": iteration,
        "training_total_loss": row["total_loss"],
        "training_coarse_loss": row["coarse_loss"],
        "training_fine_loss": row["fine_loss"],
        "training_batch_psnr_db": row["psnr"],
        "elapsed_time_seconds": row["elapsed_time"],
        "gpu_memory_allocated_bytes": row["gpu_memory_allocated_bytes"],
        "gpu_memory_reserved_bytes": row["gpu_memory_reserved_bytes"],
        "gpu_peak_allocated_since_stage_start_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "gpu_peak_reserved_since_stage_start_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        ),
        "checkpoint": str(checkpoint.resolve()),
        "diagnostic_preview": str(preview.resolve()),
        "validation_metrics": "See full-image evaluation_results.csv; training batch PSNR is not validation PSNR.",
    }
    path = run_directory / "checkpoints" / f"iter_{iteration:06d}_telemetry.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _save_checkpoint(run_directory: Path, iteration: int, model: CoarseFineNeRF,
                     optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler,
                     config: dict) -> Path:
    path = run_directory / "checkpoints" / f"iter_{iteration:06d}.pt"
    save_checkpoint(
        path, coarse_model=model.coarse_model, fine_model=model.fine_model,
        optimizer=optimizer, scheduler=scheduler, iteration=iteration, config=config,
    )
    return path


def _render_preview(run_directory: Path, iteration: int, model: CoarseFineNeRF,
                    validation_split: BlenderSplit, config: dict) -> dict[str, Path]:
    view = validation_split.get_view(0)
    resolution = (config["preview_width"], config["preview_height"])
    prediction = render_image(
        model, view.camera_to_world, view.intrinsics, resolution,
        render_chunk_size=config["render_chunk_size"],
    )
    ground_truth = aligned_ground_truth(view.image, view.intrinsics, resolution)
    return save_preview(run_directory, iteration, prediction, ground_truth)


def train(args: argparse.Namespace) -> Path:
    stage_iterations = parse_stage_iterations(args.stage_checkpoints)
    if args.extend_to is not None and args.resume is None:
        raise ValueError("--extend-to requires --resume; a new run is not a continuation")
    if stage_iterations and args.extend_to is None:
        raise ValueError("--stage-checkpoints requires --extend-to")
    if args.extend_to is not None and args.stop_after is not None:
        raise ValueError("use either --extend-to or --stop-after")
    if args.resume is None:
        config_path = args.config or PROJECT_ROOT / "configs" / "smoke.yaml"
        config = read_training_config(config_path)
        seed_all(config["random_seed"])
        supplied_run = getattr(args, "run_directory", None)
        if supplied_run is None:
            run_directory = create_training_run(PROJECT_ROOT / "runs", args.label, config)
        else:
            run_directory = Path(supplied_run).resolve()
            assert_run_config_compatible(run_directory, config)
        first_iteration = 0
    else:
        run_directory = args.resume.resolve()
        snapshot_path = run_directory / "config_snapshot.yaml"
        config = read_training_config(snapshot_path)
        if args.config is not None and read_training_config(args.config) != config:
            raise ValueError("requested config differs from the saved run configuration")
        assert_run_config_compatible(run_directory, config)
        seed_all(config["random_seed"])  # Checkpoint later restores exact RNG state.
        first_iteration = -1

    run_log = run_directory / "train.log"
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_log, encoding="utf-8")],
        force=True,
    )
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        logging.info("GPU: %s; total VRAM %.2f GiB", torch.cuda.get_device_name(device),
                     torch.cuda.get_device_properties(device).total_memory / 2**30)
    else:
        logging.info("Device: CPU")

    data_root = Path(config["dataset_root"])
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    scene = load_blender_scene(data_root, config["scene"], background_color=(1.0, 1.0, 1.0))
    model = CoarseFineNeRF(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    scheduler = make_scheduler(optimizer, config)
    if first_iteration == -1:
        checkpoints = sorted((run_directory / "checkpoints").glob("iter_*.pt"))
        if not checkpoints:
            raise FileNotFoundError("resume run has no checkpoint")
        checkpoint_path = checkpoints[-1]
        first_iteration = load_checkpoint(
            checkpoint_path, coarse_model=model.coarse_model,
            fine_model=model.fine_model, optimizer=optimizer, scheduler=scheduler,
            expected_config=config, map_location=device,
        )
        reconcile_metrics(run_directory, first_iteration)
        logging.info("Resumed %s from completed iteration %d", checkpoint_path, first_iteration)
    else:
        preview_paths = _render_preview(run_directory, 0, model, scene.val, config)
        logging.info("Initial validation preview: %s", preview_paths["contact_sheet"])

    training_iterations = config["training_iterations"]
    stop_iteration = (args.extend_to if args.extend_to is not None else
                      min(training_iterations, args.stop_after or training_iterations))
    if stop_iteration <= first_iteration:
        raise ValueError("stop iteration must be greater than checkpoint iteration")
    if args.extend_to is not None:
        if stop_iteration <= training_iterations:
            raise ValueError("--extend-to must exceed the frozen original training budget")
        _write_extension_record(
            run_directory, original_budget=training_iterations,
            source_iteration=first_iteration, target_iteration=stop_iteration,
            stage_iterations=stage_iterations,
        )
        logging.info(
            "Step 4 baseline continuation to %d; frozen Step 3 budget %d; "
            "batch %d and scientific config unchanged; output milestones %s",
            stop_iteration, training_iterations, config["batch_size"],
            list(stage_iterations) if stage_iterations else "stage boundary only",
        )
    elapsed_before = _last_elapsed(run_directory)
    session_start = time.perf_counter()
    last_checkpoint = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for iteration in range(first_iteration + 1, stop_iteration + 1):
            step_start = time.perf_counter()
            origins, directions, target = sample_training_batch(
                scene.train, config["batch_size"], device,
                normalize_ray_directions=config["normalize_ray_directions"],
            )
            metrics = train_step(model, optimizer, origins, directions, target, scheduler=scheduler)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_seconds = time.perf_counter() - step_start
            elapsed = elapsed_before + time.perf_counter() - session_start
            row = {
                "iteration": iteration,
                "total_loss": metrics["total_loss"],
                "coarse_loss": metrics["coarse_loss"],
                "fine_loss": metrics["fine_loss"],
                "psnr": metrics["PSNR"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_time": elapsed,
                "rays_per_second": config["batch_size"] / step_seconds,
                "gpu_memory_allocated_bytes": (
                    torch.cuda.memory_allocated(device) if device.type == "cuda" else None
                ),
                "gpu_memory_reserved_bytes": (
                    torch.cuda.memory_reserved(device) if device.type == "cuda" else None
                ),
            }
            append_metrics(run_directory, row)
            stop_file = getattr(args, "stop_file", None)
            stop_requested = stop_file is not None and Path(stop_file).is_file()
            if iteration % config["progress_interval"] == 0 or iteration == first_iteration + 1:
                eta = (stop_iteration - iteration) * (
                    (time.perf_counter() - session_start) / (iteration - first_iteration)
                )
                vram = (f"{row['gpu_memory_allocated_bytes'] / 2**30:.2f} GiB" if
                        device.type == "cuda" else "n/a")
                logging.info(
                    "iter %d/%d loss %.6f fine PSNR %.2f dB LR %.3g VRAM %s elapsed %.1fs ETA %.1fs",
                    iteration, stop_iteration, metrics["total_loss"], metrics["PSNR"],
                    row["learning_rate"], vram, elapsed, eta,
                )
            checkpoint_due = output_due(
                iteration, stop_iteration, config["checkpoint_interval"],
                stage_iterations if args.extend_to is not None else None,
            )
            preview_due = output_due(
                iteration, stop_iteration, config["preview_interval"],
                stage_iterations if args.extend_to is not None else None,
            )
            if checkpoint_due or stop_requested:
                last_checkpoint = _save_checkpoint(
                    run_directory, iteration, model, optimizer, scheduler, config
                )
                logging.info("Saved checkpoint: %s", last_checkpoint)
            if preview_due and not stop_requested:
                paths = _render_preview(run_directory, iteration, model, scene.val, config)
                logging.info("Validation preview: %s", paths["contact_sheet"])
            if args.extend_to is not None and checkpoint_due and preview_due and not stop_requested:
                telemetry = _write_checkpoint_telemetry(
                    run_directory, iteration=iteration, row=row,
                    checkpoint=last_checkpoint, preview=paths["contact_sheet"], device=device,
                )
                logging.info("Checkpoint telemetry: %s", telemetry)
            if stop_requested:
                write_status(run_directory, {
                    "status": "stopped", "termination_reason": "user_requested",
                    "last_completed_iteration": iteration,
                    "checkpoint": str(last_checkpoint),
                    "config": str(run_directory / "config_snapshot.yaml"),
                    "metrics": str(run_directory / "metrics.csv"),
                })
                logging.info("Stopped after iteration %d and saved %s", iteration, last_checkpoint)
                return run_directory
        write_status(run_directory, {
            "status": "completed" if args.extend_to is None and stop_iteration == training_iterations
                      else "paused_after_checkpoint",
            "last_completed_iteration": stop_iteration,
            "original_training_iterations": training_iterations,
            "requested_stop_iteration": stop_iteration,
            "checkpoint": str(last_checkpoint),
            "config": str(run_directory / "config_snapshot.yaml"),
            "metrics": str(run_directory / "metrics.csv"),
        })
        return run_directory
    except Exception as exc:
        logging.exception("Training failed")
        write_status(run_directory, {
            "status": "failed", "last_completed_iteration": iteration - 1,
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None, help="run directory to resume")
    parser.add_argument("--stop-after", type=int, default=None,
                        help="pause after this completed iteration, preserving configured total")
    parser.add_argument("--extend-to", type=int, default=None,
                        help="Step 4 only: resume frozen run to an absolute iteration past its original budget")
    parser.add_argument("--stage-checkpoints", default=None,
                        help="comma-separated absolute Step 4 checkpoint/preview iterations; "
                             "the requested stop is always saved")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--label", default="step3_smoke")
    parser.add_argument("--run-directory", type=Path, default=None,
                        help="use an already frozen compatible run created by the console")
    parser.add_argument("--stop-file", type=Path, default=None,
                        help="gracefully checkpoint and stop after the current iteration when this file appears")
    args = parser.parse_args()
    if args.stop_after is not None and args.stop_after <= 0:
        parser.error("--stop-after must be positive")
    if args.extend_to is not None and args.extend_to <= 0:
        parser.error("--extend-to must be positive")
    run_directory = train(args)
    print(json.dumps({"run_directory": str(run_directory.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
