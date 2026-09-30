"""Evaluate a frozen Step 3/4 checkpoint on selected full Blender views."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nerf_step1.dataset import load_blender_scene  # noqa: E402
from nerf_step1.evaluation import (  # noqa: E402
    ImageMetricSuite, LPIPSMetric, evaluate_model, export_evaluation_csv, export_summary_csv,
)
from nerf_step1.evaluation_io import (  # noqa: E402
    evaluation_write_lock, file_sha256, verify_checkpoint_provenance,
)
from nerf_step1.pipeline import CoarseFineNeRF, make_scheduler  # noqa: E402
from nerf_step1.training_io import load_checkpoint  # noqa: E402
from step4_artifacts import save_evaluation_view  # noqa: E402


def _scientific_metadata(metadata: dict) -> dict:
    """Compare all recorded scientific choices, excluding only local operation details."""
    science = copy.deepcopy(metadata)
    science.pop("run_directory", None)
    science.pop("performance_peak_vram_scope", None)
    lpips_settings = science.get("lpips")
    if not isinstance(lpips_settings, dict):
        raise ValueError("metric metadata is missing LPIPS settings")
    for key in ("device", "calibrated_linear_weights_path", "imagenet_backbone_weights_path"):
        lpips_settings.pop(key, None)
    ssim_settings = science.get("ssim")
    if isinstance(ssim_settings, dict) and ssim_settings.get("package_version") == "0.26.0":
        # Earlier Step 4 invocations used these documented skimage defaults.
        ssim_settings.setdefault("K1", 0.01)
        ssim_settings.setdefault("K2", 0.03)
    return science


def _ensure_metric_metadata(path: Path, metadata: dict, per_view_csv_path: Path) -> None:
    """Pin metric provenance before any new images or CSV rows are produced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read existing metric metadata: {path}") from exc
        if not isinstance(existing, dict) or _scientific_metadata(existing) != _scientific_metadata(metadata):
            raise ValueError("scientific metric metadata differs from existing evaluation; refusing to append rows")
        ssim_settings = existing.get("ssim")
        if isinstance(ssim_settings, dict) and ("K1" not in ssim_settings or "K2" not in ssim_settings):
            ssim_settings["K1"] = 0.01
            ssim_settings["K2"] = 0.03
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".metric_metadata_", suffix=".json", dir=path.parent
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    json.dump(existing, stream, indent=2, ensure_ascii=False)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_name, path)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
        return
    if per_view_csv_path.exists():
        raise ValueError("evaluation CSV exists without metric metadata; refusing unprovenanced append")
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        # Another evaluation process may have created it after our first check.
        _ensure_metric_metadata(path, metadata, per_view_csv_path)


def _checkpoint_path(run_directory: Path, checkpoint: int) -> Path:
    if checkpoint < 0:
        raise ValueError("checkpoint iteration must be nonnegative")
    path = run_directory / "checkpoints" / f"iter_{checkpoint:06d}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    return path


def evaluate_checkpoint(
    run_directory: Path, checkpoint: int, split: str, view_indices: list[int],
    *, device: str, render_chunk_size: int | None = None,
) -> None:
    run_directory = Path(run_directory).resolve()
    config_path = run_directory / "config_snapshot.yaml"
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    if not isinstance(config, dict):
        raise ValueError("run config snapshot must be a mapping")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    chunk_size = render_chunk_size if render_chunk_size is not None else int(config["render_chunk_size"])
    model = CoarseFineNeRF(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    scheduler = make_scheduler(optimizer, config)
    checkpoint_path = _checkpoint_path(run_directory, checkpoint)
    loaded_checkpoint_digest = file_sha256(checkpoint_path)
    actual_iteration = load_checkpoint(
        checkpoint_path,
        coarse_model=model.coarse_model, fine_model=model.fine_model,
        optimizer=optimizer, scheduler=scheduler, expected_config=config,
        restore_rng=False,
    )
    if actual_iteration != checkpoint:
        raise ValueError(f"checkpoint filename says {checkpoint} but contains {actual_iteration}")
    del optimizer, scheduler
    model.to(target_device).eval()
    dataset_root = Path(config["dataset_root"])
    if not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root
    scene = load_blender_scene(dataset_root, str(config["scene"]))
    metrics = ImageMetricSuite(LPIPSMetric(target_device))
    evaluation_directory = run_directory / "evaluation"
    metadata = metrics.metadata()
    metadata.update({
        "run_directory": str(run_directory),
        "config_snapshot_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "scene": scene.name,
        "renderer": "nerf_step1.preview.render_image, deterministic fine RGB, original integer pixel grid",
        "metric_input": "unquantized float image; visualization PNGs are not metric inputs",
    })
    metadata_path = evaluation_directory / "metric_metadata.json"
    per_view_csv_path = run_directory / "evaluation_results.csv"
    with evaluation_write_lock(run_directory):
        _ensure_metric_metadata(metadata_path, metadata, per_view_csv_path)
        verify_checkpoint_provenance(run_directory, actual_iteration, loaded_checkpoint_digest)

    rendered_views = []
    def save_view(view, prediction, row):
        rendered_views.append((view, prediction, row))

    result = evaluate_model(
        model, scene, split, view_indices, metrics=metrics,
        render_chunk_size=chunk_size, iteration=actual_iteration,
        on_view=save_view,
    )
    with evaluation_write_lock(run_directory):
        _ensure_metric_metadata(metadata_path, metadata, per_view_csv_path)
        verify_checkpoint_provenance(run_directory, actual_iteration, loaded_checkpoint_digest)
        for view, prediction, row in rendered_views:
            save_evaluation_view(
                run_directory, actual_iteration, split, row.view_index, prediction, view.image
            )
        export_evaluation_csv(per_view_csv_path, result)
        export_summary_csv(
            run_directory / "evaluation_summary.csv", result,
            per_view_csv_path=per_view_csv_path,
        )
    print(json.dumps({
        "iteration": result.iteration, "split": result.split,
        "views": [row.view_index for row in result.views],
        "summary": result.summary,
        "per_view_csv": str(run_directory / "evaluation_results.csv"),
        "summary_csv": str(run_directory / "evaluation_summary.csv"),
        "metadata_json": str(evaluation_directory / "metric_metadata.json"),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="existing training run directory")
    parser.add_argument("--checkpoint", type=int, required=True, help="completed iteration")
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--views", type=int, nargs="+", required=True,
                        help="full image indices in the selected split")
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--render-chunk-size", type=int, default=None)
    args = parser.parse_args()
    evaluate_checkpoint(args.run, args.checkpoint, args.split, args.views,
                        device=args.device, render_chunk_size=args.render_chunk_size)


if __name__ == "__main__":
    main()
