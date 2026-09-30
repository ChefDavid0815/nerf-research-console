"""Validate a real Blender scene and save Step 1 camera/image diagnostics.

Run from the project root: ``.venv\\Scripts\\python scripts/validate_step1.py``.
This script never trains a model and never substitutes generated research data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nerf_step1.dataset import load_blender_scene  # noqa: E402
from nerf_step1.diagnostics import save_camera_ray_plot, save_view_contact_sheet  # noqa: E402
from nerf_step1.rays import generate_rays  # noqa: E402
from nerf_step1.run_logging import start_run, write_status  # noqa: E402


REQUIRED_CONFIG_KEYS = (
    "position_encoding_L", "direction_encoding_L", "network_depth",
    "network_width", "num_coarse_samples", "num_fine_samples", "batch_size",
    "learning_rate", "training_iterations", "random_seed", "scene",
    "image_resolution", "dataset_root", "white_background",
    "normalize_ray_directions",
)


def read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"{path}: missing required keys: {', '.join(missing)}")
    if config["image_resolution"] != "original":
        raise ValueError("Step 1 supports only image_resolution: original; no resizing is implemented")
    if not isinstance(config["scene"], str) or not config["scene"]:
        raise ValueError("scene must be a nonempty string")
    if not isinstance(config["dataset_root"], str) or not config["dataset_root"]:
        raise ValueError("dataset_root must be a nonempty path string")
    if not isinstance(config["white_background"], bool):
        raise ValueError("white_background must be true or false")
    if not isinstance(config["normalize_ray_directions"], bool):
        raise ValueError("normalize_ray_directions must be true or false")
    return config


def _dataset_sha256(scene) -> str:
    """Fingerprint metadata and every referenced source image in split order."""
    digest = hashlib.sha256()
    files = [scene.directory / f"transforms_{name}.json" for name in ("train", "val", "test")]
    for split in (scene.train, scene.val, scene.test):
        files.extend(split.image_paths)
    for path in files:
        relative_path = path.relative_to(scene.directory).as_posix().encode("utf-8")
        digest.update(len(relative_path).to_bytes(4, "big"))
        digest.update(relative_path)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def validate_real_scene(config: dict, artifact_directory: Path) -> dict:
    data_root = Path(config["dataset_root"])
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    background = (1.0, 1.0, 1.0) if config["white_background"] else (0.0, 0.0, 0.0)
    scene = load_blender_scene(data_root, config["scene"], background_color=background)
    split_counts = {split.name: len(split) for split in (scene.train, scene.val, scene.test)}

    # Decode every referenced color view, one at a time, so a damaged middle
    # frame cannot pass merely because its PNG header and sampled views work.
    for split in (scene.train, scene.val, scene.test):
        center = torch.tensor([[split.intrinsics.width // 2, split.intrinsics.height // 2]])
        for index in range(len(split)):
            view = split.get_view(index)
            if view.image.shape != (split.intrinsics.height, split.intrinsics.width, 3):
                raise AssertionError(f"{split.name}[{index}]: decoded RGB shape is wrong")
            if not bool(torch.isfinite(view.image).all()):
                raise AssertionError(f"{split.name}[{index}]: RGB contains non-finite values")
            if not torch.equal(view.camera_to_world, split.camera_to_world[index]):
                raise AssertionError(f"{split.name}[{index}]: view pose does not match metadata order")
            origins, directions = generate_rays(
                view.camera_to_world, split.intrinsics, center,
                normalize_directions=config["normalize_ray_directions"],
            )
            if origins.shape != (1, 3) or directions.shape != (1, 3):
                raise AssertionError(f"{split.name}[{index}]: center ray shape is wrong")

    ray_path = artifact_directory / f"camera_rays_{scene.name}.png"
    views_path = artifact_directory / f"dataset_views_{scene.name}.png"
    origin_cosines = save_camera_ray_plot(scene, ray_path)
    save_view_contact_sheet(scene, views_path)
    if not all(math.isfinite(value) and value > 0.0 for value in origin_cosines):
        raise AssertionError(
            "One or more sampled center rays point away from the reference world origin; "
            "inspect the saved camera plot and dataset convention"
        )
    return {
        "scene_directory": str(scene.directory),
        "split_counts": split_counts,
        "decoded_color_views": sum(split_counts.values()),
        "dataset_content_sha256": _dataset_sha256(scene),
        "image_size": [scene.train.intrinsics.width, scene.train.intrinsics.height],
        "intrinsic_matrix": scene.train.intrinsics.matrix().tolist(),
        "sampled_center_ray_cosine_toward_origin": origin_cosines,
        "min_sampled_cosine": min(origin_cosines),
        "artifacts": [str(ray_path.resolve()), str(views_path.resolve())],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/baseline.yaml")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = read_config(config_path)
    run_directory, logger = start_run(PROJECT_ROOT / "runs", "step1_validation", config)
    try:
        result = validate_real_scene(config, PROJECT_ROOT / "artifacts" / run_directory.name)
    except Exception as exc:
        logger.exception("Step 1 validation failed")
        write_status(run_directory, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        return 1
    write_status(run_directory, {"status": "passed", **result})
    logger.info("Step 1 validation passed: %s", json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
