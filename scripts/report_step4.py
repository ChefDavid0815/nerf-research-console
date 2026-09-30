"""Build Step 4 figures and evidence reports from actual training/evaluation runs.

This script never substitutes training-batch PSNR for full-view metrics. Missing
requested checkpoints remain visibly missing in the reconstruction figure and
are listed in the report; no metric or view is interpolated or fabricated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from nerf_step1.dataset import load_blender_scene  # noqa: E402
from nerf_step1.pipeline import CoarseFineNeRF  # noqa: E402
from nerf_step1.rays import generate_rays  # noqa: E402
from nerf_step1.training_io import assert_run_config_compatible  # noqa: E402
from step4_artifacts import render_image_paths  # noqa: E402

EVALUATION_FIELDS = (
    "iteration", "split", "view_index", "psnr", "ssim", "lpips",
    "render_seconds", "rays_per_second", "peak_vram_bytes", "width", "height",
    "render_chunk_size",
)
METRIC_NAMES = ("psnr", "ssim", "lpips")
LONGITUDINAL_TEST_VIEWS = (0, 1, 2)
NOVEL_TEST_VIEWS = (0, 66, 133)
BUDGET_FIXED_KEYS = (
    "scene", "dataset_root", "image_resolution", "white_background",
    "normalize_ray_directions", "direction_encoding_L", "network_depth",
    "network_width", "skip_connection_layer", "view_width",
    "num_coarse_samples", "num_fine_samples", "learning_rate",
    "learning_rate_decay_steps", "learning_rate_decay_factor", "random_seed",
    "density_initial_bias", "near", "far",
)


def _number(value: str, name: str, *, optional: bool = False) -> float | None:
    if value == "" and optional:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def read_training_metrics(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"iteration", "total_loss", "coarse_loss", "fine_loss", "psnr",
                    "elapsed_time", "rays_per_second"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("training metrics CSV lacks Step 3 scalar fields")
        rows = []
        for raw in reader:
            if not raw or raw["iteration"] is None:
                raise ValueError("malformed training metric row")
            row = {key: _number(value, key, optional=True) for key, value in raw.items()}
            iteration = int(raw["iteration"])
            if float(iteration) != row["iteration"] or iteration != len(rows) + 1:
                raise ValueError("training iterations must be contiguous from 1")
            if any(row[key] is None for key in required):
                raise ValueError("required training metric is empty")
            rows.append(row)
    if not rows:
        raise ValueError("training metrics CSV has no rows")
    return rows


def read_evaluation_metrics(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not set(EVALUATION_FIELDS).issubset(reader.fieldnames or ()):
            raise ValueError("evaluation CSV lacks the full-view metric/performance schema")
        rows: list[dict[str, object]] = []
        seen: set[tuple[int, str, int]] = set()
        for raw in reader:
            iteration, view_index = int(raw["iteration"]), int(raw["view_index"])
            split = raw["split"]
            key = (iteration, split, view_index)
            if iteration < 0 or view_index < 0 or split not in ("train", "val", "test"):
                raise ValueError(f"invalid evaluation identity: {key}")
            if key in seen:
                raise ValueError(f"duplicate evaluation identity: {key}")
            seen.add(key)
            numeric = {name: _number(raw[name], name, optional=name == "peak_vram_bytes")
                       for name in EVALUATION_FIELDS[3:]}
            if any(numeric[name] is None for name in METRIC_NAMES):
                raise ValueError(f"missing image metric for {key}")
            if not (-1 <= numeric["ssim"] <= 1) or numeric["lpips"] < 0:
                raise ValueError(f"out-of-range image metric for {key}")
            if (numeric["width"] < 1 or numeric["height"] < 1 or
                    numeric["render_chunk_size"] < 1 or numeric["render_seconds"] <= 0):
                raise ValueError(f"invalid render dimensions or timing for {key}")
            if numeric["peak_vram_bytes"] is not None and numeric["peak_vram_bytes"] < 0:
                raise ValueError(f"invalid peak VRAM for {key}")
            rows.append({"iteration": iteration, "split": split,
                         "view_index": view_index, **numeric})
    if not rows:
        raise ValueError("evaluation CSV has no full-view rows")
    return sorted(rows, key=lambda row: (row["iteration"], row["split"], row["view_index"]))


def verify_evaluation_summary(path: Path, rows: list[dict[str, object]]) -> None:
    """Ensure the evaluator's aggregate export agrees with per-view evidence."""
    expected: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        expected[(row["iteration"], row["split"])].append(row)
    seen: set[tuple[int, str]] = set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"iteration", "split", "view_count"} | {
            f"{metric}_{stat}" for metric in METRIC_NAMES
            for stat in ("mean", "median", "std")}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("evaluation summary CSV lacks aggregate fields")
        for raw in reader:
            key = (int(raw["iteration"]), raw["split"])
            if key in seen or key not in expected:
                raise ValueError(f"summary has duplicate or unexpected checkpoint/split {key}")
            seen.add(key)
            if int(raw["view_count"]) != len(expected[key]):
                raise ValueError(f"summary view count differs from per-view rows for {key}")
            for metric in METRIC_NAMES:
                values = [float(row[metric]) for row in expected[key]]
                computed = _summarize(values)
                for stat, actual in zip(("mean", "median", "std"), computed):
                    if not math.isclose(float(raw[f"{metric}_{stat}"]), actual,
                                        rel_tol=1e-9, abs_tol=1e-9):
                        raise ValueError(f"summary {metric}_{stat} differs from per-view rows for {key}")
    if seen != set(expected):
        raise ValueError(f"evaluation summary is missing checkpoint/split groups: {sorted(set(expected)-seen)}")


def rows_for(rows: list[dict[str, object]], iteration: int, split: str) -> list[dict[str, object]]:
    return [row for row in rows if row["iteration"] == iteration and row["split"] == split]


def comparable_view_indices(rows: list[dict[str, object]], split: str) -> list[int]:
    groups = [set(row["view_index"] for row in rows_for(rows, iteration, split))
              for iteration in sorted({row["iteration"] for row in rows if row["split"] == split})]
    return sorted(set.intersection(*groups)) if groups else []


def _summarize(values: list[float]) -> tuple[float, float, float]:
    if not values:
        raise ValueError("cannot summarize an empty metric set")
    return statistics.mean(values), statistics.median(values), statistics.pstdev(values)


def _rolling(values: list[float], window: int = 100) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    cumulative = np.cumsum(np.r_[0.0, array])
    start = np.maximum(0, np.arange(len(array)) + 1 - window)
    return (cumulative[1:] - cumulative[start]) / (np.arange(len(array)) + 1 - start)


def sustained_slowing(rates: list[tuple[float, float, float]]) -> bool:
    """Require each of the last two rates to be nonnegative and no faster than all prior rates."""
    if len(rates) < 3:
        return False
    return all(
        all(rate[metric] >= 0 and rate[metric] <= min(
            previous[metric] for previous in rates[:index])
            for metric in range(3))
        for index, rate in enumerate(rates) if index >= len(rates) - 2
    )


def training_loss_interval_trend(training: list[dict[str, float]],
                                 validation_steps: list[int]) -> str:
    """Describe late loss with interval medians, keeping it separate from the metric gate."""
    if len(validation_steps) < 3:
        return "Fewer than three validation checkpoints exist for a paired training-loss interval comparison."
    a, b, c = validation_steps[-3:]
    prior = [float(row["total_loss"]) for row in training if a < row["iteration"] <= b]
    latest = [float(row["total_loss"]) for row in training if b < row["iteration"] <= c]
    if not prior or not latest:
        return "Training-loss interval medians are unavailable for the last two validation intervals."
    prior_median, latest_median = statistics.median(prior), statistics.median(latest)
    change = latest_median - prior_median
    return (f"Training total-loss medians over iterations ({a:,}, {b:,}] and "
            f"({b:,}, {c:,}] were {prior_median:.6f} (n={len(prior):,}) and "
            f"{latest_median:.6f} (n={len(latest):,}), respectively "
            f"(later minus earlier {change:+.6f}). This noisy training-ray statistic "
            "is context for researcher review, not an automatic convergence gate.")


def training_budget_scope(run_config: dict, proposed_config_path: Path) -> tuple[str, bool]:
    """State which future training settings this iteration curve can inform."""
    run_batch = int(run_config["batch_size"])
    if not proposed_config_path.is_file():
        return (f"This convergence run used {run_batch} training rays per batch. Its iteration "
                "budget can inform future L runs only when they retain that batch size and "
                "the other frozen training settings, apart from the intended L change. "
                "No proposed formal-run config was available for comparison.", False)
    proposed = yaml.safe_load(proposed_config_path.read_text(encoding="utf-8"))
    proposed_batch = int(proposed["batch_size"])
    proposal_label = (proposed_config_path.relative_to(PROJECT_ROOT).as_posix()
                      if proposed_config_path.is_relative_to(PROJECT_ROOT)
                      else proposed_config_path.as_posix())
    changed_settings = [key for key in BUDGET_FIXED_KEYS
                        if run_config.get(key) != proposed.get(key)]
    if proposed_batch != run_batch:
        return (f"This true Step 3 resume used the frozen batch size of {run_batch} training "
                f"rays, while `{proposal_label}` proposes {proposed_batch}. "
                f"Any iteration budget inferred here applies only to future L runs using "
                f"{run_batch} rays per batch and the other fixed training settings, apart "
                "from the intended L change. If the formal study keeps the proposed "
                f"{proposed_batch}-ray batch, a new matched {proposed_batch}-ray baseline "
                "convergence run is required before choosing its iteration budget.", False)
    if changed_settings:
        return (f"This convergence run and `{proposal_label}` both specify {run_batch} "
                f"training rays per batch, but the proposed config differs in other "
                f"fixed settings: {changed_settings}. An iteration budget from this run "
                "does not transfer until those settings match, apart from the intended "
                "position L change; otherwise run a matched baseline convergence study.", False)
    return (f"This convergence run and `{proposal_label}` both specify "
            f"{run_batch} training rays per batch. Its iteration budget applies only "
            "while future L runs retain this batch size and the other fixed training "
            "settings, apart from the intended L change.", True)


def budget_recommendation_text(recommended_budget: int | None,
                               budget_rationale: str | None,
                               validation_steps: list[int],
                               plateau_evidence: bool) -> tuple[str, bool]:
    """Distinguish a measured budget proposal from an unsupported provisional choice."""
    supported = bool(recommended_budget is not None and plateau_evidence and
                     len(validation_steps) >= 3 and recommended_budget in validation_steps and
                     recommended_budget >= validation_steps[-3])
    if supported:
        return (f"Proposed fixed Step 5 budget for runs matching this baseline's "
                f"training settings: **{recommended_budget:,} iterations per model**. "
                f"Researcher-supplied metric rationale: {budget_rationale}", True)
    if recommended_budget is not None:
        return (f"Provisional researcher-entered budget: **{recommended_budget:,} "
                "iterations per model**. The automated curve does not yet support "
                "this as a fixed Step 5 recommendation: it needs a measured plateau "
                "and a chosen checkpoint at or after the start of the last two "
                f"validation intervals. Supplied rationale: {budget_rationale}", False)
    return ("Current recommendation: **defer setting a fixed Step 5 iteration "
            "budget** until the completed matched-view curve supports a plateau.", False)


def selected_test_subset_mean(rows: list[dict[str, object]],
                              indices: tuple[int, ...], label: str) -> str:
    """Report one fixed test subset without silently pooling other test rows."""
    matched = [row for row in rows if row["view_index"] in indices]
    missing = sorted(set(indices) - {row["view_index"] for row in matched})
    if missing:
        return f"{label} indices {list(indices)}: unavailable (missing views {missing})."
    means = [statistics.mean(float(row[metric]) for row in matched)
             for metric in METRIC_NAMES]
    return (f"{label} indices {list(indices)} mean: PSNR {means[0]:.3f} dB, "
            f"SSIM {means[1]:.4f}, LPIPS {means[2]:.4f}.")


def panel_limit_text(progress_missing: list[tuple[int, int]],
                     novel_missing: list[int]) -> str:
    if progress_missing or novel_missing:
        return "Missing full-image panels are listed above."
    return "All requested full-image panels are present."


def run_status_explanation(status: dict, selected: int) -> str:
    if status.get("status") == "paused_after_checkpoint" and status.get("last_completed_iteration") == selected:
        return ("The `paused_after_checkpoint` status is the trainer's intentional "
                f"extension-stage stop after saving iteration {selected:,}, not a training failure.")
    return ""


def configure_training_loss_axis(axis: plt.Axes, training: list[dict[str, float]]) -> None:
    if any(float(row[name]) <= 0 for row in training
           for name in ("total_loss", "coarse_loss", "fine_loss")):
        raise ValueError("training MSE must be positive for the log-scaled loss axis")
    axis.set_yscale("log")
    axis.set_ylabel("Training-ray RGB MSE (log scale)")


def save_metrics_curve(training: list[dict[str, float]],
                       evaluation: list[dict[str, object]], path: Path) -> list[int]:
    """Plot one matched held-out validation view set across checkpoints."""
    val_views = comparable_view_indices(evaluation, "val")
    if not val_views:
        raise ValueError("validation checkpoints have no common full-image view")
    checkpoints = sorted({row["iteration"] for row in evaluation if row["split"] == "val"})
    means = {name: [statistics.mean(
        float(row[name]) for row in rows_for(evaluation, step, "val")
        if row["view_index"] in val_views) for step in checkpoints] for name in METRIC_NAMES}
    steps = [int(row["iteration"]) for row in training]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    for axis in axes.flat:
        axis.set_facecolor("#fafbfc")
        axis.grid(color="#d9e0e4", alpha=0.8, linewidth=0.7)
        axis.set_xlabel("Completed training iteration")
        axis.spines[["top", "right"]].set_visible(False)
    loss_axis = axes[0, 0]
    for name, color, label in (("total_loss", "#ad4f35", "Total"),
                               ("coarse_loss", "#44708e", "Coarse"),
                               ("fine_loss", "#8566a0", "Fine")):
        values = [float(row[name]) for row in training]
        loss_axis.plot(steps, values, color=color, alpha=0.13, linewidth=0.55)
        loss_axis.plot(steps, _rolling(values), color=color, linewidth=1.7,
                       label=f"{label} MSE (100-step mean)")
    configure_training_loss_axis(loss_axis, training)
    loss_axis.legend(fontsize=8)
    for axis, name, color, ylabel in (
        (axes[0, 1], "psnr", "#275d83", "Validation PSNR ↑ (dB)"),
        (axes[1, 0], "ssim", "#487a4e", "Validation SSIM ↑"),
        (axes[1, 1], "lpips", "#8d4f85", "Validation LPIPS ↓"),
    ):
        axis.plot(checkpoints, means[name], "o-", color=color, linewidth=2,
                  markersize=5)
        axis.set_xlim(0, max(steps) * 1.04)
        axis.set_ylabel(ylabel)
        for step, value in zip(checkpoints, means[name]):
            axis.annotate(f"{value:.3f}" if name != "psnr" else f"{value:.2f}",
                          (step, value), xytext=(4, 5), textcoords="offset points",
                          fontsize=7, color=color)
    fig.suptitle("Lego baseline convergence | train batches and held-out validation\n"
                 f"Validation curve uses matched view indices {val_views} at every checkpoint",
                 fontsize=13, fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return val_views


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _render_panel(run_directory: Path, row: dict[str, object] | None,
                  key: str, size: int) -> Image.Image | None:
    if row is None:
        return None
    path = render_image_paths(run_directory, row["iteration"], row["split"],
                              row["view_index"])[key]
    if not path.is_file():
        return None
    with Image.open(path) as source:
        expected = (int(row["width"]), int(row["height"]))
        if source.size != expected:
            raise ValueError(f"{path}: visualization is not full image {expected}")
        return source.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def _missing_images(run_directory: Path, row: dict[str, object] | None) -> bool:
    return row is None or any(not path.is_file() for path in
                              render_image_paths(run_directory, row["iteration"],
                                                 row["split"], row["view_index"]).values())


def save_progress(run_directory: Path, evaluation: list[dict[str, object]],
                  stages: list[int], views: list[int], path: Path) -> list[tuple[int, int]]:
    panel, gap, margin, label_height = 200, 6, 26, 62
    block_width = 3 * panel + 2 * gap
    width = margin * 2 + len(stages) * (block_width + 18)
    height = 108 + len(views) * (panel + label_height)
    sheet = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(sheet)
    heading, body, stage_font = _font(21), _font(15), _font(17)
    draw.text((margin, 8), "Baseline reconstruction progress | held-out validation views",
              fill="#15232d", font=heading)
    draw.text((margin, 35), "Each triptych: ground truth | prediction | absolute RGB difference (display-scaled only)",
              fill="#52616d", font=body)
    missing: list[tuple[int, int]] = []
    for column, stage in enumerate(stages):
        x = margin + column * (block_width + 18)
        draw.text((x, 62), f"Iteration {stage:,}", fill="#15232d", font=stage_font)
        for k, title in enumerate(("Ground truth", "Prediction", "Absolute difference")):
            draw.text((x + k * (panel + gap), 85), title, fill="#52616d", font=body)
        for view_position, view_index in enumerate(views):
            y = 108 + view_position * (panel + label_height)
            row = next((r for r in evaluation if r["iteration"] == stage and
                        r["split"] == "val" and r["view_index"] == view_index), None)
            if _missing_images(run_directory, row):
                missing.append((stage, view_index))
                draw.rectangle((x, y, x + block_width, y + panel), fill="#eef1f2")
                draw.text((x + 8, y + panel // 2), "No full-view render", fill="#6b7880", font=body)
            else:
                for k, key in enumerate(("ground_truth", "prediction", "absolute_difference")):
                    image = _render_panel(run_directory, row, key, panel)
                    sheet.paste(image, (x + k * (panel + gap), y))
            metric_text = (f"val view {view_index}  PSNR {row['psnr']:.2f} dB  "
                           f"SSIM {row['ssim']:.3f}  LPIPS {row['lpips']:.3f}"
                           if row is not None else f"val view {view_index}: not evaluated")
            draw.text((x, y + panel + 6), metric_text, fill="#24323d", font=body)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return missing


def save_novel_views(run_directory: Path, evaluation: list[dict[str, object]],
                     iteration: int, views: list[int], path: Path) -> list[int]:
    panel, gap, margin = 250, 7, 22
    width = margin * 2 + 3 * panel + 2 * gap
    height = 80 + len(views) * (panel + 55)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    heading, body = _font(18), _font(14)
    draw.text((margin, 10), f"Spatially separated held-out test views | iteration {iteration:,}",
              fill="#15232d", font=heading)
    for col, title in enumerate(("Ground truth", "Prediction", "Absolute RGB difference")):
        draw.text((margin + col * (panel + gap), 47), title, fill="#52616d", font=body)
    missing: list[int] = []
    for position, view_index in enumerate(views):
        y = 80 + position * (panel + 55)
        row = next((r for r in evaluation if r["iteration"] == iteration and
                    r["split"] == "test" and r["view_index"] == view_index), None)
        if _missing_images(run_directory, row):
            missing.append(view_index)
            draw.rectangle((margin, y, width - margin, y + panel), fill="#eef1f2")
            draw.text((margin + 10, y + panel // 2), f"Test view {view_index}: no full-view render",
                      fill="#6b7880", font=body)
        else:
            for col, key in enumerate(("ground_truth", "prediction", "absolute_difference")):
                image = _render_panel(run_directory, row, key, panel)
                sheet.paste(image, (margin + col * (panel + gap), y))
            draw.text((margin, y + panel + 9),
                      f"test view {view_index}  PSNR {row['psnr']:.2f} dB   "
                      f"SSIM {row['ssim']:.3f}   LPIPS {row['lpips']:.3f}",
                      fill="#24323d", font=body)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return missing


def _load_model(run_directory: Path, config: dict, iteration: int,
                device: torch.device) -> CoarseFineNeRF:
    path = run_directory / "checkpoints" / f"iter_{iteration:06d}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["iteration"] != iteration or payload["config"] != config:
        raise ValueError(f"checkpoint identity/config mismatch: {path}")
    model = CoarseFineNeRF(config)
    model.coarse_model.load_state_dict(payload["coarse_model"])
    model.fine_model.load_state_dict(payload["fine_model"])
    return model.to(device).eval()


def _select_object_pixels(image: torch.Tensor) -> list[tuple[int, int]]:
    """Pick deterministic nonwhite pixels from left, middle and right thirds."""
    height, width = image.shape[:2]
    mask = ((1.0 - image).abs().mean(dim=-1) > 0.08).numpy()
    pixels: list[tuple[int, int]] = []
    for region in range(3):
        left, right = region * width // 3, (region + 1) * width // 3
        yx = np.argwhere(mask[:, left:right])
        if len(yx) == 0:
            continue
        center = np.array([height / 2, (right - left) / 2])
        y, x_local = yx[np.argmin(np.square(yx - center).sum(axis=1))]
        pixels.append((int(x_local + left), int(y)))
    if len(pixels) < 3:
        yx = np.argwhere(mask)
        for index in np.linspace(0, len(yx) - 1, num=min(3, len(yx)), dtype=int):
            candidate = (int(yx[index, 1]), int(yx[index, 0]))
            if candidate not in pixels:
                pixels.append(candidate)
            if len(pixels) == 3:
                break
    if len(pixels) < 3:
        raise ValueError("held-out view has fewer than three distinct nonwhite object pixels")
    return pixels[:3]


def central_weight_bin_interval(depths: np.ndarray, weights: np.ndarray,
                                near: float, far: float) -> tuple[float, float] | None:
    """Coarse midpoint-bin span containing central 80% of positive weight."""
    if depths.ndim != 1 or weights.shape != depths.shape or len(depths) < 2:
        raise ValueError("depths and weights must be aligned one-dimensional arrays")
    total = float(weights.sum())
    if total <= 1e-6:
        return None
    edges = np.r_[near, 0.5 * (depths[1:] + depths[:-1]), far]
    cdf = np.cumsum(weights) / total
    first = int(np.searchsorted(cdf, 0.1))
    last = int(np.searchsorted(cdf, 0.9))
    return float(edges[first]), float(edges[last + 1])


def save_sampling_diagnostic(run_directory: Path, config: dict, selected: int,
                             path: Path, *, device: torch.device) -> tuple[list[tuple[int, int]], dict[str, tuple[float, float]]]:
    scene_root = Path(config["dataset_root"])
    if not scene_root.is_absolute():
        scene_root = PROJECT_ROOT / scene_root
    scene = load_blender_scene(scene_root, config["scene"], background_color=(1, 1, 1))
    view = scene.val.get_view(0)
    pixels = _select_object_pixels(view.image)
    stages = list(dict.fromkeys((500, selected)))
    fig, axes = plt.subplots(len(pixels), len(stages), figsize=(6 * len(stages), 9),
                             sharex=True, squeeze=False, constrained_layout=True)
    pixel_tensor = torch.tensor(pixels, dtype=torch.int64)
    origins, directions = generate_rays(
        view.camera_to_world, view.intrinsics, pixel_tensor,
        device=device, dtype=torch.float32,
        normalize_directions=bool(config["normalize_ray_directions"]),
    )
    concentration: dict[str, tuple[float, float]] = {}
    for col, stage in enumerate(stages):
        model = _load_model(run_directory, config, stage, device)
        with torch.no_grad():
            output = model(origins, directions, randomized=False)
        coarse_depths = output.coarse_depths.cpu().numpy()
        coarse_weights = output.coarse.weights.cpu().numpy()
        fine_depths = output.fine_depths.cpu().numpy()
        combined_depths = output.combined_depths.cpu().numpy()
        fine_weights = output.fine.weights.cpu().numpy()
        for ray in range(len(pixels)):
            axis = axes[ray, col]
            axis.plot(coarse_depths[ray], coarse_weights[ray], ".-", color="#ad4f35",
                      markersize=3, linewidth=1.2, label="Coarse weights at coarse depths")
            axis.plot(combined_depths[ray], fine_weights[ray], ".", color="#275d83",
                      markersize=2, alpha=0.7, label="Fine weights on merged depths")
            axis.eventplot(coarse_depths[ray], lineoffsets=-0.019,
                           linelengths=0.004, colors="#7a7f86", linewidths=0.7)
            axis.eventplot(fine_depths[ray], lineoffsets=-0.012,
                           linelengths=0.009, colors="#487a4e", linewidths=0.7)
            axis.set_ylim(bottom=-0.022)
            axis.set_xlim(float(config["near"]), float(config["far"]))
            axis.set_title(f"val view 0, pixel {pixels[ray]} | iteration {stage:,}", fontsize=10)
            if col == 0:
                axis.set_ylabel("Rendering weight")
            axis.grid(alpha=0.2)
            # A mass interval measures sampling concentration, not true surface
            # depth; Blender GT here contains RGB only, without depth labels.
            interval = central_weight_bin_interval(
                coarse_depths[ray], coarse_weights[ray],
                float(config["near"]), float(config["far"]))
            if interval is not None:
                lower, upper = interval
                fraction = float(np.mean((fine_depths[ray] >= lower) &
                                         (fine_depths[ray] <= upper)))
                span_fraction = (upper - lower) / (float(config["far"]) - float(config["near"]))
                concentration[f"iter_{stage}_pixel_{pixels[ray][0]}_{pixels[ray][1]}"] = (
                    fraction, span_fraction)
                axis.axvspan(lower, upper, color="#487a4e", alpha=0.08)
    for axis in axes[-1]:
        axis.set_xlabel("Ray parameter t (near–far)")
    axes[0, 0].plot([], [], color="#7a7f86", linewidth=1, label="Coarse sample depths (lower rug)")
    axes[0, 0].plot([], [], color="#487a4e", linewidth=1, label="Fine sample depths (upper rug)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Lego hierarchical sampling | same held-out rays",
                 fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return pixels, concentration


def _checkpoint_adam_step_count(checkpoint: dict, iteration: int) -> int:
    optimizer = checkpoint.get("optimizer")
    if not isinstance(optimizer, dict):
        raise ValueError(f"checkpoint {iteration}: optimizer state is missing")
    states, groups = optimizer.get("state"), optimizer.get("param_groups")
    if not isinstance(states, dict) or not states or not isinstance(groups, list):
        raise ValueError(f"checkpoint {iteration}: Adam state is empty or malformed")
    parameter_ids = [parameter for group in groups for parameter in group["params"]]
    if len(parameter_ids) != len(set(parameter_ids)) or set(states) != set(parameter_ids):
        raise ValueError(f"checkpoint {iteration}: Adam states do not cover each optimizer parameter")
    for state in states.values():
        raw = state.get("step")
        if isinstance(raw, torch.Tensor):
            if raw.numel() != 1:
                raise ValueError(f"checkpoint {iteration}: Adam step is not scalar")
            raw = raw.item()
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw != iteration:
            raise ValueError(f"checkpoint {iteration}: Adam step counter differs from iteration")
    return len(states)


def verify_checkpoint_continuity(run_directory: Path, config: dict,
                                 source_iteration: int, selected: int) -> dict:
    """Check present-file resume state without claiming historical byte identity."""
    if selected < source_iteration:
        raise ValueError("selected checkpoint cannot precede source checkpoint")
    checkpoints = {}
    state_counts = {}
    scheduler_counts = {}
    for iteration in dict.fromkeys((source_iteration, selected)):
        path = run_directory / "checkpoints" / f"iter_{iteration:06d}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("iteration") != iteration:
            raise ValueError(f"checkpoint {iteration}: saved iteration differs from filename")
        if checkpoint.get("config") != config:
            raise ValueError(f"checkpoint {iteration}: saved config differs from run snapshot")
        state_counts[iteration] = _checkpoint_adam_step_count(checkpoint, iteration)
        scheduler = checkpoint.get("scheduler")
        if not isinstance(scheduler, dict) or scheduler.get("last_epoch") != iteration:
            raise ValueError(f"checkpoint {iteration}: scheduler epoch differs from iteration")
        scheduler_counts[iteration] = int(scheduler["last_epoch"])
        checkpoints[iteration] = checkpoint
    changes = {}
    for network in (() if selected == source_iteration else ("coarse_model", "fine_model")):
        early = checkpoints[source_iteration].get(network)
        late = checkpoints[selected].get(network)
        if not isinstance(early, dict) or not isinstance(late, dict) or early.keys() != late.keys():
            raise ValueError(f"{network}: source/selected parameter keys differ")
        if not early:
            raise ValueError(f"{network}: no saved parameter tensors")
        changed_count = 0
        squared_norm = 0.0
        for key, first in early.items():
            last = late[key]
            if not isinstance(first, torch.Tensor) or not isinstance(last, torch.Tensor) or first.shape != last.shape:
                raise ValueError(f"{network}.{key}: source/selected tensor mismatch")
            delta = last.to(dtype=torch.float64) - first.to(dtype=torch.float64)
            squared_norm += float(delta.square().sum())
            changed_count += not torch.equal(first, last)
        if not changed_count:
            raise ValueError(f"{network}: no saved parameter tensors changed after resume")
        changes[network] = {"changed_tensors": changed_count,
                            "total_tensors": len(early),
                            "l2_change": math.sqrt(squared_norm)}
    return {"source_iteration": source_iteration, "selected_iteration": selected,
            "adam_parameter_states": state_counts, "scheduler_epochs": scheduler_counts,
            "network_changes": changes}


def checkpoint_provenance_summary(run_directory: Path,
                                  iterations: tuple[int, int]) -> str:
    """Describe recorded vs retrospective hashes for the files present now."""
    provenance_path = run_directory / "evaluation" / "checkpoint_provenance.json"
    if not provenance_path.is_file():
        return ("No evaluator checkpoint-hash manifest is available. The present checkpoint "
                "contents were checked, but historical evaluation-byte identity is unverified.")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("algorithm") != "sha256":
        raise ValueError("checkpoint provenance uses an unsupported hash algorithm")
    descriptions = []
    for iteration in dict.fromkeys(iterations):
        entry = provenance.get("checkpoints", {}).get(str(iteration))
        if entry is None:
            descriptions.append(f"iteration {iteration:,}: no evaluator hash entry")
            continue
        relative = f"checkpoints/iter_{iteration:06d}.pt"
        if entry.get("relative_path") != relative:
            raise ValueError(f"checkpoint {iteration}: provenance path differs")
        source = entry.get("source")
        if source not in ("recorded_before_evaluation",
                          "backfilled_from_current_file_after_existing_metrics"):
            raise ValueError(f"checkpoint {iteration}: unknown provenance source marker")
        checkpoint_path = run_directory / relative
        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if digest != entry.get("sha256"):
            raise ValueError(f"checkpoint {iteration}: present file differs from recorded SHA-256")
        descriptions.append(f"iteration {iteration:,}: SHA-256 `{digest}` ({source})")
    return ("Evaluator checkpoint provenance for current files: " + "; ".join(descriptions) +
            ". A `recorded_before_evaluation` digest can link a later evaluation to "
            "the recorded file if its current digest still matches. A "
            "`backfilled_from_current_file_after_existing_metrics` digest checks only "
            "the file present when backfilled; it cannot prove these exact weight bytes "
            "were used when earlier metric rows were first scored. Neither digest nor "
            "serialized step counters alone proves every intermediate update occurred "
            "uninterrupted.")


def _pytest_summary(path: Path | None) -> str:
    if path is None:
        return "No Step 4 pytest XML was supplied to this report generator."
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        raise ValueError("pytest XML has no testsuite")
    tests, failures, errors, skipped = (int(suite.get(k, "0")) for k in
                                        ("tests", "failures", "errors", "skipped"))
    passed = tests - failures - errors - skipped
    return f"{passed}/{tests} passed; {skipped} skipped, {failures} failed, {errors} errors (`{path}`)."


def _metric_table(evaluation: list[dict[str, object]], iterations: list[int]) -> str:
    header = "| Iteration | Split | View indices | PSNR mean / median / SD (dB) | SSIM mean / median / SD | LPIPS mean / median / SD |\n|---:|---|---|---:|---:|---:|"
    lines = [header]
    for step in iterations:
        for split in ("train", "val", "test"):
            rows = rows_for(evaluation, step, split)
            if not rows:
                continue
            triples = [_summarize([float(row[name]) for row in rows]) for name in METRIC_NAMES]
            fields = [f"{mean:.3f} / {median:.3f} / {sd:.3f}" for mean, median, sd in triples]
            indices = ", ".join(str(row["view_index"]) for row in rows)
            lines.append(f"| {step:,} | {split} | {indices} | {' | '.join(fields)} |")
    return "\n".join(lines)


def overfitting_assessment(evaluation: list[dict[str, object]]) -> tuple[str, bool, list[int]]:
    """Compare PSNR only on identical view IDs within each split over time."""
    full_split_steps = sorted(set(r["iteration"] for r in evaluation if r["split"] == "train") &
                              set(r["iteration"] for r in evaluation if r["split"] == "val") &
                              set(r["iteration"] for r in evaluation if r["split"] == "test"))
    if len(full_split_steps) < 3:
        return ("Overfitting trend cannot be determined: fewer than three checkpoints have train, validation, and test full-view PSNR.",
                False, full_split_steps)
    tail = full_split_steps[-3:]
    matched = {split: sorted(set.intersection(*[
        set(r["view_index"] for r in rows_for(evaluation, step, split))
        for step in tail])) for split in ("train", "val", "test")}
    matched["test"] = sorted(set(matched["test"]) & set(LONGITUDINAL_TEST_VIEWS))
    if any(not indices for indices in matched.values()):
        return ("Overfitting trend cannot be determined: train/validation/test evaluations "
                "do not contain a common view within each split across the last three shared checkpoints.",
                False, full_split_steps)

    def split_mean(step: int, split: str) -> float:
        return statistics.mean(float(r["psnr"]) for r in rows_for(evaluation, step, split)
                               if r["view_index"] in matched[split])

    train_up = all(split_mean(b, "train") > split_mean(a, "train")
                   for a, b in zip(tail, tail[1:]))
    val_down = all(split_mean(b, "val") <= split_mean(a, "val")
                   for a, b in zip(tail, tail[1:]))
    test_down = all(split_mean(b, "test") <= split_mean(a, "test")
                    for a, b in zip(tail, tail[1:]))
    possible = train_up and (val_down or test_down)
    description = ("Possible overfitting pattern: matched-view train full-view PSNR rose through the last two intervals while validation or test PSNR did not."
                   if possible else
                   "No sustained matched-view train-up versus validation/test-down PSNR pattern across the last two common intervals; limited views/checkpoints still restrict inference.")
    return description + f" Matched per-split indices across checkpoints {tail}: {matched}.", possible, full_split_steps


def _metadata_report(metadata: dict | None) -> str:
    if metadata is None:
        return "Metric implementation metadata JSON was unavailable; verify implementations before comparing experiments."
    ssim = metadata.get("ssim", {})
    lpips = metadata.get("lpips", {})
    summary = (
        f"SSIM: `{ssim.get('implementation', 'unrecorded')}` version "
        f"`{ssim.get('package_version', 'unrecorded')}`, win_size={ssim.get('win_size', 'unrecorded')}, "
        f"K1={ssim.get('K1', 'unrecorded')}, K2={ssim.get('K2', 'unrecorded')}, "
        f"data_range={ssim.get('data_range', 'unrecorded')}, channel_axis={ssim.get('channel_axis', 'unrecorded')}. "
        f"LPIPS: `{lpips.get('implementation', 'unrecorded')}` version "
        f"`{lpips.get('package_version', 'unrecorded')}`, backbone `{lpips.get('backbone', 'unrecorded')}`, "
        f"input convention `{lpips.get('input', 'unrecorded')}`. Weight SHA-256 values and the "
        "RGB roundoff rule are recorded below.\n\n"
    )
    return summary + "```json\n" + json.dumps(metadata, indent=2, ensure_ascii=False) + "\n```"


def _checkpoint_table(run_directory: Path, training: list[dict[str, float]],
                      evaluation: list[dict[str, object]], val_views: list[int],
                      selected: int) -> str:
    checkpoint_steps = sorted(int(path.stem.removeprefix("iter_")) for path in
                              (run_directory / "checkpoints").glob("iter_*.pt")
                              if 500 <= int(path.stem.removeprefix("iter_")) <= selected)
    lines = ["| Checkpoint | Total training loss | Train-batch PSNR (dB) | Held-out val PSNR (dB) | Elapsed (h) | Post-step GPU allocated (GiB) | Stage peak allocated (GiB) | Preview |",
             "|---:|---:|---:|---:|---:|---:|---:|---|"]
    for step in checkpoint_steps:
        row = training[step - 1]
        matched = [r for r in rows_for(evaluation, step, "val")
                   if r["view_index"] in val_views]
        validation = f"{statistics.mean(float(r['psnr']) for r in matched):.3f}" if matched else "not evaluated"
        allocated = row.get("gpu_memory_allocated_bytes")
        allocated_text = f"{allocated/2**30:.3f}" if allocated is not None else "unavailable"
        telemetry_path = run_directory / "checkpoints" / f"iter_{step:06d}_telemetry.json"
        telemetry = json.loads(telemetry_path.read_text(encoding="utf-8")) if telemetry_path.is_file() else {}
        peak = telemetry.get("gpu_peak_allocated_since_stage_start_bytes")
        peak_text = f"{peak/2**30:.3f}" if peak is not None else "unavailable"
        preview_path = run_directory / "renders" / f"comparison_iter_{step:06d}.png"
        preview = "saved" if preview_path.is_file() else "missing"
        lines.append(f"| {step:,} | {row['total_loss']:.6f} | {row['psnr']:.3f} | {validation} | "
                     f"{row['elapsed_time']/3600:.2f} | {allocated_text} | {peak_text} | {preview} |")
    return "\n".join(lines)


def _performance_table(evaluation: list[dict[str, object]], selected: int) -> str:
    lines = ["| Split | View | Render time (s) | Rays/s | Peak allocated VRAM (GiB) | Resolution | Chunk size |",
             "|---|---:|---:|---:|---:|---|---:|"]
    for row in (r for r in evaluation if r["iteration"] == selected):
        vram = row["peak_vram_bytes"]
        vram_text = f"{vram/2**30:.3f}" if vram is not None else "unavailable"
        lines.append(f"| {row['split']} | {row['view_index']} | {row['render_seconds']:.2f} | "
                     f"{row['rays_per_second']:,.0f} | {vram_text} | "
                     f"{int(row['width'])}×{int(row['height'])} | {int(row['render_chunk_size'])} |")
    return "\n".join(lines)


def hardware_summary(snapshot_path: Path) -> str:
    """Describe recorded run hardware without inferring current GPU state."""
    if not snapshot_path.is_file():
        return "Run environment snapshot unavailable; GPU model and total VRAM were not recorded here."
    environment = json.loads(snapshot_path.read_text(encoding="utf-8"))
    name = environment.get("gpu_name")
    total_bytes = environment.get("gpu_total_vram_bytes")
    if name is None or total_bytes is None:
        return "Run environment snapshot reports no available CUDA GPU or total VRAM."
    return (f"Recorded GPU: **{name}** with **{total_bytes / 2**30:.2f} GiB** total VRAM "
            f"(driver {environment.get('nvidia_driver', 'unrecorded')}, "
            f"PyTorch {environment.get('pytorch', 'unrecorded')}, "
            f"CUDA runtime {environment.get('pytorch_cuda_runtime', 'unrecorded')}). "
            f"Snapshot time: {environment.get('captured_utc', 'unrecorded')}.")


def _check_disjoint_source_splits(config: dict) -> dict[str, int]:
    root = Path(config["dataset_root"])
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    directory = root / config["scene"]
    sources: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        payload = json.loads((directory / f"transforms_{split}.json").read_text(encoding="utf-8"))
        sources[split] = {str(frame["file_path"]) for frame in payload["frames"]}
        if len(sources[split]) != len(payload["frames"]):
            raise ValueError(f"duplicate source path within {split} split")
    if any(sources[a] & sources[b] for a, b in
           (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Lego source image paths overlap across splits")
    return {split: len(paths) for split, paths in sources.items()}


def write_reports(run_directory: Path, artifacts: Path, training: list[dict[str, float]],
                  evaluation: list[dict[str, object]], selected: int, stages: list[int],
                  val_views: list[int], progress_missing: list[tuple[int, int]],
                  novel_missing: list[int], sampling_pixels: list[tuple[int, int]],
                  concentration: dict[str, tuple[float, float]], metadata: dict | None,
                  continuity: dict, provenance_text: str,
                  pytest_xml: Path | None, recommended_budget: int | None,
                  budget_rationale: str | None,
                  performance_workload_note: str | None) -> None:
    config = yaml.safe_load((run_directory / "config_snapshot.yaml").read_text(encoding="utf-8"))
    hardware = hardware_summary(run_directory / "environment_snapshot.json")
    split_counts = _check_disjoint_source_splits(config)
    status_path = run_directory / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    status_iteration = status.get("last_completed_iteration")
    train_log = (run_directory / "train.log").read_text(encoding="utf-8")
    resume_evidence = "Resumed " in train_log and "completed iteration 500" in train_log
    checkpoints = sorted(int(path.stem.removeprefix("iter_")) for path in
                         (run_directory / "checkpoints").glob("iter_*.pt")
                         if int(path.stem.removeprefix("iter_")) <= selected)
    selected_row = training[selected - 1]
    changes = continuity["network_changes"]
    eval_steps = sorted({row["iteration"] for row in evaluation})
    latest_val = rows_for(evaluation, selected, "val")
    latest_test = rows_for(evaluation, selected, "test")
    latest_train = rows_for(evaluation, selected, "train")
    novel_test_mean = selected_test_subset_mean(
        latest_test, NOVEL_TEST_VIEWS, "Spatially separated novel-view test subset")
    fixed_test_mean = selected_test_subset_mean(
        latest_test, LONGITUDINAL_TEST_VIEWS, "Fixed longitudinal test subset")
    panel_status = panel_limit_text(progress_missing, novel_missing)
    status_explanation = run_status_explanation(status, selected)
    metric_trend = []
    for step in sorted({row["iteration"] for row in evaluation if row["split"] == "val"}):
        matched = [row for row in rows_for(evaluation, step, "val")
                   if row["view_index"] in val_views]
        metric_trend.append((step, *(statistics.mean(float(row[name]) for row in matched)
                                     for name in METRIC_NAMES)))
    trend_lines = ["| Iteration | Matched val PSNR ↑ | Matched val SSIM ↑ | Matched val LPIPS ↓ |",
                   "|---:|---:|---:|---:|"]
    trend_lines += [f"| {step:,} | {psnr:.3f} | {ssim:.4f} | {lpips:.4f} |"
                    for step, psnr, ssim, lpips in metric_trend]
    if len(metric_trend) >= 2:
        changes_text = (f"From the first to last matched validation evaluation, PSNR changed by "
                        f"{metric_trend[-1][1]-metric_trend[0][1]:+.3f} dB, SSIM by "
                        f"{metric_trend[-1][2]-metric_trend[0][2]:+.4f}, and LPIPS by "
                        f"{metric_trend[-1][3]-metric_trend[0][3]:+.4f}.")
    else:
        changes_text = "Only one matched validation evaluation exists, so no validation trend is established."
    # No fixed numeric threshold is imposed before seeing the curve. A
    # conservative, data-relative diminishing-return signal requires two late
    # nonnegative improvement rates for every metric, each no faster than all
    # preceding rates. This is evidence for review, not a universal threshold.
    rates = [tuple(1000 * (b[index] - a[index]) / (b[0] - a[0]) * sign
                   for index, sign in ((1, 1), (2, 1), (3, -1)))
             for a, b in zip(metric_trend, metric_trend[1:])]
    slowing_rates = sustained_slowing(rates)
    late_intervals = list(zip(metric_trend[-3:], metric_trend[-2:])) if len(metric_trend) >= 3 else []
    paired_gains: list[tuple[int, int, dict[str, int]]] = []
    for earlier, later in late_intervals:
        before = {r["view_index"]: r for r in rows_for(evaluation, earlier[0], "val")
                  if r["view_index"] in val_views}
        after = {r["view_index"]: r for r in rows_for(evaluation, later[0], "val")
                 if r["view_index"] in val_views}
        counts = {metric: sum((float(after[index][metric]) - float(before[index][metric])) * sign > 0
                              for index in val_views)
                  for metric, sign in (("psnr", 1), ("ssim", 1), ("lpips", -1))}
        paired_gains.append((earlier[0], later[0], counts))
    mixed_view_gains = bool(len(paired_gains) == 2 and all(
        count < len(val_views) for _, _, counts in paired_gains for count in counts.values()))
    plateau_evidence = slowing_rates and mixed_view_gains
    if len(rates) >= 3:
        convergence = ("Each of the last two validation intervals has nonnegative aggregate gains no faster than every preceding interval, and gains are not unanimous across matched views in any metric: a measured diminishing-return signal."
                       if plateau_evidence else
                       "The last two validation intervals do not yet combine consistently nonincreasing, nonnegative aggregate gains with non-unanimous paired-view improvement in all three metrics; practical convergence is not established.")
    else:
        convergence = "Fewer than four validation checkpoints exist; practical convergence cannot be established from sustained late trends."
    rate_lines = ["| Validation interval | PSNR gain (dB / 1k iter) | SSIM gain (/ 1k iter) | LPIPS reduction (/ 1k iter) |",
                  "|---|---:|---:|---:|"]
    rate_lines += [f"| {a[0]:,} → {b[0]:,} | {rate[0]:+.4f} | {rate[1]:+.5f} | {rate[2]:+.5f} |"
                   for a, b, rate in zip(metric_trend, metric_trend[1:], rates)]
    paired_lines = ["| Late validation interval | Views improving PSNR | Views improving SSIM | Views reducing LPIPS |",
                    "|---|---:|---:|---:|"]
    paired_lines += [f"| {a:,} → {b:,} | {counts['psnr']}/{len(val_views)} | "
                     f"{counts['ssim']}/{len(val_views)} | {counts['lpips']}/{len(val_views)} |"
                     for a, b, counts in paired_gains]
    overfit, possible_overfit, full_split_steps = overfitting_assessment(evaluation)
    missing_stages = [step for step in stages if any((step, view) in progress_missing for view in (0, 1, 2))]
    performance_rows = [row for row in evaluation if row["iteration"] == selected]
    perf = (f"At selected iteration {selected:,}, {len(performance_rows)} full-image renders took "
            f"median {statistics.median(float(r['render_seconds']) for r in performance_rows):.2f} s/image "
            f"(range {min(float(r['render_seconds']) for r in performance_rows):.2f}–"
            f"{max(float(r['render_seconds']) for r in performance_rows):.2f}); median "
            f"{statistics.median(float(r['rays_per_second']) for r in performance_rows):,.0f} rays/s. "
            f"Resolutions: {sorted({(int(r['width']), int(r['height'])) for r in performance_rows})}; "
            f"chunk sizes: {sorted({int(r['render_chunk_size']) for r in performance_rows})}."
            if performance_rows else "No full-image render performance rows exist at the selected checkpoint.")
    vram = [float(r["peak_vram_bytes"]) for r in performance_rows if r["peak_vram_bytes"] is not None]
    peak_text = f"Peak allocated CUDA memory during a render: {max(vram)/2**30:.3f} GiB." if vram else "Peak evaluation VRAM was unavailable."
    train_allocated = selected_row.get("gpu_memory_allocated_bytes")
    train_vram_text = (f"{train_allocated/2**30:.3f} GiB" if train_allocated is not None
                       else "unavailable")
    scope_text, proposed_settings_match = training_budget_scope(
        config, PROJECT_ROOT / "configs" / "baseline.yaml")
    budget_text, supported_budget = budget_recommendation_text(
        recommended_budget, budget_rationale,
        [point[0] for point in metric_trend], plateau_evidence)
    parameter_text = (f"From source iteration 500 to {selected:,}, saved coarse tensors changed "
                      f"{changes['coarse_model']['changed_tensors']}/"
                      f"{changes['coarse_model']['total_tensors']} (L2 "
                      f"{changes['coarse_model']['l2_change']:.6f}) and saved fine tensors "
                      f"changed {changes['fine_model']['changed_tensors']}/"
                      f"{changes['fine_model']['total_tensors']} (L2 "
                      f"{changes['fine_model']['l2_change']:.6f}). This proves changed "
                      "saved parameters; Step 3 gradient tests provide separate update-path evidence."
                      if changes else "Selected checkpoint is the iteration-500 source, so no post-resume parameter-change comparison exists.")
    checkpoint_state_text = (f"Present source and selected checkpoint configs both exactly "
                             "match the immutable run snapshot. Adam step counters span "
                             f"all {continuity['adam_parameter_states'][500]} parameter "
                             f"states at source iteration 500 and all "
                             f"{continuity['adam_parameter_states'][selected]} at selected "
                             f"iteration {selected:,}; every counter equals its saved "
                             "iteration. Scheduler last_epoch is "
                             f"{continuity['scheduler_epochs'][500]} at source and "
                             f"{continuity['scheduler_epochs'][selected]} at selected. "
                             "These are checks of serialized state, not direct observation "
                             "of every intervening optimizer update.")
    sampling_text = ", ".join(
        f"{key}: {fine_fraction:.1%} of fine draws in {span_fraction:.1%} of near–far span"
        for key, (fine_fraction, span_fraction) in concentration.items())
    sampling_stages = list(dict.fromkeys((500, selected)))
    workload_note = (performance_workload_note or
                     "Concurrent GPU workload was not recorded for this report; timings reflect the observed conditions.")
    test_text = _pytest_summary(pytest_xml)
    readiness_issues = []
    if not supported_budget:
        readiness_issues.append("no fixed budget is supported by a late matched-view plateau and an evaluated checkpoint")
    if not plateau_evidence:
        readiness_issues.append("the matched-view validation curve lacks sustained measured diminishing returns")
    if not proposed_settings_match:
        readiness_issues.append("the proposed formal training settings differ from this run or are unrecorded; readiness for that configuration needs matched baseline convergence")
    if possible_overfit:
        readiness_issues.append("a possible overfitting pattern needs investigation")
    if progress_missing or novel_missing:
        readiness_issues.append("requested staged or test full-image panels are missing")
    if len(latest_train) < 1 or len(latest_val) < 3 or len(latest_test) < 3:
        readiness_issues.append("selected checkpoint lacks three validation/test and one train full-view evaluations")
    if not set(LONGITUDINAL_TEST_VIEWS).issubset({r["view_index"] for r in latest_test}):
        readiness_issues.append("selected checkpoint lacks fixed test views 0, 1, 2 for longitudinal comparison")
    if len(full_split_steps) < 3 or any(
        not set(LONGITUDINAL_TEST_VIEWS).issubset({r["view_index"] for r in rows_for(evaluation, step, split)})
        for step in full_split_steps[-3:] for split in ("train", "val", "test")
    ):
        readiness_issues.append("fewer than three shared checkpoints have fixed views 0, 1, 2 in every split")
    if pytest_xml is None:
        readiness_issues.append("Step 4 test evidence was not supplied")
    if not resume_evidence:
        readiness_issues.append("Step 4 resume from iteration 500 is not evidenced in the train log")
    if not changes:
        readiness_issues.append("selected checkpoint does not follow the iteration-500 source for parameter-change evidence")
    if status_iteration != selected:
        readiness_issues.append("saved training status does not identify the selected checkpoint as the last completed iteration")
    if pytest_xml is not None:
        suite_root = ET.parse(pytest_xml).getroot()
        suite = suite_root if suite_root.tag == "testsuite" else suite_root.find("testsuite")
        if suite is None or int(suite.get("failures", "0")) + int(suite.get("errors", "0")):
            readiness_issues.append("Step 4 pytest has failures/errors or an unreadable suite")
    readiness = ("Step 5 readiness for future L runs matching this baseline's fixed training settings is supported by this report's completed checkpoints, split metrics, tests, and selected fixed budget."
                 if not readiness_issues else
                 "Step 5 readiness is **not yet established**: " + "; ".join(readiness_issues) + ".")
    run_rel = run_directory.relative_to(PROJECT_ROOT).as_posix() if run_directory.is_relative_to(PROJECT_ROOT) else str(run_directory)
    report = f"""# Step 4 baseline convergence report

This report uses the single Lego baseline with position L={config['position_encoding_L']}, direction L={config['direction_encoding_L']}, and seed={config['random_seed']}. It is **not** a positional-encoding sweep or an EE result. Source run: `{run_rel}`. Saved status: `{status.get('status', 'unknown')}` at iteration {status_iteration if status_iteration is not None else 'unknown'}; saved checkpoints: {checkpoints}. {status_explanation}

## Training and held-out evaluation

Resume evidence in `train.log`: **{'found' if resume_evidence else 'not found'}** for loading a completed iteration-500 checkpoint. {checkpoint_state_text} The current final observed iteration is **{selected:,}** of {len(training):,} logged iterations. First recorded total loss: **{training[0]['total_loss']:.6f}**; at selected iteration: **{selected_row['total_loss']:.6f}**. The `metrics.csv` PSNR column is computed on sampled training rays, so it is not used as held-out reconstruction accuracy. Elapsed training time through this iteration: **{selected_row['elapsed_time']/3600:.2f} h**. Training post-step allocated VRAM at this iteration: **{train_vram_text}**; peak render VRAM is reported separately.

Fixed run settings from the immutable config snapshot: batch size {config['batch_size']}, Adam initial learning rate {config['learning_rate']}, {config['num_coarse_samples']} coarse + {config['num_fine_samples']} fine samples, {config['network_depth']} layers × width {config['network_width']} per position trunk, density initial bias {config['density_initial_bias']}, white background {config['white_background']}, ray direction normalization {config['normalize_ray_directions']}. No architecture change is introduced by this artifact generator. {scope_text}

Matched validation curve uses exact held-out view indices **{val_views}** at every plotted checkpoint. The source scene has disjoint train/validation/test image paths with counts {split_counts}; full-view split metrics are kept separate in `evaluation_results.csv`.

### Staged checkpoint ledger

{_checkpoint_table(run_directory, training, evaluation, val_views, selected)}

Stage peak VRAM is read from checkpoint telemetry JSON when available. Post-step GPU allocation is the training CSV's point measurement; it does not stand in for a peak. The saved preview is the existing 100×100 diagnostic, while validation PSNR above comes only from complete source-resolution views.

{chr(10).join(trend_lines)}

{chr(10).join(rate_lines)}

{chr(10).join(paired_lines)}

{changes_text} {convergence}

The proposed practical-convergence screen requires two late validation intervals with nonnegative per-1,000-iteration gains no faster than every preceding interval in PSNR, SSIM and LPIPS, plus at least one matched view without improvement for each metric in each late interval. This uses no preselected absolute metric threshold and is deliberately conservative. {training_loss_interval_trend(training, [point[0] for point in metric_trend])} The researcher must still review image quality, split behavior and cost; three matched views do not establish universal convergence. {budget_text}

## Train / validation / test

{_metric_table(evaluation, eval_steps)}

{overfit} Common full-view train/validation/test checkpoints: {full_split_steps}. Selected-checkpoint view counts: train={len(latest_train)}, validation={len(latest_val)}, test={len(latest_test)}. All selected test views {sorted(r['view_index'] for r in latest_test)} mean: {('PSNR %.3f dB, SSIM %.4f, LPIPS %.4f' % tuple(statistics.mean(float(r[name]) for r in latest_test) for name in METRIC_NAMES)) if latest_test else 'unavailable'}. {novel_test_mean} {fixed_test_mean}

## Reconstruction and sampling evidence

Progress panels request iterations {stages} for validation views [0, 1, 2]. Missing checkpoint/view full-image panels: {progress_missing if progress_missing else 'none'}. Missing requested stages: {missing_stages if missing_stages else 'none'}. Test novel-view figure uses spatially separated held-out indices {list(NOVEL_TEST_VIEWS)} at checkpoint {selected:,}; missing full-image panels: {novel_missing if novel_missing else 'none'}. Fixed test views {list(LONGITUDINAL_TEST_VIEWS)} remain the longitudinal subset for matched-checkpoint overfitting analysis. The final test summary/table may include both sets; its mean then describes all listed test views, while the overfitting trend uses the matched fixed subset. PNGs are 8-bit displays; all metric calculations use unquantized float RGB.

Full-image PSNR and SSIM include the large white-background area. Good scores there can coexist with visible Lego object-detail errors, so the prediction and absolute-difference panels must be examined alongside scalar metrics; this pipeline does not claim an object-mask score.

The hierarchical diagnostic evaluates the same held-out validation view 0 pixels {sampling_pixels} with deterministic sampling at checkpoints {sampling_stages}. Each plot shows coarse rendering weights at coarse sample depths, fine rendering weights on merged depths, a lower gray rug for coarse depths, and an upper green rug for fine depth draws. Fractions of fine samples inside midpoint bins crossing the 10th–90th percentiles of cumulative coarse rendering weight: {sampling_text or 'unavailable'}. Midpoint bin edges keep the interval nonzero even if one sample carries almost all weight. These weight-defined intervals indicate learned sampler focus, **not labeled ground-truth surface depth**. Fine draws come from the coarse weight distribution, so a high fraction inside an interval defined by those weights is partly an expected algorithmic result. Its narrowing does not independently prove correct geometry or a reconstruction advantage for the fine network. {parameter_text}

## Performance and reproducibility

{hardware} {perf} {peak_text} {workload_note} These are engineering observations, not an answer to the research question. Run configuration remains at `config_snapshot.yaml`; evaluation details are in `evaluation/metric_metadata.json`. {provenance_text} Test suite: {test_text}

## Artifacts and limits

- `artifacts/baseline_metrics_curve.png`
- `artifacts/baseline_reconstruction_progress.png`
- `artifacts/novel_view_validation.png`
- `artifacts/hierarchical_sampling_after_training.png`
- `artifacts/evaluation_metric_report.md`
- `artifacts/baseline_performance_report.md`
- `{run_rel}/evaluation_results.csv`

{panel_status} Remaining limits include the small matched held-out view set and any unobserved future convergence or overfitting. {readiness}
"""
    (artifacts / "baseline_convergence_report.md").write_text(report, encoding="utf-8")
    metric_report = f"""# Step 4 evaluation metric report

Source: `{run_rel}/evaluation_results.csv`. Rows identify checkpoint iteration, dataset split and original view index. Full native-resolution [H,W,3] RGB predictions and white-composited source targets are scored **before PNG quantization**. Image metrics are per full view; mean, median and population standard deviation are computed within one checkpoint and split. Training `metrics.csv` PSNR is a separate sampled-ray optimization statistic.

## Locked implementation metadata

{_metadata_report(metadata)}

## Per-checkpoint aggregates

{_metric_table(evaluation, eval_steps)}

Validation comparison curve uses matched view indices {val_views}. PSNR and SSIM are better when higher; LPIPS is better when lower. No cross-split pooled mean is reported. Individual per-view rows and render performance are preserved in the CSV.

The white-composited background contributes to full-image PSNR and SSIM. Inspect the absolute-difference figures for object-detail failures that aggregate scores may obscure; no object-mask score is reported.

For test comparisons over time, the fixed longitudinal subset is {list(LONGITUDINAL_TEST_VIEWS)}. The final novel-view figure instead shows the spatially separated subset {list(NOVEL_TEST_VIEWS)}. The final test aggregate may include their union; do not interpret it as the matched-view longitudinal trend.
"""
    (artifacts / "evaluation_metric_report.md").write_text(metric_report, encoding="utf-8")
    performance_report = f"""# Step 4 baseline render performance

Source: `{run_rel}/evaluation_results.csv` at selected iteration {selected:,}. {hardware} {perf} {peak_text} {workload_note}

{_performance_table(evaluation, selected)}

Each CSV row records its own wall-clock render time, rays/s, peak allocated VRAM, native width and height, and `render_chunk_size`. Statistics above span only the selected-checkpoint rows. Training `metrics.csv` records post-step allocated/reserved memory and training rays/s, which are distinct measurements. This performance section is an engineering observation only and is not used to interpret positional-encoding accuracy.
"""
    (artifacts / "baseline_performance_report.md").write_text(performance_report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--selected-iteration", type=int, required=True)
    parser.add_argument("--evaluation-csv", type=Path)
    parser.add_argument("--metric-metadata", type=Path)
    parser.add_argument("--pytest-xml", type=Path)
    parser.add_argument("--artifact-directory", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--recommended-budget", type=int,
                        help="fixed budget chosen after reviewing the measured baseline curve")
    parser.add_argument("--budget-rationale",
                        help="specific metric evidence supporting --recommended-budget")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="write an explicitly incomplete progress report while evaluations are pending")
    parser.add_argument("--performance-workload-note",
                        help="observed concurrent-workload condition for selected-checkpoint renders")
    args = parser.parse_args()
    run = args.run.resolve()
    evaluation_path = (args.evaluation_csv or run / "evaluation_results.csv").resolve()
    metadata_path = (args.metric_metadata or run / "evaluation" / "metric_metadata.json").resolve()
    artifacts = args.artifact_directory.resolve()
    config = yaml.safe_load((run / "config_snapshot.yaml").read_text(encoding="utf-8"))
    assert_run_config_compatible(run, config)
    if config["scene"] != "lego" or config["position_encoding_L"] != 10 or config["direction_encoding_L"] != 4:
        raise ValueError("Step 4 report requires the fixed Lego L=10/direction-L=4 baseline")
    training = read_training_metrics(run / "metrics.csv")
    evaluation = read_evaluation_metrics(evaluation_path)
    summary_path = run / "evaluation_summary.csv"
    if summary_path.is_file():
        verify_evaluation_summary(summary_path, evaluation)
    elif not args.allow_incomplete:
        raise FileNotFoundError("final report requires evaluation_summary.csv")
    selected = args.selected_iteration
    if selected < 500 or selected > len(training):
        raise ValueError("selected iteration must have training metrics and be at least 500")
    # A progress report for an earlier checkpoint must not silently include
    # later training or evaluation measurements in the same active run.
    training = training[:selected]
    evaluation = [row for row in evaluation if row["iteration"] <= selected]
    if not (run / "checkpoints" / f"iter_{selected:06d}.pt").is_file():
        raise FileNotFoundError("selected checkpoint is absent")
    if not rows_for(evaluation, selected, "val"):
        raise ValueError("selected checkpoint requires full-view validation metrics")
    if not args.allow_incomplete and not rows_for(evaluation, selected, "test"):
        raise ValueError("selected checkpoint requires full-view test metrics unless --allow-incomplete")
    if (args.recommended_budget is None) != (args.budget_rationale is None):
        raise ValueError("recommended budget and data-based rationale must be supplied together")
    if args.recommended_budget is not None and (
        args.recommended_budget < 500 or args.recommended_budget > selected or
        len({r["iteration"] for r in evaluation if r["split"] == "val"}) < 3
    ):
        raise ValueError("budget needs at least three validation checkpoints and cannot exceed measured training")
    stages = list(dict.fromkeys((500, 2000, 5000, 10000, selected)))
    if not args.allow_incomplete:
        missing = []
        for step in stages:
            for view_index in (0, 1, 2):
                row = next((r for r in evaluation if r["iteration"] == step and
                            r["split"] == "val" and r["view_index"] == view_index), None)
                if _missing_images(run, row):
                    missing.append((step, "val", view_index))
        for view_index in NOVEL_TEST_VIEWS:
            row = next((r for r in evaluation if r["iteration"] == selected and
                        r["split"] == "test" and r["view_index"] == view_index), None)
            if _missing_images(run, row):
                missing.append((selected, "test", view_index))
        if missing:
            raise ValueError(f"final report requires all requested full-image renders: {missing}")
        if not rows_for(evaluation, selected, "train"):
            raise ValueError("final report requires selected-checkpoint full-view train metrics")
        selected_test_indices = {r["view_index"] for r in rows_for(evaluation, selected, "test")}
        if not set(LONGITUDINAL_TEST_VIEWS).issubset(selected_test_indices):
            raise ValueError("final report requires fixed test views 0, 1, 2 for longitudinal analysis")
        _, _, common_split_steps = overfitting_assessment(evaluation)
        if len(common_split_steps) < 3:
            raise ValueError("final report requires at least three checkpoints with train/val/test full-view metrics")
        for step in common_split_steps[-3:]:
            for split in ("train", "val", "test"):
                indices = {r["view_index"] for r in rows_for(evaluation, step, split)}
                if not set(LONGITUDINAL_TEST_VIEWS).issubset(indices):
                    raise ValueError(
                        f"final report requires fixed views 0, 1, 2 for {split} at checkpoint {step}"
                    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else None
    if not args.allow_incomplete and metadata is None:
        raise FileNotFoundError("final report requires locked metric_metadata.json")
    if not args.allow_incomplete and args.pytest_xml is None:
        raise ValueError("final report requires --pytest-xml")
    if args.pytest_xml is not None:
        suite_root = ET.parse(args.pytest_xml).getroot()
        suite = suite_root if suite_root.tag == "testsuite" else suite_root.find("testsuite")
        if suite is None:
            raise ValueError("pytest XML has no testsuite")
        if not args.allow_incomplete and (
            int(suite.get("failures", "0")) or int(suite.get("errors", "0"))
        ):
            raise ValueError("final report requires a passing pytest suite")
    continuity = verify_checkpoint_continuity(run, config, 500, selected)
    # Check hashes before replacing any figure/report. Backfilled hashes retain
    # their narrower historical evidentiary scope in the report text.
    provenance_text = checkpoint_provenance_summary(run, (500, selected))
    artifacts.mkdir(parents=True, exist_ok=True)
    val_views = save_metrics_curve(training, evaluation, artifacts / "baseline_metrics_curve.png")
    progress_missing = save_progress(run, evaluation, stages, [0, 1, 2],
                                     artifacts / "baseline_reconstruction_progress.png")
    novel_missing = save_novel_views(run, evaluation, selected, list(NOVEL_TEST_VIEWS),
                                     artifacts / "novel_view_validation.png")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    pixels, concentration = save_sampling_diagnostic(
        run, config, selected, artifacts / "hierarchical_sampling_after_training.png", device=device)
    write_reports(run, artifacts, training, evaluation, selected, stages, val_views,
                  progress_missing, novel_missing, pixels, concentration, metadata,
                  continuity, provenance_text,
                  args.pytest_xml, args.recommended_budget, args.budget_rationale,
                  args.performance_workload_note)
    print(json.dumps({"run": str(run), "selected_iteration": selected,
                      "validation_curve_views": val_views,
                      "missing_progress_panels": progress_missing,
                      "missing_test_views": novel_missing,
                      "artifacts": str(artifacts)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
