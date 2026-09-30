"""Full-resolution, split-qualified reconstruction evaluation for Step 4.

All image metrics use float RGB in [0, 1] before visualization quantization.
The existing deterministic renderer preserves every original Blender pixel.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import statistics
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import numpy as np
import torch
from skimage.metrics import structural_similarity

from .dataset import BlenderScene, View
from .pipeline import CoarseFineNeRF
from .preview import render_image


SplitName = Literal["train", "val", "test"]
EVALUATION_FIELDS = (
    "iteration", "split", "view_index", "psnr", "ssim", "lpips",
    "render_seconds", "rays_per_second", "peak_vram_bytes", "width", "height",
    "render_chunk_size",
)
SUMMARY_FIELDS = (
    "iteration", "split", "view_count", "view_indices",
    "psnr_mean", "psnr_median", "psnr_std",
    "ssim_mean", "ssim_median", "ssim_std",
    "lpips_mean", "lpips_median", "lpips_std",
)
_RGB_ROUNDOFF_TOLERANCE = 1e-6


def _checked_images(
    prediction: torch.Tensor, ground_truth: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.ndim != 3 or prediction.shape[-1] != 3 or prediction.shape != ground_truth.shape:
        raise ValueError("prediction and ground truth must be matching [H, W, 3] RGB images")
    for name, image in (("prediction", prediction), ("ground truth", ground_truth)):
        if not bool(torch.isfinite(image).all()):
            raise ValueError(f"{name} must contain finite RGB values")
        if bool(((image < -_RGB_ROUNDOFF_TOLERANCE) |
                 (image > 1 + _RGB_ROUNDOFF_TOLERANCE)).any()):
            raise ValueError(f"{name} RGB values must be in [0, 1] within numerical tolerance")
    # White-background accumulation may overshoot 1 by a few float32 ULPs.
    # Clip only this bounded roundoff before every scientific image metric.
    return prediction.clamp(0, 1), ground_truth.clamp(0, 1)


def compute_psnr(prediction: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """Fine-image RGB MSE PSNR with Step 3's 1e-12 zero-error floor."""
    prediction, ground_truth = _checked_images(prediction, ground_truth)
    mse = float(torch.mean((prediction.float() - ground_truth.float()).square()))
    return -10.0 * math.log10(max(mse, 1e-12))


def compute_ssim(prediction: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """scikit-image SSIM, RGB channel average, 7-pixel uniform window, range 1."""
    prediction, ground_truth = _checked_images(prediction, ground_truth)
    if min(prediction.shape[:2]) < 7:
        raise ValueError("SSIM requires image height and width of at least 7 pixels")
    target = ground_truth.detach().cpu().numpy().astype(np.float64, copy=False)
    estimate = prediction.detach().cpu().numpy().astype(np.float64, copy=False)
    return float(structural_similarity(
        target, estimate, data_range=1.0, channel_axis=-1,
        win_size=7, gaussian_weights=False, use_sample_covariance=True,
        K1=0.01, K2=0.03,
    ))


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LPIPSMetric:
    """Official calibrated LPIPS v0.1 with AlexNet, on [0, 1] RGB inputs."""

    def __init__(self, device: torch.device | str = "cpu", *, network: torch.nn.Module | None = None) -> None:
        self.device = torch.device(device)
        if network is None:
            import lpips

            network = lpips.LPIPS(net="alex", version="0.1", pretrained=True, verbose=False)
        self.network = network.to(self.device).eval()
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, prediction: torch.Tensor, ground_truth: torch.Tensor) -> float:
        prediction, ground_truth = _checked_images(prediction, ground_truth)
        inputs = [image.detach().to(self.device, dtype=torch.float32).permute(2, 0, 1)[None]
                  for image in (prediction, ground_truth)]
        # The official forward(normalize=True) converts [0, 1] to [-1, 1].
        output = self.network(inputs[0], inputs[1], normalize=True)
        if not isinstance(output, torch.Tensor) or output.numel() != 1:
            raise ValueError("LPIPS backend must return one scalar per full image")
        value = float(output.item())
        if not math.isfinite(value):
            raise ValueError("LPIPS backend returned a nonfinite score")
        return value

    def metadata(self) -> dict[str, object]:
        """Record installed implementations and the actual local weight hashes."""
        import lpips
        from torchvision.models import AlexNet_Weights

        linear_path = Path(lpips.__file__).parent / "weights" / "v0.1" / "alex.pth"
        trunk_url = AlexNet_Weights.IMAGENET1K_V1.url
        trunk_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(trunk_url).path).name
        return {
            "implementation": "richzhang/PerceptualSimilarity lpips.LPIPS",
            "package_version": version("lpips"),
            "backbone": "alex",
            "model_version": "0.1",
            "calibrated_linear_weights_source": "bundled lpips/weights/v0.1/alex.pth",
            "calibrated_linear_weights_path": str(linear_path),
            "calibrated_linear_weights_sha256": _sha256(linear_path),
            "imagenet_backbone_weights_source": trunk_url,
            "imagenet_backbone_weights_path": str(trunk_path),
            "imagenet_backbone_weights_sha256": _sha256(trunk_path),
            "input": "float32 RGB [0,1], HWC to NCHW, official normalize=True converts to [-1,1]",
            "device": str(self.device),
        }


class ImageMetricSuite:
    def __init__(self, lpips_metric: LPIPSMetric) -> None:
        self.lpips_metric = lpips_metric

    def compute(self, prediction: torch.Tensor, ground_truth: torch.Tensor) -> dict[str, float]:
        return {
            "psnr": compute_psnr(prediction, ground_truth),
            "ssim": compute_ssim(prediction, ground_truth),
            "lpips": self.lpips_metric(prediction, ground_truth),
        }

    def metadata(self) -> dict[str, object]:
        return {
            "image_space": "original-resolution, white-composited RGB float [0,1]",
            "rgb_roundoff_convention": {
                "accepted_excursion_beyond_0_or_1": _RGB_ROUNDOFF_TOLERANCE,
                "action": "clamp both prediction and ground truth to [0,1] before PSNR, SSIM and LPIPS; reject larger excursions",
            },
            "psnr": {"implementation": "-10 log10(mean RGB squared error)", "zero_mse_floor": 1e-12,
                     "aggregation": "mean of per-view dB scores"},
            "ssim": {"implementation": "skimage.metrics.structural_similarity",
                     "package_version": version("scikit-image"), "data_range": 1.0,
                     "channel_axis": -1, "win_size": 7, "gaussian_weights": False,
                     "use_sample_covariance": True, "K1": 0.01, "K2": 0.03},
            "lpips": self.lpips_metric.metadata(),
            "aggregate_std": "population standard deviation, ddof=0, across views within one split",
            "performance_peak_vram_scope": "PyTorch peak allocated GPU memory measured during rendering only, before LPIPS",
        }


@dataclass(frozen=True)
class ViewMetrics:
    iteration: int
    split: SplitName
    view_index: int
    psnr: float
    ssim: float
    lpips: float
    render_seconds: float
    rays_per_second: float
    peak_vram_bytes: int | None
    width: int
    height: int
    render_chunk_size: int


@dataclass(frozen=True)
class EvaluationResult:
    iteration: int
    split: SplitName
    views: tuple[ViewMetrics, ...]
    summary: dict[str, dict[str, float]]


def aggregate_evaluation(views: Sequence[ViewMetrics]) -> dict[str, dict[str, float]]:
    """Keep aggregation scoped to one iteration and one data split."""
    if not views:
        raise ValueError("at least one evaluated view is required")
    if len({(item.iteration, item.split) for item in views}) != 1:
        raise ValueError("cannot aggregate across different iterations or splits")
    result = {}
    for key in ("psnr", "ssim", "lpips"):
        values = [float(getattr(item, key)) for item in views]
        result[key] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "std": statistics.pstdev(values),
        }
    return result


def evaluate_model(
    model: CoarseFineNeRF,
    scene: BlenderScene,
    split: SplitName,
    view_indices: Sequence[int],
    *,
    metrics: ImageMetricSuite,
    render_chunk_size: int,
    iteration: int,
    csv_path: Path | None = None,
    on_view: Callable[[View, torch.Tensor, ViewMetrics], None] | None = None,
) -> EvaluationResult:
    """Render every pixel of selected views and score fine predictions.

    `view_indices` selects camera views, never pixels. Call separately for
    train, validation and test to prevent accidental split mixing.
    """
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val or test")
    if scene.name != model.config["scene"]:
        raise ValueError("scene differs from the model's frozen configuration")
    if not view_indices or any(isinstance(index, bool) or not isinstance(index, int) for index in view_indices):
        raise ValueError("view_indices must contain integer indices")
    if len(set(view_indices)) != len(view_indices):
        raise ValueError("view_indices must be unique")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("iteration must be a nonnegative integer")
    if isinstance(render_chunk_size, bool) or not isinstance(render_chunk_size, int) or render_chunk_size <= 0:
        raise ValueError("render_chunk_size must be a positive integer")
    data_split = getattr(scene, split)
    if any(index < 0 or index >= len(data_split) for index in view_indices):
        raise IndexError(f"{split} view index is out of range")
    device = next(model.parameters()).device
    rows: list[ViewMetrics] = []
    for index in view_indices:
        view = data_split.get_view(index)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        prediction = render_image(
            model, view.camera_to_world, view.intrinsics,
            (view.intrinsics.width, view.intrinsics.height),
            render_chunk_size=render_chunk_size,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        peak_vram = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        scores = metrics.compute(prediction, view.image)
        count = view.intrinsics.width * view.intrinsics.height
        row = ViewMetrics(
            iteration=iteration, split=split, view_index=index,
            psnr=scores["psnr"], ssim=scores["ssim"], lpips=scores["lpips"],
            render_seconds=elapsed, rays_per_second=count / max(elapsed, 1e-12),
            peak_vram_bytes=peak_vram, width=view.intrinsics.width,
            height=view.intrinsics.height, render_chunk_size=render_chunk_size,
        )
        rows.append(row)
        if on_view is not None:
            on_view(view, prediction, row)
    result = EvaluationResult(iteration, split, tuple(rows), aggregate_evaluation(rows))
    if csv_path is not None:
        export_evaluation_csv(csv_path, result)
    return result


def export_evaluation_csv(path: Path, result: EvaluationResult) -> None:
    """Atomically upsert split/view/iteration rows without duplicate records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[tuple[int, str, int], dict[str, str]] = {}
    if path.is_file():
        with path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != EVALUATION_FIELDS:
                raise ValueError("evaluation CSV header differs from the fixed schema")
            for row in reader:
                if set(row) != set(EVALUATION_FIELDS) or any(value is None for value in row.values()):
                    raise ValueError("evaluation CSV contains a malformed row")
                key = (int(row["iteration"]), row["split"], int(row["view_index"]))
                if key in existing:
                    raise ValueError("evaluation CSV has duplicate split/view/iteration rows")
                existing[key] = row
    for item in result.views:
        key = (item.iteration, item.split, item.view_index)
        row = asdict(item)
        row["peak_vram_bytes"] = "" if item.peak_vram_bytes is None else item.peak_vram_bytes
        if key in existing:
            for metric_name in ("psnr", "ssim", "lpips"):
                if not math.isclose(float(existing[key][metric_name]), getattr(item, metric_name), abs_tol=1e-4):
                    raise ValueError(f"reevaluation changed {metric_name} for {key}")
            # Keep the first performance observation, which can vary by run.
            continue
        existing[key] = row
    descriptor, temporary_name = tempfile.mkstemp(prefix=".evaluation_", suffix=".csv", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=EVALUATION_FIELDS)
            writer.writeheader()
            for key in sorted(existing):
                writer.writerow(existing[key])
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def export_summary_csv(
    path: Path, result: EvaluationResult, *, per_view_csv_path: Path | None = None
) -> None:
    """Atomically upsert one aggregate row per checkpoint and split.

    When a per-view CSV is supplied, aggregate all persisted views for this
    checkpoint/split so evaluating another subset cannot shrink the summary.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[tuple[int, str], dict[str, str | int | float]] = {}
    if path.is_file():
        with path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            source_fields = tuple(reader.fieldnames or ())
            legacy_fields = tuple(field for field in SUMMARY_FIELDS if field != "view_indices")
            if source_fields not in (SUMMARY_FIELDS, legacy_fields):
                raise ValueError("evaluation summary CSV header differs from the fixed schema")
            for row in reader:
                if set(row) != set(source_fields) or any(value is None for value in row.values()):
                    raise ValueError("evaluation summary CSV contains a malformed row")
                key = (int(row["iteration"]), row["split"])
                if key in existing:
                    raise ValueError("evaluation summary CSV has duplicate checkpoint/split rows")
                existing[key] = row
        if source_fields == legacy_fields:
            if per_view_csv_path is None:
                raise ValueError("legacy evaluation summary migration requires the per-view CSV")
            with Path(per_view_csv_path).open("r", newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if tuple(reader.fieldnames or ()) != EVALUATION_FIELDS:
                    raise ValueError("evaluation CSV header differs from the fixed schema")
                indices: dict[tuple[int, str], list[int]] = {key: [] for key in existing}
                for view_row in reader:
                    key = (int(view_row["iteration"]), view_row["split"])
                    if key in indices:
                        indices[key].append(int(view_row["view_index"]))
            for key, old_row in existing.items():
                if len(indices[key]) != int(old_row["view_count"]):
                    raise ValueError(f"legacy summary view count differs from per-view CSV for {key}")
                old_row["view_indices"] = ";".join(str(index) for index in sorted(indices[key]))
    summary_views = result.views
    if per_view_csv_path is not None:
        with Path(per_view_csv_path).open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != EVALUATION_FIELDS:
                raise ValueError("evaluation CSV header differs from the fixed schema")
            matching = [row for row in reader if int(row["iteration"]) == result.iteration
                        and row["split"] == result.split]
        if not matching:
            raise ValueError("no persisted per-view rows for summary")
        summary_views = tuple(ViewMetrics(
            iteration=int(row["iteration"]), split=row["split"],
            view_index=int(row["view_index"]), psnr=float(row["psnr"]),
            ssim=float(row["ssim"]), lpips=float(row["lpips"]),
            render_seconds=float(row["render_seconds"]),
            rays_per_second=float(row["rays_per_second"]),
            peak_vram_bytes=(int(row["peak_vram_bytes"]) if row["peak_vram_bytes"] else None),
            width=int(row["width"]), height=int(row["height"]),
            render_chunk_size=int(row["render_chunk_size"]),
        ) for row in matching)
    summary_values = aggregate_evaluation(summary_views)
    row: dict[str, str | int | float] = {
        "iteration": result.iteration, "split": result.split, "view_count": len(summary_views),
        "view_indices": ";".join(str(index) for index in sorted(item.view_index for item in summary_views)),
    }
    for metric_name, summary in summary_values.items():
        for statistic_name, value in summary.items():
            row[f"{metric_name}_{statistic_name}"] = value
    existing[(result.iteration, result.split)] = row
    descriptor, temporary_name = tempfile.mkstemp(prefix=".evaluation_summary_", suffix=".csv", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            for key in sorted(existing):
                writer.writerow(existing[key])
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
