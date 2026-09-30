"""Formal full-image metrics, split isolation, CSV export and checkpoint loading."""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_nerf  # noqa: E402
from nerf_step1.dataset import load_blender_scene  # noqa: E402
from nerf_step1.evaluation import (  # noqa: E402
    EvaluationResult, ImageMetricSuite, LPIPSMetric, aggregate_evaluation, compute_psnr, compute_ssim,
    evaluate_model, export_evaluation_csv, export_summary_csv,
)
from nerf_step1.pipeline import CoarseFineNeRF, make_scheduler  # noqa: E402
from nerf_step1.preview import render_image  # noqa: E402
from nerf_step1.training_io import create_training_run, save_checkpoint  # noqa: E402
from test_pipeline import tiny_config  # noqa: E402


class FakeLPIPS(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[tuple[int, ...], bool]] = []

    def forward(self, prediction, ground_truth, *, normalize=False):
        self.calls.append((tuple(prediction.shape), normalize))
        return (prediction - ground_truth).abs().mean().reshape(1, 1, 1, 1)


def _metric_suite() -> ImageMetricSuite:
    return ImageMetricSuite(LPIPSMetric("cpu", network=FakeLPIPS()))


def _scene(tmp_path: Path):
    scene_directory = tmp_path / "data" / "lego"
    pose = np.eye(4).tolist()
    for split_name, red in (("train", 64), ("val", 128), ("test", 192)):
        frames = []
        for index in range(2):
            frame = f"{split_name}/r_{index}"
            path = scene_directory / f"{frame}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            pixels = np.zeros((8, 8, 4), dtype=np.uint8)
            pixels[..., :3] = (red + index, 24, 48)
            pixels[..., 3] = 255
            Image.fromarray(pixels).save(path)
            frames.append({"file_path": frame, "transform_matrix": pose})
        (scene_directory / f"transforms_{split_name}.json").write_text(
            json.dumps({"camera_angle_x": 0.7, "frames": frames}), encoding="utf-8"
        )
    return load_blender_scene(tmp_path / "data", "lego")


def _model() -> CoarseFineNeRF:
    return CoarseFineNeRF(tiny_config())


def test_psnr_and_ssim_formulas_and_input_range() -> None:
    reference = torch.ones(8, 8, 3)
    estimate = torch.full_like(reference, 0.5)
    assert compute_psnr(estimate, reference) == pytest.approx(6.0205999, abs=1e-5)
    assert compute_psnr(reference, reference) == pytest.approx(120.0)
    assert compute_ssim(reference, reference) == pytest.approx(1.0)
    assert compute_ssim(estimate, reference) < 1.0
    slight_overshoot = reference + 2.4e-7
    assert compute_psnr(slight_overshoot, reference) == pytest.approx(120.0)
    assert compute_ssim(slight_overshoot, reference) == pytest.approx(1.0)
    assert _metric_suite().lpips_metric(slight_overshoot, reference) == pytest.approx(0.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        compute_ssim(estimate + 1, reference)


def test_lpips_wrapper_uses_official_normalize_option_and_rgb_layout() -> None:
    backend = FakeLPIPS()
    metric = LPIPSMetric("cpu", network=backend)
    a = torch.zeros(8, 8, 3)
    b = torch.ones_like(a)
    assert metric(a, b) == pytest.approx(1.0)
    assert backend.calls == [((1, 3, 8, 8), True)]
    with pytest.raises(ValueError, match="matching"):
        metric(a, b[:7])


def test_full_image_evaluation_is_chunked_and_split_qualified(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    model = _model()
    calls: list[int] = []
    original_forward = model.forward

    def counted_forward(origins, directions, **kwargs):
        calls.append(len(origins))
        assert kwargs["randomized"] is False
        return original_forward(origins, directions, **kwargs)

    model.forward = counted_forward
    csv_path = tmp_path / "evaluation_results.csv"
    seen: list[tuple[str, int, tuple[int, ...]]] = []
    result = evaluate_model(
        model, scene, "val", [0, 1], metrics=_metric_suite(),
        render_chunk_size=11, iteration=500, csv_path=csv_path,
        on_view=lambda view, prediction, row: seen.append(
            (view.split, row.view_index, tuple(prediction.shape))
        ),
    )
    assert result.split == "val" and len(result.views) == 2
    assert all(item.width == item.height == 8 and item.render_chunk_size == 11
               for item in result.views)
    assert seen == [("val", 0, (8, 8, 3)), ("val", 1, (8, 8, 3))]
    assert sum(calls) == 2 * 64 and max(calls) <= 11
    assert len(calls) == 12
    assert model.training
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [(row["split"], row["view_index"]) for row in rows] == [("val", "0"), ("val", "1")]
    assert result.summary["psnr"]["mean"] == pytest.approx(
        sum(item.psnr for item in result.views) / 2
    )

    train = evaluate_model(model, scene, "train", [0], metrics=_metric_suite(),
                           render_chunk_size=11, iteration=500, csv_path=csv_path)
    test = evaluate_model(model, scene, "test", [0], metrics=_metric_suite(),
                          render_chunk_size=11, iteration=500, csv_path=csv_path)
    with pytest.raises(ValueError, match="different iterations or splits"):
        aggregate_evaluation((*result.views, *train.views))
    with pytest.raises(ValueError, match="split must"):
        evaluate_model(model, scene, "validation", [0], metrics=_metric_suite(),
                       render_chunk_size=11, iteration=500)
    with pytest.raises(IndexError, match="out of range"):
        evaluate_model(model, scene, "test", [3], metrics=_metric_suite(),
                       render_chunk_size=11, iteration=500)
    assert train.views[0].split == "train" and test.views[0].split == "test"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        assert {(row["split"], row["view_index"]) for row in csv.DictReader(stream)} == {
            ("train", "0"), ("val", "0"), ("val", "1"), ("test", "0")
        }


def test_full_image_render_is_equivalent_across_chunk_sizes(tmp_path: Path) -> None:
    view = _scene(tmp_path).val.get_view(0)
    model = _model()
    model.eval()
    small_chunks = render_image(model, view.camera_to_world, view.intrinsics, (8, 8),
                                render_chunk_size=7)
    one_chunk = render_image(model, view.camera_to_world, view.intrinsics, (8, 8),
                             render_chunk_size=64)
    assert small_chunks.shape == (8, 8, 3)
    torch.testing.assert_close(small_chunks, one_chunk, rtol=1e-6, atol=1e-6)


def test_evaluation_csv_and_summary_upsert_without_duplicate_rows(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    result = evaluate_model(_model(), scene, "val", [0, 1], metrics=_metric_suite(),
                            render_chunk_size=32, iteration=5)
    per_view = tmp_path / "evaluation_results.csv"
    summary = tmp_path / "evaluation_summary.csv"
    export_evaluation_csv(per_view, result)
    export_summary_csv(summary, result)
    original = per_view.read_bytes()
    export_evaluation_csv(per_view, result)
    export_summary_csv(summary, result)
    assert per_view.read_bytes() == original
    with summary.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["view_count"] == "2"
    assert rows[0]["view_indices"] == "0;1"
    assert float(rows[0]["lpips_std"]) == pytest.approx(result.summary["lpips"]["std"])

    # A later command may target one already-scored view. The summary must
    # still describe all persisted views at this checkpoint and split.
    subset = EvaluationResult(
        result.iteration, result.split, result.views[:1], aggregate_evaluation(result.views[:1])
    )
    export_summary_csv(summary, subset, per_view_csv_path=per_view)
    with summary.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["view_count"] == "2"
    assert rows[0]["view_indices"] == "0;1"
    assert float(rows[0]["psnr_mean"]) == pytest.approx(result.summary["psnr"]["mean"])

    # A long-running first evaluator may have written the original Step 4
    # summary header before view_indices was added; migrate from source rows.
    legacy_fields = [field for field in rows[0] if field != "view_indices"]
    with summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=legacy_fields)
        writer.writeheader()
        writer.writerow({field: rows[0][field] for field in legacy_fields})
    export_summary_csv(summary, subset, per_view_csv_path=per_view)
    with summary.open(newline="", encoding="utf-8") as stream:
        migrated = list(csv.DictReader(stream))
    assert migrated[0]["view_indices"] == "0;1"


def test_checkpoint_evaluation_loads_frozen_models_and_exports(tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    scene = _scene(tmp_path)
    config = tiny_config()
    config["dataset_root"] = str(scene.directory.parent)
    model = CoarseFineNeRF(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    scheduler = make_scheduler(optimizer, config)
    run = create_training_run(tmp_path / "runs", "evaluation_test", config)
    checkpoint = run / "checkpoints" / "iter_000003.pt"
    save_checkpoint(checkpoint, coarse_model=model.coarse_model,
                    fine_model=model.fine_model, optimizer=optimizer,
                    scheduler=scheduler, iteration=3, config=config)

    class FakeMetric:
        def __init__(self, device):
            self.metric = LPIPSMetric(device, network=FakeLPIPS())

        def __call__(self, prediction, ground_truth):
            return self.metric(prediction, ground_truth)

        def metadata(self):
            return {"implementation": "test fake", "input": "[0,1]"}

    monkeypatch.setattr(evaluate_nerf, "LPIPSMetric", FakeMetric)
    evaluate_nerf.evaluate_checkpoint(run, 3, "test", [0], device="cpu", render_chunk_size=13)
    with (run / "evaluation_results.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["iteration"] == "3" and rows[0]["split"] == "test"
    assert rows[0]["width"] == rows[0]["height"] == "8"
    assert math.isfinite(float(rows[0]["ssim"]))
    assert (run / "evaluation" / "renders" / "iter_000003" / "test_view_000_prediction.png").is_file()
    assert (run / "evaluation" / "metric_metadata.json").is_file()
    assert (run / "evaluation_summary.csv").is_file()
    with pytest.raises(FileNotFoundError, match="checkpoint does not exist"):
        evaluate_nerf.evaluate_checkpoint(run, 2, "test", [0], device="cpu")
    shutil.copyfile(checkpoint, run / "checkpoints" / "iter_000002.pt")
    with pytest.raises(ValueError, match="filename says"):
        evaluate_nerf.evaluate_checkpoint(run, 2, "test", [0], device="cpu")


def test_checkpoint_evaluation_rejects_scientific_metadata_drift_before_csv_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = _scene(tmp_path)
    config = tiny_config()
    config["dataset_root"] = str(scene.directory.parent)
    model = CoarseFineNeRF(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    scheduler = make_scheduler(optimizer, config)
    run = create_training_run(tmp_path / "runs", "metadata_test", config)
    save_checkpoint(run / "checkpoints" / "iter_000004.pt",
                    coarse_model=model.coarse_model, fine_model=model.fine_model,
                    optimizer=optimizer, scheduler=scheduler, iteration=4, config=config)

    class FakeMetric:
        def __init__(self, device):
            self.metric = LPIPSMetric(device, network=FakeLPIPS())

        def __call__(self, prediction, ground_truth):
            return self.metric(prediction, ground_truth)

        def metadata(self):
            return {
                "implementation": "test LPIPS", "package_version": "0.1.4",
                "backbone": "alex", "model_version": "0.1",
                "calibrated_linear_weights_sha256": "aaa",
                "imagenet_backbone_weights_sha256": "bbb",
                "input": "normalize=True", "device": "cpu",
            }

    monkeypatch.setattr(evaluate_nerf, "LPIPSMetric", FakeMetric)
    evaluate_nerf.evaluate_checkpoint(run, 4, "test", [0], device="cpu", render_chunk_size=13)
    csv_path = run / "evaluation_results.csv"
    original_csv = csv_path.read_bytes()
    metadata_path = run / "evaluation" / "metric_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    del metadata["ssim"]["K1"]
    del metadata["ssim"]["K2"]
    metadata["lpips"]["device"] = "another device"
    metadata["run_directory"] = "another machine"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    evaluate_nerf.evaluate_checkpoint(run, 4, "test", [0], device="cpu", render_chunk_size=13)
    assert csv_path.read_bytes() == original_csv
    migrated = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert migrated["ssim"]["K1"] == 0.01 and migrated["ssim"]["K2"] == 0.03
    assert migrated["lpips"]["device"] == "another device"

    metadata["lpips"]["calibrated_linear_weights_sha256"] = "different weight"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="scientific metric metadata differs"):
        evaluate_nerf.evaluate_checkpoint(run, 4, "val", [0], device="cpu", render_chunk_size=13)
    assert csv_path.read_bytes() == original_csv
    assert not (run / "evaluation" / "renders" / "iter_000004" / "val_view_000_prediction.png").exists()


@pytest.mark.parametrize("field,value", [
    (("ssim", "package_version"), "changed"),
    (("lpips", "package_version"), "changed"),
    (("lpips", "input"), "changed normalization"),
    (("renderer",), "changed ray convention"),
    (("metric_input",), "quantized PNGs"),
])
def test_scientific_metadata_signature_tracks_implementation_and_conventions(
    field: tuple[str, ...], value: str
) -> None:
    original = {
        "ssim": {"package_version": "0.26.0", "K1": 0.01, "K2": 0.03},
        "lpips": {"package_version": "0.1.4", "input": "normalize=True",
                  "device": "cpu", "calibrated_linear_weights_path": "machine A"},
        "renderer": "original pixel grid", "metric_input": "raw float RGB",
        "run_directory": "machine A",
    }
    modified = json.loads(json.dumps(original))
    target = modified
    for key in field[:-1]:
        target = target[key]
    target[field[-1]] = value
    assert evaluate_nerf._scientific_metadata(modified) != evaluate_nerf._scientific_metadata(original)
