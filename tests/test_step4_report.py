"""Scientific bookkeeping checks for Step 4 report generation."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from report_step4 import (EVALUATION_FIELDS, NOVEL_TEST_VIEWS,
                          budget_recommendation_text,
                          central_weight_bin_interval,
                          checkpoint_provenance_summary,
                          comparable_view_indices,
                          configure_training_loss_axis,
                          hardware_summary, overfitting_assessment, panel_limit_text,
                          read_evaluation_metrics, read_training_metrics,
                          run_status_explanation, save_novel_views,
                          selected_test_subset_mean, sustained_slowing,
                          training_budget_scope, training_loss_interval_trend,
                          verify_checkpoint_continuity,
                          verify_evaluation_summary)  # noqa: E402
from step4_artifacts import save_evaluation_view  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def _write_evaluation(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=EVALUATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _row(iteration: int, split: str, view_index: int) -> dict[str, object]:
    return dict(iteration=iteration, split=split, view_index=view_index,
                psnr=21.5, ssim=0.73, lpips=0.29, render_seconds=1.25,
                rays_per_second=512000, peak_vram_bytes=1200000,
                width=800, height=800, render_chunk_size=512)


def test_validation_curve_uses_only_view_indices_common_to_checkpoints(tmp_path: Path) -> None:
    path = tmp_path / "evaluation_results.csv"
    _write_evaluation(path, [_row(500, "val", 0), _row(500, "val", 1),
                             _row(2000, "val", 1), _row(2000, "val", 2),
                             _row(2000, "test", 1)])
    rows = read_evaluation_metrics(path)
    assert comparable_view_indices(rows, "val") == [1]
    assert comparable_view_indices(rows, "test") == [1]
    with path.open("a", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=EVALUATION_FIELDS).writerow(_row(2000, "val", 2))
    with pytest.raises(ValueError, match="duplicate evaluation identity"):
        read_evaluation_metrics(path)


def test_training_iteration_gap_and_split_mislabel_are_rejected(tmp_path: Path) -> None:
    train = tmp_path / "metrics.csv"
    train.write_text(
        "iteration,total_loss,coarse_loss,fine_loss,psnr,elapsed_time,rays_per_second\n"
        "1,0.4,0.2,0.2,7,1,1000\n3,0.3,0.1,0.2,7,2,1000\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contiguous"):
        read_training_metrics(train)
    eval_path = tmp_path / "evaluation_results.csv"
    _write_evaluation(eval_path, [_row(500, "validation", 0)])
    with pytest.raises(ValueError, match="invalid evaluation identity"):
        read_evaluation_metrics(eval_path)


def test_overfitting_assessment_abstains_without_matched_views() -> None:
    rows = [_row(step, split, view_index)
            for step, view_index in ((500, 0), (2000, 1), (5000, 2))
            for split in ("train", "val", "test")]
    message, possible, checkpoints = overfitting_assessment(rows)
    assert not possible
    assert checkpoints == [500, 2000, 5000]
    assert "cannot be determined" in message

    matched = [_row(step, split, 0) for step in (500, 2000, 5000)
               for split in ("train", "val", "test")]
    for row in matched:
        index = (500, 2000, 5000).index(row["iteration"])
        row["psnr"] = 20 + index if row["split"] == "train" else 20 - index
    message, possible, _ = overfitting_assessment(matched)
    assert possible
    assert "Matched per-split indices" in message


def test_summary_cross_check_rejects_disagreement(tmp_path: Path) -> None:
    summary = tmp_path / "evaluation_summary.csv"
    summary.write_text(
        "iteration,split,view_count,psnr_mean,psnr_median,psnr_std,ssim_mean,ssim_median,ssim_std,lpips_mean,lpips_median,lpips_std\n"
        "500,val,1,99,99,0,0.73,0.73,0,0.29,0.29,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="psnr_mean differs"):
        verify_evaluation_summary(summary, [_row(500, "val", 0)])


def test_single_dominant_coarse_weight_has_nonzero_focus_interval() -> None:
    interval = central_weight_bin_interval(
        np.array([2.5, 3.5, 4.5]), np.array([0.0, 1.0, 0.0]), 2.0, 5.0)
    assert interval == (3.0, 4.0)


def test_novel_figure_requires_spatially_separated_test_views(tmp_path: Path) -> None:
    image = torch.zeros((2, 3, 3), dtype=torch.float32)
    rows = []
    for view_index in (0, 1, 2):
        row = _row(10000, "test", view_index)
        row.update(width=3, height=2)
        rows.append(row)
        save_evaluation_view(tmp_path, 10000, "test", view_index, image, image)
    output = tmp_path / "novel.png"
    assert NOVEL_TEST_VIEWS == (0, 66, 133)
    assert save_novel_views(tmp_path, rows, 10000, list(NOVEL_TEST_VIEWS), output) == [66, 133]
    assert output.is_file()


def test_longitudinal_overfit_ignores_extra_diverse_views() -> None:
    rows = []
    for step_index, step in enumerate((500, 2000, 10000)):
        for split in ("train", "val", "test"):
            for view_index in (0, 1, 2):
                row = _row(step, split, view_index)
                row["psnr"] = 20 + step_index if split == "train" else 20 - step_index
                rows.append(row)
            if split == "test":
                for view_index in (66, 133):
                    row = _row(step, split, view_index)
                    row["psnr"] = 100 + step_index  # Would mask test decline if pooled.
                    rows.append(row)
    message, possible, _ = overfitting_assessment(rows)
    assert possible
    assert "'test': [0, 1, 2]" in message


def test_hardware_report_uses_recorded_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "environment_snapshot.json"
    snapshot.write_text(json.dumps({
        "gpu_name": "Recorded GPU Model", "gpu_total_vram_bytes": 2**30,
        "nvidia_driver": "example-driver", "pytorch": "example-torch",
        "pytorch_cuda_runtime": "example-cuda", "captured_utc": "example-time",
    }), encoding="utf-8")
    text = hardware_summary(snapshot)
    assert "Recorded GPU Model" in text
    assert "1.00 GiB" in text
    assert "example-driver" in text


def test_plateau_rejects_rebound_in_final_interval() -> None:
    assert sustained_slowing([(1.0, 1.0, 1.0), (0.4, 0.4, 0.4),
                              (0.2, 0.2, 0.2)])
    assert not sustained_slowing([(1.0, 1.0, 1.0), (0.4, 0.4, 0.4),
                                  (0.8, 0.8, 0.8)])
    assert not sustained_slowing([(1.0, 1.0, 1.0), (0.4, 0.4, 0.4),
                                  (-0.1, 0.2, 0.2)])


def test_training_loss_uses_interval_medians_not_single_endpoint() -> None:
    training = [dict(iteration=step, total_loss=(20.0 if step == 12 else
                                                 10.0 if step == 6 else
                                                 0.8 if step > 6 else 1.0))
                for step in range(1, 13)]
    description = training_loss_interval_trend(training, [2, 6, 12])
    assert "(2, 6]" in description and "(6, 12]" in description
    assert "1.000000 (n=4)" in description
    assert "0.800000 (n=6)" in description
    assert "not an automatic convergence gate" in description


def test_budget_scope_requires_matching_frozen_batch(tmp_path: Path) -> None:
    proposal = tmp_path / "baseline.yaml"
    proposal.write_text("batch_size: 4096\n", encoding="utf-8")
    description, matched = training_budget_scope({"batch_size": 256}, proposal)
    assert not matched
    assert "256" in description and "4096" in description
    assert "new matched 4096-ray baseline convergence run" in description
    proposal.write_text("batch_size: 256\n", encoding="utf-8")
    description, matched = training_budget_scope({"batch_size": 256}, proposal)
    assert matched and "both specify 256" in description
    proposal.write_text("batch_size: 256\nnear: 3.0\n", encoding="utf-8")
    description, matched = training_budget_scope({"batch_size": 256, "near": 2.0}, proposal)
    assert not matched and "'near'" in description


def test_budget_is_provisional_without_plateau_or_evaluated_checkpoint() -> None:
    steps = [500, 1000, 2000, 5000, 10000]
    description, supported = budget_recommendation_text(10000, "example", steps, False)
    assert not supported and "Provisional" in description
    description, supported = budget_recommendation_text(500, "example", steps, True)
    assert not supported and "Provisional" in description
    description, supported = budget_recommendation_text(7500, "example", steps, True)
    assert not supported and "Provisional" in description
    description, supported = budget_recommendation_text(5000, "measured plateau", steps, True)
    assert supported and "Proposed fixed Step 5 budget" in description


def _saved_checkpoint(iteration: int, config: dict, *, shift: float = 0.0) -> dict:
    return {
        "iteration": iteration, "config": config,
        "coarse_model": {"weight": torch.tensor([shift, 1.0])},
        "fine_model": {"weight": torch.tensor([shift + 1.0, 2.0])},
        "optimizer": {
            "state": {0: {"step": torch.tensor(float(iteration))},
                      1: {"step": torch.tensor(float(iteration))}},
            "param_groups": [{"params": [0, 1]}],
        },
        "scheduler": {"last_epoch": iteration},
    }


def test_checkpoint_continuity_checks_config_adam_scheduler_and_both_networks(
        tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    config = {"batch_size": 256}
    source = _saved_checkpoint(500, config)
    selected = _saved_checkpoint(2000, config, shift=2.0)
    torch.save(source, checkpoint_dir / "iter_000500.pt")
    selected_path = checkpoint_dir / "iter_002000.pt"
    torch.save(selected, selected_path)
    evidence = verify_checkpoint_continuity(tmp_path, config, 500, 2000)
    assert evidence["adam_parameter_states"] == {500: 2, 2000: 2}
    assert evidence["scheduler_epochs"] == {500: 500, 2000: 2000}
    assert evidence["network_changes"]["coarse_model"]["changed_tensors"] == 1
    assert evidence["network_changes"]["fine_model"]["l2_change"] > 0

    altered = _saved_checkpoint(2000, {"batch_size": 4096}, shift=2.0)
    torch.save(altered, selected_path)
    with pytest.raises(ValueError, match="saved config differs"):
        verify_checkpoint_continuity(tmp_path, config, 500, 2000)

    altered = _saved_checkpoint(2000, config, shift=2.0)
    altered["optimizer"]["state"][1]["step"] = torch.tensor(1999.0)
    torch.save(altered, selected_path)
    with pytest.raises(ValueError, match="Adam step counter"):
        verify_checkpoint_continuity(tmp_path, config, 500, 2000)

    altered = _saved_checkpoint(2000, config, shift=2.0)
    altered["scheduler"]["last_epoch"] = 1999
    torch.save(altered, selected_path)
    with pytest.raises(ValueError, match="scheduler epoch"):
        verify_checkpoint_continuity(tmp_path, config, 500, 2000)

    altered = _saved_checkpoint(2000, config, shift=2.0)
    altered["fine_model"] = source["fine_model"]
    torch.save(altered, selected_path)
    with pytest.raises(ValueError, match="fine_model: no saved parameter tensors changed"):
        verify_checkpoint_continuity(tmp_path, config, 500, 2000)


def test_checkpoint_hash_report_marks_retrospective_limit(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    provenance_dir = tmp_path / "evaluation"
    checkpoint_dir.mkdir()
    provenance_dir.mkdir()
    entries = {}
    for step, source in ((500, "backfilled_from_current_file_after_existing_metrics"),
                         (2000, "recorded_before_evaluation")):
        file = checkpoint_dir / f"iter_{step:06d}.pt"
        file.write_bytes(f"checkpoint {step}".encode())
        entries[str(step)] = {"relative_path": f"checkpoints/iter_{step:06d}.pt",
                              "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                              "source": source}
    (provenance_dir / "checkpoint_provenance.json").write_text(
        json.dumps({"algorithm": "sha256", "checkpoints": entries}), encoding="utf-8")
    description = checkpoint_provenance_summary(tmp_path, (500, 2000))
    assert "backfilled_from_current_file_after_existing_metrics" in description
    assert "recorded_before_evaluation" in description
    assert "cannot prove these exact weight bytes" in description
    (checkpoint_dir / "iter_002000.pt").write_bytes(b"replaced")
    with pytest.raises(ValueError, match="present file differs"):
        checkpoint_provenance_summary(tmp_path, (500, 2000))


def test_final_test_subset_means_remain_separate_from_union() -> None:
    rows = []
    for view, psnr in ((0, 10.0), (1, 100.0), (2, 100.0),
                       (66, 20.0), (133, 30.0)):
        row = _row(50000, "test", view)
        row["psnr"] = psnr
        rows.append(row)
    novel = selected_test_subset_mean(rows, NOVEL_TEST_VIEWS, "Novel")
    fixed = selected_test_subset_mean(rows, (0, 1, 2), "Fixed")
    assert "indices [0, 66, 133] mean: PSNR 20.000 dB" in novel
    assert "indices [0, 1, 2] mean: PSNR 70.000 dB" in fixed
    assert "unavailable (missing views [133])" in selected_test_subset_mean(
        rows[:-1], NOVEL_TEST_VIEWS, "Novel")


def test_complete_panel_wording_and_intentional_stage_stop() -> None:
    assert panel_limit_text([], []) == "All requested full-image panels are present."
    assert "Missing full-image panels" in panel_limit_text([(500, 0)], [])
    assert "intentional extension-stage stop" in run_status_explanation(
        {"status": "paused_after_checkpoint", "last_completed_iteration": 50000}, 50000)
    assert not run_status_explanation(
        {"status": "paused_after_checkpoint", "last_completed_iteration": 40000}, 50000)


def test_training_loss_axis_exposes_late_changes_on_log_scale() -> None:
    fig, axis = plt.subplots()
    try:
        configure_training_loss_axis(axis, [
            {"total_loss": 0.4, "coarse_loss": 0.2, "fine_loss": 0.2},
            {"total_loss": 0.006, "coarse_loss": 0.003, "fine_loss": 0.003},
        ])
        assert axis.get_yscale() == "log"
        assert axis.get_ylabel() == "Training-ray RGB MSE (log scale)"
        with pytest.raises(ValueError, match="must be positive"):
            configure_training_loss_axis(axis, [
                {"total_loss": 0.0, "coarse_loss": 0.0, "fine_loss": 0.0},
            ])
    finally:
        plt.close(fig)
