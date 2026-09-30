"""Checkpoint provenance and cross-process evaluation commit checks."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nerf_step1.evaluation_io import (  # noqa: E402
    evaluation_write_lock, file_sha256, verify_checkpoint_provenance,
)


def test_checkpoint_provenance_backfills_existing_rows_and_detects_drift(tmp_path: Path) -> None:
    run = tmp_path / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    old_checkpoint = checkpoints / "iter_000500.pt"
    new_checkpoint = checkpoints / "iter_001000.pt"
    old_checkpoint.write_bytes(b"old checkpoint")
    new_checkpoint.write_bytes(b"new checkpoint")
    csv_path = run / "evaluation_results.csv"
    csv_path.write_text("iteration,split,view_index\n500,val,0\n", encoding="utf-8")
    digest = file_sha256(new_checkpoint)
    with evaluation_write_lock(run):
        provenance_path = verify_checkpoint_provenance(run, 1000, digest)
    manifest = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert manifest["checkpoints"]["500"]["source"] == "backfilled_from_current_file_after_existing_metrics"
    assert manifest["checkpoints"]["500"]["sha256"] == file_sha256(old_checkpoint)
    assert manifest["checkpoints"]["1000"]["source"] == "recorded_before_evaluation"
    assert manifest["checkpoints"]["1000"]["sha256"] == digest

    old_checkpoint.write_bytes(b"historical checkpoint replaced")
    with evaluation_write_lock(run), pytest.raises(ValueError, match="provenance mismatch at iteration 500"):
        verify_checkpoint_provenance(run, 1000, digest)
    assert csv_path.read_text(encoding="utf-8") == "iteration,split,view_index\n500,val,0\n"


_CONCURRENT_WRITER = r"""
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from nerf_step1.evaluation import EvaluationResult, ViewMetrics, aggregate_evaluation, export_evaluation_csv, export_summary_csv
from nerf_step1.evaluation_io import evaluation_write_lock
run = Path(sys.argv[2]); role = sys.argv[3]
if role == 'a':
    with evaluation_write_lock(run):
        (run / 'a_entered').touch()
        deadline = time.monotonic() + 15
        while not (run / 'release_a').exists():
            if time.monotonic() >= deadline: raise TimeoutError('release signal absent')
            time.sleep(0.02)
        row = ViewMetrics(1, 'val', 0, 20.0, 0.8, 0.2, 1.0, 64.0, None, 8, 8, 8)
        result = EvaluationResult(1, 'val', (row,), aggregate_evaluation((row,)))
        export_evaluation_csv(run / 'evaluation_results.csv', result)
        export_summary_csv(run / 'evaluation_summary.csv', result,
                           per_view_csv_path=run / 'evaluation_results.csv')
else:
    (run / 'b_attempting').touch()
    with evaluation_write_lock(run):
        (run / 'b_entered').touch()
        row = ViewMetrics(1, 'test', 0, 19.0, 0.7, 0.3, 1.0, 64.0, None, 8, 8, 8)
        result = EvaluationResult(1, 'test', (row,), aggregate_evaluation((row,)))
        export_evaluation_csv(run / 'evaluation_results.csv', result)
        export_summary_csv(run / 'evaluation_summary.csv', result,
                           per_view_csv_path=run / 'evaluation_results.csv')
"""


def _wait_for(path: Path, *, timeout_seconds: float = 20) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"worker did not create {path}")
        time.sleep(0.02)


def test_two_processes_serialize_csv_read_merge_replace(tmp_path: Path) -> None:
    run = tmp_path / "concurrent_run"
    run.mkdir()
    base = [sys.executable, "-c", _CONCURRENT_WRITER, str(PROJECT_ROOT / "src"), str(run)]
    first = subprocess.Popen([*base, "a"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    second = None
    try:
        _wait_for(run / "a_entered")
        second = subprocess.Popen([*base, "b"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        _wait_for(run / "b_attempting")
        # B has begun its lock attempt while A still owns the critical section.
        assert not (run / "b_entered").exists()
        (run / "release_a").touch()
        out_a, err_a = first.communicate(timeout=20)
        out_b, err_b = second.communicate(timeout=20)
        assert first.returncode == 0, (out_a, err_a)
        assert second.returncode == 0, (out_b, err_b)
        with (run / "evaluation_results.csv").open(newline="", encoding="utf-8") as stream:
            assert {(row["split"], row["view_index"]) for row in csv.DictReader(stream)} == {
                ("val", "0"), ("test", "0")
            }
        with (run / "evaluation_summary.csv").open(newline="", encoding="utf-8") as stream:
            assert {row["split"] for row in csv.DictReader(stream)} == {"val", "test"}
    finally:
        (run / "release_a").touch()
        for process in (first, second):
            if process is not None and process.poll() is None:
                process.terminate()
                process.communicate(timeout=5)
