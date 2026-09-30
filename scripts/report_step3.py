"""Build Step 3 validation figures and reports from one completed smoke run."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from nerf_step1.hierarchical import sample_pdf  # noqa: E402


def read_metrics(run_directory: Path) -> list[dict[str, float]]:
    with (run_directory / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or [int(row["iteration"]) for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("smoke metrics must contain contiguous iterations starting at 1")
    return [{key: float(value) for key, value in row.items() if value != ""} for row in rows]


def rolling_mean(values: list[float], window: int = 25) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    cumulative = np.cumsum(np.r_[0.0, vector])
    result = np.empty_like(vector)
    for index in range(len(vector)):
        start = max(0, index + 1 - window)
        result[index] = (cumulative[index + 1] - cumulative[start]) / (index + 1 - start)
    return result


def save_curve(rows: list[dict[str, float]], path: Path) -> None:
    x = [int(row["iteration"]) for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    for key, color, label in (
        ("total_loss", "#a53924", "Total MSE"),
        ("coarse_loss", "#386f8c", "Coarse MSE"),
        ("fine_loss", "#8a6795", "Fine MSE"),
    ):
        values = [row[key] for row in rows]
        axes[0].plot(x, values, color=color, alpha=0.17, lw=0.8)
        axes[0].plot(x, rolling_mean(values), color=color, lw=1.9, label=f"{label}, 25-step mean")
    axes[0].set_ylabel("RGB squared error")
    axes[0].legend(loc="upper right", fontsize=8)
    psnr = [row["psnr"] for row in rows]
    axes[1].plot(x, psnr, color="#8d7132", alpha=0.22, lw=0.8)
    axes[1].plot(x, rolling_mean(psnr), color="#8d7132", lw=2.0,
                 label="Fine PSNR, 25-step mean")
    axes[1].set_ylabel("Fine PSNR (dB)")
    axes[1].set_xlabel("Completed training iteration")
    axes[1].legend(loc="lower right", fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.axvline(50, color="#555555", lw=1, ls="--", alpha=0.6)
    fig.suptitle("Step 3 Lego engineering smoke run | L=10, seed=0 | resumed after iteration 50")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_preview_sheet(run_directory: Path, path: Path) -> None:
    stages = (0, 250, 500)
    columns = ("prediction", "ground_truth", "absolute_difference")
    scale = 2
    sample = Image.open(run_directory / "renders" / "prediction_iter_000000.png")
    width, height = sample.size
    sample.close()
    panel_width, panel_height = width * scale, height * scale
    header_height, row_header = 26, 24
    sheet = Image.new("RGB", (panel_width * 3, header_height + len(stages) * (panel_height + row_header)),
                      (247, 247, 245))
    draw = ImageDraw.Draw(sheet)
    for column, title in enumerate(("Prediction", "Ground truth", "Absolute difference")):
        draw.text((column * panel_width + 6, 7), title, fill=(20, 20, 20))
    for row, step in enumerate(stages):
        top = header_height + row * (panel_height + row_header)
        draw.text((6, top + 5), f"Iteration {step}", fill=(20, 20, 20))
        for column, key in enumerate(columns):
            with Image.open(run_directory / "renders" / f"{key}_iter_{step:06d}.png") as source:
                panel = source.convert("RGB").resize((panel_width, panel_height), Image.Resampling.NEAREST)
            sheet.paste(panel, (column * panel_width, top + row_header))
    sheet.save(path)


def save_sampling_figure(path: Path) -> float:
    near, far, n_coarse, n_fine = 2.0, 6.0, 64, 128
    edges = torch.linspace(near, far, n_coarse + 1, dtype=torch.float64)
    coarse = 0.5 * (edges[:-1] + edges[1:])
    midpoint_edges = 0.5 * (coarse[1:] + coarse[:-1])
    weights = torch.exp(-0.5 * ((coarse[1:-1] - 4.1) / 0.18) ** 2) + 0.0001
    fine = sample_pdf(midpoint_edges[None], weights[None], n_fine)[0]
    focus = (3.7, 4.5)
    focused_fraction = float(((fine >= focus[0]) & (fine <= focus[1])).double().mean())
    fig, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
    axes[0].eventplot(coarse.numpy(), colors="#43738a", lineoffsets=0, linelengths=0.8)
    axes[0].set_yticks([])
    axes[0].set_ylabel("Coarse")
    axes[1].plot(coarse[1:-1], weights, color="#a44630", marker=".", markersize=4)
    axes[1].set_ylabel("Synthetic weight")
    axes[2].eventplot(fine.numpy(), colors="#825997", lineoffsets=0, linelengths=0.8)
    axes[2].set_yticks([])
    axes[2].set_ylabel("Fine")
    axes[2].set_xlabel("Ray depth parameter t")
    for axis in axes:
        axis.axvspan(*focus, color="#c1a447", alpha=0.13)
        axis.set_xlim(near, far)
        axis.grid(axis="x", alpha=0.17)
    fig.suptitle(f"Synthetic PDF sanity check: {focused_fraction:.1%} of fine samples inside a 20% depth interval")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return focused_fraction


def pytest_totals(path: Path) -> tuple[int, int, int, int]:
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        raise ValueError("pytest XML contains no testsuite")
    return tuple(int(suite.get(key, "0")) for key in ("tests", "failures", "errors", "skipped"))


def _parameter_changes(run_directory: Path) -> dict[str, float]:
    files = [run_directory / "checkpoints" / f"iter_{step:06d}.pt" for step in (50, 500)]
    early, late = [torch.load(path, map_location="cpu", weights_only=True) for path in files]
    changes = {}
    for key in ("coarse_model", "fine_model"):
        changes[key] = float(sum(
            (late[key][name] - early[key][name]).square().sum()
            for name in late[key]
        ).sqrt())
    return changes


def gpu_probe_summary(path: Path) -> tuple[int, int, int, float, int]:
    """Summarize nvidia-smi polls while the separate GPU probe held >900 MiB."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        readings = [row for row in csv.reader(stream) if row]
    active = [(int(row[1].strip()), int(row[2].strip())) for row in readings
              if int(row[2].strip()) >= 900]
    if not active:
        raise ValueError("GPU probe has no active CUDA samples")
    utilizations = [value for value, _ in active]
    return (len(active), min(utilizations), max(utilizations),
            statistics.median(utilizations), max(memory for _, memory in active))


def write_reports(run_directory: Path, artifact_directory: Path,
                  rows: list[dict[str, float]], focused_fraction: float,
                  tests: tuple[int, int, int, int],
                  gpu_probe: tuple[int, int, int, float, int],
                  gpu_probe_run: Path) -> None:
    test_count, failures, errors, skipped = tests
    first, last = rows[0], rows[-1]
    early_loss = statistics.mean(row["total_loss"] for row in rows[:25])
    late_loss = statistics.mean(row["total_loss"] for row in rows[-25:])
    early_psnr = statistics.mean(row["psnr"] for row in rows[:25])
    late_psnr = statistics.mean(row["psnr"] for row in rows[-25:])
    allocated = [row["gpu_memory_allocated_bytes"] for row in rows]
    reserved = [row["gpu_memory_reserved_bytes"] for row in rows]
    changes = _parameter_changes(run_directory)
    probe_samples, probe_min, probe_max, probe_median, probe_memory = gpu_probe
    rel_run = run_directory.relative_to(PROJECT_ROOT).as_posix()
    passed = test_count - failures - errors - skipped

    volume = f"""# Step 3 volume-rendering validation

## Mathematical contract

For ordered ray depths `t_i` and physical gaps `delta_i = (t_(i+1)-t_i) ||d||`,
`alpha_i = 1-exp(-sigma_i delta_i)`, `T_i = product_(j<i)(1-alpha_j)`,
and `w_i = T_i alpha_i`. `rgb_map = sum_i(w_i rgb_i) + (1-sum_i w_i)[1,1,1]`;
`depth_map = sum_i(w_i t_i)` and `accumulated_opacity = sum_i w_i`.
The last gap is `1e10 ||d||`, matching the [official NeRF renderer](https://github.com/bmild/nerf/blob/master/run_nerf.py).
The large terminal gap models the remaining ray segment; it is not a finite
integration only to `far`. Step 1's ray direction remains unnormalized, and
the norm converts parameter gaps to world distance. Only the MLP's viewing
direction is normalized. White residual matches the Step 1 RGBA target composite.

`src/nerf_step1/volume_rendering.py` uses `-expm1(-sigma*delta)` for alpha
and an exclusive cumulative optical depth for transmittance. It clips optical
depth only once transmission is below the floating dtype's normal range.
It checks finite inputs and outputs and rejects negative densities/distances.
The returned `RenderOutput` contains RGB, weights, alpha, transmittance,
depth, opacity; disparity can be derived later without changing the contract.

## Direct checks

- Artificial two-sample ray: each density `ln(2)` over unit distance gives
  alpha `[0.5,0.5]`, transmittance `[1,0.5]`, weights `[0.5,0.25]`,
  opacity `0.75`, and white-backed red/blue RGB `[0.75,0.25,0.5]`.
- Tests cover zero density, an opaque first or red sample, two-color hand
  compositing, nonnegative and bounded weight sums, output shape, finite extreme
  densities, RGB/density gradients, and CPU/CUDA agreement.
- Full suite evidence: {passed}/{test_count} passed, {skipped} skipped,
  {failures} failures, {errors} errors (`{rel_run}/pytest_results.xml`).

The positional encoding formula and Step 1 camera-ray generator were not changed.
"""
    (artifact_directory / "volume_rendering_validation.md").write_text(volume, encoding="utf-8")

    hierarchical = f"""# Step 3 hierarchical-sampling validation

The coarse pass draws 64 stratified depths from uniform bins (fixed midpoints
for deterministic evaluation). The pipeline renders coarse weights, forms
midpoint edges from neighboring coarse depths, and uses interior weights
`weights[...,1:-1]`, following [the released NeRF implementation](https://github.com/bmild/nerf/blob/master/run_nerf.py).
`sample_pdf()` adds `1e-5` per interval so empty rays fall back to a defined
PDF, builds a CDF, and uses inverse transform sampling to draw 128 fine depths.
The draw is detached from the coarse graph. All coarse and fine depths are
retained, sorted, and evaluated by a separate fine `VanillaNeRF` instance.

The deterministic artificial-ray figure is
`artifacts/hierarchical_sampling_visualization.png`. Its weight peak is near
`t=4.1`; **{focused_fraction:.1%}** of fine samples fall in `[3.7,4.5]`,
which covers 20% of the `[2,6]` interval. This is a sampler diagnostic only,
not an observation about Lego scene complexity.

Tests cover uniform and focused weights, zero and tiny weights, legal bounds,
shape, seed reproducibility, CPU/CUDA, finite outputs, and merged-depth order.
The merged array is nondecreasing. Exact duplicate coarse/fine depths retain
zero-width intervals, so strict inequality cannot be guaranteed while also
retaining exactly `Nc+Nf` samples. The renderer accepts those zero gaps.
Full suite: {passed}/{test_count} passed, {skipped} skipped, {failures} failures,
{errors} errors.
"""
    (artifact_directory / "hierarchical_sampling_validation.md").write_text(hierarchical, encoding="utf-8")

    report = f"""# NeRF EE Research Prototype — Step 3 completion report

This is an **engineering smoke test**, not a formal EE experiment, L sweep,
multi-scene comparison, SSIM/LPIPS evaluation, or research conclusion.

## Files and end-to-end path

New implementation: `src/nerf_step1/sampling.py`, `volume_rendering.py`,
`hierarchical.py`, `pipeline.py`, `preview.py`, `training_io.py`,
`scripts/train_nerf.py`, and `scripts/report_step3.py`. New tests:
`tests/test_sampling.py`, `test_volume_rendering.py`, `test_hierarchical.py`,
`test_pipeline.py`, `test_preview.py`, and `test_training_io.py`.
Updated: `configs/baseline.yaml` only to add explicit ray bounds, decay,
initialization and operational settings; its earlier scientific values remain
unchanged. `configs/smoke.yaml` independently reduces batch and iterations.
`README.md` and `artifacts/research_log.md` describe this stage.

The actual path is official Lego train PNG/pixels -> Step 1 `generate_rays`
-> 64 stratified positions -> Step 2 position/direction encoding and coarse
`VanillaNeRF` -> white-background volume rendering -> coarse PDF -> 128 extra
depths, merged with coarse -> a separate Step 2 `VanillaNeRF` -> final RGB
-> RGB MSE -> backward -> Adam update. The default model has eight 256-wide
position layers and a 128-wide view branch in **each** network.

## Loss, optimizer and controlled settings

`coarse_loss = mean((coarse_rgb - target)^2)`;
`fine_loss = mean((fine_rgb - target)^2)`; `total_loss` is their sum.
`PSNR = -10 log10(fine_loss)` in dB. Adam starts at `5e-4` and a configured
exponential scheduler reaches 0.1x after 250,000 iterations. The smoke run
used Lego, white background, unnormalized Step 1 rays, near=2, far=6,
position L=10 with the **paper's `2^k pi p` convention**, direction L=4,
seed=0, 64 coarse + 128 fine samples, default 8x256/128 architecture,
batch 256, 500 iterations, 128-ray backward chunks, and 100x100 validation
previews. Its `density_initial_bias: 0.1` is explicit: initial random ReLU
density sometimes produced all-transparent coarse rays and no density gradient.
This bias starts both networks with trainable opacity but changes no layer or
encoding definition. Formal baseline retains batch 4096 and 500000 iterations.

## Smoke evidence

Run: `{rel_run}/` (`config_snapshot.yaml`, `environment_snapshot.json`,
`run_manifest.json`, `metrics.csv`, checkpoints, and iteration-stamped renders).
The first 50 updates ran, saved `iter_000050.pt`, then a new process loaded
model, optimizer, scheduler and RNG state and continued from **iteration 51**
through 500. Metric iterations are contiguous 1–500. A separate uninterrupted
same-seed 100-step probe (`{gpu_probe_run.relative_to(PROJECT_ROOT).as_posix()}/`)
produced **exactly identical first 100 total/coarse/fine losses, PSNR and LR**
to this run, including its resumed iterations 51–100. Direct CPU save/load
checks and an end-to-end stochastic next-step equality test also passed.

| Measure | Iteration 1 | Iteration 500 |
|---|---:|---:|
| Total loss | {first['total_loss']:.6f} | {last['total_loss']:.6f} |
| Coarse loss | {first['coarse_loss']:.6f} | {last['coarse_loss']:.6f} |
| Fine loss | {first['fine_loss']:.6f} | {last['fine_loss']:.6f} |
| Fine PSNR | {first['psnr']:.2f} dB | {last['psnr']:.2f} dB |

First versus last 25-iteration mean: total loss {early_loss:.6f} ->
{late_loss:.6f}; fine PSNR {early_psnr:.2f} -> {late_psnr:.2f} dB.
Both checkpointed networks changed between iterations 50 and 500:
parameter L2 change coarse={changes['coarse_model']:.4f},
fine={changes['fine_model']:.4f}. This supplements the explicit two-network
gradient/update tests. The 100x100 validation preview starts near white,
develops a blurred but recognizable Lego silhouette, and remains far from
converged; see `artifacts/smoke_preview.png` and per-iteration prediction,
ground truth, and absolute difference under the run's `renders/` directory.

CUDA ran on the NVIDIA GeForce RTX 5090 Laptop GPU (23.86 GiB total).
Post-step allocated VRAM ranged {min(allocated)/2**30:.3f}–{max(allocated)/2**30:.3f} GiB;
reserved VRAM ranged {min(reserved)/2**30:.3f}–{max(reserved)/2**30:.3f} GiB
across all 500 metrics. These are post-step readings; peak per-step allocation
was not sampled in the main run. A separate **100-iteration engineering GPU
probe with the same smoke config** polled `nvidia-smi` approximately every
0.3 seconds. While process VRAM was at least 900 MiB, {probe_samples} readings
showed utilization {probe_min}–{probe_max}% (median {probe_median:.0f}%) and
up to {probe_memory} MiB total GPU memory used. The probe's raw samples are in
`artifacts/gpu_utilization_probe.csv`; they are not metrics of the 500-step run.
CUDA forward/backward tests and actual optimization completed.

Full pytest suite: **{passed}/{test_count} passed**, {skipped} skipped,
{failures} failures, {errors} errors; includes Step 1, Step 2, and Step 3.
The tiny synthetic overfit test substantially reduces total loss.

## Artifacts and limits

- `artifacts/volume_rendering_validation.md`
- `artifacts/hierarchical_sampling_validation.md`
- `artifacts/training_pipeline_report.md`
- `artifacts/smoke_training_curve.png`
- `artifacts/smoke_preview.png`
- `artifacts/hierarchical_sampling_visualization.png`
- `artifacts/gpu_utilization_probe.csv` (separate operational probe)

Known limits: this 500-step run only demonstrates the mathematical and
engineering loop; the validation silhouette is blurred, no full-resolution
evaluation has been run, and GPU utilization percentage was sampled only in
the separate short probe rather than inside the main run.
The terminal 1e10 interval is the original NeRF renderer convention, beyond
the finite sampling far bound. Exact duplicate merged depths can be equal.
The run has no Git commit because this directory is not a Git repository;
the manifest records null. No SSIM/LPIPS or formal scene/L comparisons exist.

**Step 4 readiness:** the Step 3 code, tests, checkpoint resume, actual loss
decrease and emerging structure satisfy the core pipeline gate. Step 4 may
start as a separate authorized stage, with convergence/evaluation handled
there. This report does not start Step 4.
"""
    (artifact_directory / "training_pipeline_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pytest-xml", type=Path, required=True)
    parser.add_argument("--gpu-probe-run", type=Path, required=True)
    args = parser.parse_args()
    run_directory = args.run.resolve()
    status = json.loads((run_directory / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "completed" or status.get("last_completed_iteration") != 500:
        raise ValueError("report requires a completed 500-iteration smoke run")
    rows = read_metrics(run_directory)
    if len(rows) != 500:
        raise ValueError("smoke metrics must contain all 500 iterations")
    probe_run = args.gpu_probe_run.resolve()
    probe_rows = read_metrics(probe_run)
    if len(probe_rows) != 100:
        raise ValueError("GPU probe must have 100 metric rows")
    if (run_directory / "config_snapshot.yaml").read_bytes() != (probe_run / "config_snapshot.yaml").read_bytes():
        raise ValueError("GPU probe config differs from the primary smoke run")
    compare_keys = ("total_loss", "coarse_loss", "fine_loss", "psnr", "learning_rate")
    if any(any(rows[index][key] != probe_rows[index][key] for key in compare_keys)
           for index in range(100)):
        raise ValueError("same-seed probe did not reproduce the first 100 metrics")
    artifacts = PROJECT_ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    save_curve(rows, artifacts / "smoke_training_curve.png")
    save_preview_sheet(run_directory, artifacts / "smoke_preview.png")
    fraction = save_sampling_figure(artifacts / "hierarchical_sampling_visualization.png")
    write_reports(run_directory, artifacts, rows, fraction, pytest_totals(args.pytest_xml),
                  gpu_probe_summary(artifacts / "gpu_utilization_probe.csv"), probe_run)
    print(json.dumps({"run": str(run_directory), "focused_fraction": fraction,
                      "artifacts": str(artifacts)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
