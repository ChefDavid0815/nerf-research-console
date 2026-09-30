"""Validate the Step 2 encoding/MLP and create reproducible diagnostics.

This script runs software tests and tiny forward/backward probes. It never
loads a scene, renders pixels, optimizes weights, or measures reconstruction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nerf_step1.model import VanillaNeRF  # noqa: E402
from nerf_step1.positional_encoding import SinusoidalPositionalEncoding  # noqa: E402
from nerf_step1.run_logging import start_run, write_status  # noqa: E402


COMPARISON_BANDWIDTHS = (0, 2, 4, 6, 8, 10, 12, 15)
PLOT_BANDWIDTHS = (1, 2, 4, 6, 10)


def read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    required = (
        "position_encoding_L", "direction_encoding_L", "network_depth",
        "network_width", "skip_connection_layer", "view_width", "random_seed",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"{path}: missing required keys: {', '.join(missing)}")
    if type(config["random_seed"]) is not int or config["random_seed"] < 0:
        raise ValueError("random_seed must be a nonnegative integer")
    return config


def run_tests(run_directory: Path) -> dict:
    command = [
        sys.executable, "-m", "pytest", "-q",
        f"--junitxml={run_directory / 'pytest.xml'}",
    ]
    result = subprocess.run(
        command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    output = result.stdout + ("\n" + result.stderr if result.stderr else "")
    (run_directory / "pytest.log").write_text(output, encoding="utf-8")
    xml_path = run_directory / "pytest.xml"
    if not xml_path.is_file():
        raise RuntimeError(f"pytest did not produce {xml_path}:\n{output[-2000:]}")
    root = ET.parse(xml_path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    # The outer testsuites element has no counts; each testsuite is counted once.
    counts = {
        key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    counts["passed"] = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    counts["exit_code"] = result.returncode
    counts["summary"] = output.strip().splitlines()[-1] if output.strip() else "no output"
    if result.returncode != 0 or counts["failures"] or counts["errors"]:
        raise RuntimeError(f"pytest failed; inspect {run_directory / 'pytest.log'}")
    return counts


def run_forward_backward(config: dict, device: str) -> dict:
    # A dedicated probe checks the exact configured model on the selected device.
    torch.manual_seed(config["random_seed"])
    model = VanillaNeRF.from_config(config).to(device)
    positions = torch.tensor(
        [[0.1, -0.2, 0.3], [-0.4, 0.5, 0.2], [0.0, 0.0, 0.0]],
        dtype=torch.float32, device=device, requires_grad=True,
    )
    directions = torch.tensor(
        [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32, device=device, requires_grad=True,
    )
    rgb, sigma = model(positions, directions)
    if rgb.shape != (3, 3) or sigma.shape != (3, 1):
        raise AssertionError(f"{device}: incorrect output shape: {rgb.shape}, {sigma.shape}")
    if not bool(torch.isfinite(rgb).all() and torch.isfinite(sigma).all()):
        raise AssertionError(f"{device}: nonfinite output")
    if not bool(((rgb >= 0) & (rgb <= 1)).all() and (sigma >= 0).all()):
        raise AssertionError(f"{device}: output outside RGB/density range")
    # The RGB path reaches both branches without depending on untrained sigma
    # being positive at every point.
    (rgb.square().mean() + sigma.mean()).backward()
    missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    if missing:
        raise AssertionError(f"{device}: missing parameter gradients: {missing}")
    if not all(bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise AssertionError(f"{device}: nonfinite parameter gradient")
    if positions.grad is None or directions.grad is None:
        raise AssertionError(f"{device}: missing input gradient")
    return {
        "device": device,
        "rgb_shape": list(rgb.shape),
        "sigma_shape": list(sigma.shape),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "all_parameter_grads_present_and_finite": True,
        "input_grads_present": True,
    }


def save_frequency_plot(path: Path) -> None:
    positions = torch.linspace(-1.0, 1.0, 16385, dtype=torch.float64).unsqueeze(-1)
    x = positions[:, 0].numpy()
    colors = ("#176b9a", "#db8332", "#bb3b72")
    fig, axes = plt.subplots(len(PLOT_BANDWIDTHS), 2, figsize=(14, 15), sharey=True)
    fig.suptitle(
        "Sinusoidal positional encoding: more bands expose finer spatial variation",
        fontsize=17, fontweight="bold", y=0.995,
    )
    for row, bandwidth in enumerate(PLOT_BANDWIDTHS):
        encoder = SinusoidalPositionalEncoding(1, bandwidth, include_input=True)
        encoded = encoder(positions).detach().numpy()
        selected = tuple(dict.fromkeys((0, (bandwidth - 1) // 2, bandwidth - 1)))
        for column, ax in enumerate(axes[row]):
            for color, band in zip(colors, selected):
                # Scalar layout: raw p, sin(f0*pi*p), cos(f0*pi*p), ...
                ax.plot(
                    x, encoded[:, 1 + 2 * band], color=color,
                    lw=1.4 if band == 0 else 1.0,
                    alpha=(0.95 if column == 1 or band == 0 else
                           0.22 if bandwidth >= 10 and band == bandwidth - 1 else 0.65),
                    label=f"sin(2^{band} pi p)",
                )
            ax.set_ylim(-1.1, 1.1)
            ax.axhline(0, color="#777777", linewidth=0.5)
            ax.grid(alpha=0.18)
            ax.set_xlabel("position p")
            if column == 0:
                ax.set_xlim(-1, 1)
                ax.set_ylabel(f"L = {bandwidth}\nfeature value")
                ax.set_title("Full domain: p in [-1, 1]", fontsize=10)
            else:
                half_width = min(1.0, 3.0 / (2 ** (bandwidth - 1)))
                ax.set_xlim(-half_width, half_width)
                ax.set_title(
                    f"Zoom around 0: highest multiplier is 2^{bandwidth - 1}",
                    fontsize=10,
                )
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.text(
        0.5, 0.006,
        "Each row uses the actual encoder output. As L rises, the highest band oscillates faster; "
        "the right panel reveals those dense waves.",
        ha="center", fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.98), h_pad=1.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def model_counts(config: dict) -> list[dict]:
    rows = []
    for bandwidth in COMPARISON_BANDWIDTHS:
        variant = dict(config)
        variant["position_encoding_L"] = bandwidth
        model = VanillaNeRF.from_config(variant)
        rows.append({
            "L": bandwidth,
            "position_dim": model.position_encoder.output_dim,
            "direction_dim": model.direction_encoder.output_dim,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        })
    return rows


def write_encoding_report(path: Path, counts: dict) -> None:
    bandwidths = (0, 1, 2, 4, 6, 10, 15)
    lines = [
        "# Positional encoding diagnostic", "",
        "## Formula and convention", "",
        "For each scalar p: gamma_L(p) = [p, sin(2^0 pi p), cos(2^0 pi p), "
        "..., sin(2^(L-1) pi p), cos(2^(L-1) pi p)].",
        "The same bands are applied independently to x, y and z. Each band is "
        "ordered as sin(all input components), then cos(all input components).",
        "`include_input=True` is mandatory in this study. Therefore a 3D input "
        "has `3 + 6L` output dimensions. `L=0` returns the original xyz exactly.",
        "All bands are generated from L at construction and remain fixed for a run.", "",
        "The paper uses pi in its written Fourier mapping; the released TensorFlow "
        "embedder multiplies by powers of two without pi. This project follows the "
        "explicit study specification above. The released code includes the raw "
        "input, whereas the paper diagram labels only 60/24 Fourier channels.", "",
        "- Paper: https://arxiv.org/pdf/2003.08934",
        "- Official implementation: https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py", "",
        "## Dimensions and actual bands", "",
        "| L | 3D output dimensions | Frequency multipliers before pi |", "|---:|---:|---|",
    ]
    for bandwidth in bandwidths:
        bands = ", ".join(str(2 ** k) for k in range(bandwidth)) or "none"
        lines.append(f"| {bandwidth} | {3 + 6 * bandwidth} | {bands} |")
    lines += [
        "", "## Verification", "",
        f"- Full pytest suite: {counts['passed']} passed of {counts['tests']} tests; "
        f"{counts['skipped']} skipped, {counts['failures']} failures, {counts['errors']} errors.",
        "- The suite checks manual p=0.5 values, shape, frequencies, L=0 identity, "
        "determinism, dtype, CPU/CUDA agreement, finite outputs and gradients.",
        "- Plot: `artifacts/positional_encoding_frequencies.png` uses encoder "
        "output on p in [-1, 1], with close views for dense high-frequency bands.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_architecture(path: Path, config: dict, model: VanillaNeRF) -> None:
    position_dim = model.position_encoder.output_dim
    direction_dim = model.direction_encoder.output_dim
    total = sum(parameter.numel() for parameter in model.parameters())
    lines = [
        "Vanilla NeRF MLP architecture (Step 2; untrained)",
        "F_theta(positions [N,3], directions [N,3]) -> RGB [N,3], density [N,1]",
        f"position_encoding_L={config['position_encoding_L']}; gamma(x)=[N,{position_dim}]",
        f"direction_encoding_L={config['direction_encoding_L']}; gamma(d)=[N,{direction_dim}]",
        f"network_depth={config['network_depth']}; network_width={config['network_width']}",
        f"skip_connection_layer={config['skip_connection_layer']} (zero-based)",
        f"view_width={config['view_width']}", "",
        "Position branch (each Linear followed by ReLU):",
    ]
    for index, layer in enumerate(model.position_layers):
        lines.append(
            f"  layer {index + 1}: [N,{layer.in_features}] -> [N,{layer.out_features}]"
        )
        if index == config["skip_connection_layer"]:
            lines.append(
                f"  skip AFTER layer {index + 1} ReLU: concatenate gamma(x) "
                f"=> [N,{layer.out_features + position_dim}]"
            )
    for label, layer in (
        ("Density branch", model.density_layer),
        ("Feature branch", model.feature_layer),
        ("Colour/view hidden branch", model.view_layer),
        ("RGB output branch", model.rgb_layer),
    ):
        lines.append(f"{label}: [N,{layer.in_features}] -> [N,{layer.out_features}]")
    lines += [
        "Density output uses a nonnegative activation; RGB output uses sigmoid [0,1].",
        "The official release applies those output activations in its renderer; "
        "this Step 2 model returns the bounded values at its forward interface.",
        "For original NeRF view conditioning, pass unit viewing directions; "
        "Step 1 ray generation itself remains unchanged and unnormalized.",
        f"Total trainable parameters: {total:,}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_parameter_report(path: Path, config: dict, rows: list[dict]) -> None:
    baseline = next(row for row in rows if row["L"] == config["position_encoding_L"])
    lines = [
        "# Model parameter count by position encoding bandwidth", "",
        "Only `position_encoding_L` changes in this comparison. "
        "The model is instantiated but never trained.",
        f"Controlled settings: direction L={config['direction_encoding_L']}, "
        f"depth={config['network_depth']}, width={config['network_width']}, "
        f"skip index={config['skip_connection_layer']}, view width={config['view_width']}.",
        "The independent variable is fixed when each model is constructed. "
        "A future scene x L x seed run should save the entire config snapshot "
        "before fitting; the existing run ledger writes `runs/<run>/config.yaml`.", "",
        "| Position L | Position input dim | Direction input dim | Trainable parameters | "
        "Delta from baseline L |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['L']} | {row['position_dim']} | {row['direction_dim']} | "
            f"{row['parameters']:,} | {row['parameters'] - baseline['parameters']:+,} |"
        )
    lines += [
        "", "Changing L changes the first position layer and the layer after the "
        "skip concatenation. Parameter count therefore changes, even with "
        "depth and width controlled. This should be disclosed when comparing "
        "future training cost or reconstruction accuracy.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = read_config(config_path)
    run_directory, logger = start_run(PROJECT_ROOT / "runs", "step2_validation", config)
    try:
        tests = run_tests(run_directory)
        probes = {"cpu": run_forward_backward(config, "cpu")}
        if torch.cuda.is_available():
            probes["cuda"] = run_forward_backward(config, "cuda")
        else:
            probes["cuda"] = {"status": "unavailable"}
        model = VanillaNeRF.from_config(config)
        artifacts = PROJECT_ROOT / "artifacts"
        paths = {
            "frequency_plot": artifacts / "positional_encoding_frequencies.png",
            "encoding_report": artifacts / "positional_encoding_report.md",
            "architecture": artifacts / "nerf_architecture.txt",
            "parameter_report": artifacts / "model_parameter_report.md",
        }
        save_frequency_plot(paths["frequency_plot"])
        write_encoding_report(paths["encoding_report"], tests)
        write_architecture(paths["architecture"], config, model)
        write_parameter_report(paths["parameter_report"], config, model_counts(config))
        artifact_hashes = {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()
        }
        status = {
            "status": "passed" if "cuda" in probes and "device" in probes["cuda"] else "cuda_unavailable",
            "tests": tests,
            "probes": probes,
            "artifacts": {name: str(path.resolve()) for name, path in paths.items()},
            "artifact_sha256": artifact_hashes,
        }
        write_status(run_directory, status)
        logger.info("Step 2 validation: %s", json.dumps(status, ensure_ascii=False))
        return 0 if status["status"] == "passed" else 1
    except Exception as exc:
        logger.exception("Step 2 validation failed")
        write_status(run_directory, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
