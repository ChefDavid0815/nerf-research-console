"""Small, reusable run ledger for configuration, environment and status."""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


def environment_snapshot() -> dict[str, Any]:
    """Record software and CUDA state visible to the active Python process."""
    cuda_available = torch.cuda.is_available()
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        nvidia_driver = probe.stdout.strip().splitlines()[0] if probe.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, IndexError):
        nvidia_driver = None
    free_vram = torch.cuda.mem_get_info(0)[0] if cuda_available else None
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "operating_system": platform.platform(),
        "pytorch": torch.__version__,
        "pytorch_cuda_runtime": torch.version.cuda,
        "pytorch_cuda_available": cuda_available,
        "nvidia_driver": nvidia_driver,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_compute_capability": torch.cuda.get_device_capability(0) if cuda_available else None,
        "gpu_total_vram_bytes": (
            torch.cuda.get_device_properties(0).total_memory if cuda_available else None
        ),
        "gpu_free_vram_bytes_at_capture": free_vram,
    }


def start_run(base_directory: Path, label: str, config: dict[str, Any]) -> tuple[Path, logging.Logger]:
    """Create a unique run folder with frozen inputs and a text log."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_directory = base_directory / f"{label}_{timestamp}_{uuid.uuid4().hex[:6]}"
    run_directory.mkdir(parents=True, exist_ok=False)
    (run_directory / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (run_directory / "environment.json").write_text(
        json.dumps(environment_snapshot(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger = logging.getLogger(f"nerf_step1.{run_directory.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(run_directory / "run.log", encoding="utf-8"),
                    logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.info("Run directory: %s", run_directory.resolve())
    return run_directory, logger


def write_status(run_directory: Path, status: dict[str, Any]) -> None:
    (run_directory / "status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
    )
