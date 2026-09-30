"""Console worker process delegating to the validated training script."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import train_nerf  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    run = args.run.resolve()
    options = argparse.Namespace(
        config=None if args.resume else run / "config_snapshot.yaml",
        resume=run if args.resume else None,
        run_directory=None if args.resume else run,
        stop_file=run / "console_stop.request",
        stop_after=None,
        extend_to=None,
        stage_checkpoints=None,
        device=args.device,
        label=run.name,
    )
    result = train_nerf.train(options)
    print(json.dumps({"run_directory": str(result.resolve())}))


if __name__ == "__main__":
    main()
