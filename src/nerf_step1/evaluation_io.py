"""Cross-process evaluation commits and checkpoint-file provenance."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def evaluation_write_lock(run_directory: Path, *, timeout_seconds: float = 300.0) -> Iterator[None]:
    """Serialize read/merge/replace writes made by evaluators of one run.

    This is an OS advisory lock, released when a process exits or crashes.
    Every evaluation writer must acquire it around provenance and CSV commits.
    """
    run_directory = Path(run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    lock_path = run_directory / "evaluation" / ".write.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    with lock_path.open("a+b") as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"\0")
            stream.flush()
        if os.name == "nt":
            import msvcrt

            while True:
                stream.seek(0)
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"timed out waiting for evaluation write lock: {lock_path}") from exc
                    time.sleep(0.05)
        else:
            import fcntl

            while True:
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as exc:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"timed out waiting for evaluation write lock: {lock_path}") from exc
                    time.sleep(0.05)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_path(run_directory: Path, iteration: int) -> Path:
    return run_directory / "checkpoints" / f"iter_{iteration:06d}.pt"


def _evaluated_iterations(csv_path: Path) -> set[int]:
    if not csv_path.is_file():
        return set()
    with csv_path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "iteration" not in reader.fieldnames:
            raise ValueError("evaluation CSV has no iteration column")
        return {int(row["iteration"]) for row in reader}


def _atomic_json(path: Path, value: dict) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=".checkpoint_provenance_", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def verify_checkpoint_provenance(
    run_directory: Path, requested_iteration: int, loaded_checkpoint_sha256: str
) -> Path:
    """Bind each evaluated iteration to its checkpoint bytes before CSV writes.

    Existing rows from earlier Step 4 invocations are safely *backfilled* from
    their present checkpoint files. The marker explicitly avoids claiming that
    those files were hashed at the time the older metrics were first scored.
    Caller must hold `evaluation_write_lock` while calling this function.
    """
    run_directory = Path(run_directory)
    if isinstance(requested_iteration, bool) or not isinstance(requested_iteration, int) or requested_iteration < 0:
        raise ValueError("requested iteration must be a nonnegative integer")
    manifest_path = run_directory / "evaluation" / "checkpoint_provenance.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("cannot read checkpoint provenance manifest") from exc
        if not isinstance(manifest, dict) or manifest.get("format_version") != 1 or manifest.get("algorithm") != "sha256" or not isinstance(manifest.get("checkpoints"), dict):
            raise ValueError("unsupported checkpoint provenance manifest")
    else:
        manifest = {"format_version": 1, "algorithm": "sha256", "checkpoints": {}}
    entries = manifest["checkpoints"]
    evaluated = _evaluated_iterations(run_directory / "evaluation_results.csv")
    for key, entry in entries.items():
        if not key.isdecimal() or not isinstance(entry, dict) or entry.get("source") not in (
            "recorded_before_evaluation", "backfilled_from_current_file_after_existing_metrics"
        ):
            raise ValueError("checkpoint provenance contains an invalid entry")
        checkpoint_path = _checkpoint_path(run_directory, int(key))
        expected_relative_path = str(checkpoint_path.relative_to(run_directory).as_posix())
        if entry.get("relative_path") != expected_relative_path or entry.get("sha256") != file_sha256(checkpoint_path):
            raise ValueError(f"checkpoint provenance mismatch at iteration {key}")
    for iteration in sorted(evaluated - {int(key) for key in entries}):
        checkpoint_path = _checkpoint_path(run_directory, iteration)
        entries[str(iteration)] = {
            "relative_path": str(checkpoint_path.relative_to(run_directory).as_posix()),
            "sha256": file_sha256(checkpoint_path),
            "source": "backfilled_from_current_file_after_existing_metrics",
        }
    requested_path = _checkpoint_path(run_directory, requested_iteration)
    current_digest = file_sha256(requested_path)
    if current_digest != loaded_checkpoint_sha256:
        raise ValueError("checkpoint bytes changed while evaluation was running")
    requested_key = str(requested_iteration)
    if requested_key in entries:
        if entries[requested_key]["sha256"] != loaded_checkpoint_sha256:
            raise ValueError("requested checkpoint differs from recorded provenance")
    else:
        entries[requested_key] = {
            "relative_path": str(requested_path.relative_to(run_directory).as_posix()),
            "sha256": loaded_checkpoint_sha256,
            "source": "recorded_before_evaluation",
        }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(manifest_path, manifest)
    return manifest_path
