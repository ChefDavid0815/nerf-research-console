"""Read-only adapters from validated NeRF artifacts to console-safe records.

No adapter here loads a checkpoint, performs a render, or changes a research
artifact. All file paths returned to callers are relative to their run root.
"""

from .readers import (
    camera_geometry,
    hierarchical_sampling_artifacts,
    list_reconstructions,
    positional_encoding_info,
    read_evaluations,
    read_run_inventory,
    read_run_metadata,
    resolve_run_directory,
    write_baseline_migration_manifest,
)

__all__ = [
    "camera_geometry",
    "hierarchical_sampling_artifacts",
    "list_reconstructions",
    "positional_encoding_info",
    "read_evaluations",
    "read_run_inventory",
    "read_run_metadata",
    "resolve_run_directory",
    "write_baseline_migration_manifest",
]
