# The published engineering record

Read-only exports from step3_smoke_20260924T165838Z_f3bbd1f5, made on 30 September 2026. Original records and checkpoints were not edited.

- **config_snapshot.yaml:** the original 500-step, 256-ray smoke configuration. A separate extension reached 50,000; the snapshot is intentionally unchanged.
- **evaluation_results.csv:** per-view full-resolution measurements.
- **evaluation_summary.csv:** aggregates over recorded view sets, across checkpoints.
- **training-curve.csv:** every 250th original row plus the last row; a presentation sample, not the full 50,000-row record.
- **publication-manifest.json:** scene, dimensions, exact final view sets and checkpoint hash.

The final published test table uses all five views 0,1,2,66,133. A spaced three-view subset previously used in a local report has a different mean. Validation uses 0,1,2. These are selected-view checks.

Dataset and checkpoints remain outside Git. The exports support inspection of reported values; inference reproduction requires the corresponding checkpoint and dataset or a new training run.
