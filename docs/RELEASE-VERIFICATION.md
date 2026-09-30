# 1.0.0 / Release verification

Verified on **30 September 2026 (Asia/Dubai)**. The audited source has been pushed to [ChefDavid0815/nerf-research-console](https://github.com/ChefDavid0815/nerf-research-console). The following checks describe the software and source archive; formal release and live-site checks are recorded in the publication record.

| Check | Result and scope |
| :--- | :--- |
| Existing research workspace Python suite | 188 passed, 1 skipped. The skip is unavailable Windows symlink creation. |
| Changed backend contracts after portable-fixture adaptation | 10 passed. Negative-input and WebSocket checks now use isolated software fixtures. |
| Source-only checkout Python suite | 185 passed, 4 skipped. Three integrations need the intentionally excluded historical 50k run/checkpoint/dataset; one needs symlink permission. |
| Frontend contracts | 13 passed in the original workspace and the source-only checkout. |
| TypeScript / Vite production build | Passed in both workspaces. Existing bundle-size advisory: about 692 kB JavaScript before gzip; this is the local software bundle, not the portfolio's procedural renderer. |
| Source-only frontend dependency install | npm ci --prefer-offline --ignore-scripts passed, 398 packages. Electron binary installation/launch was not exercised in this isolated checkout. |
| Python environment | pip check and compileall passed in the existing Python 3.12 CUDA environment. |
| Source manifest | Approximately 1.5 MB of curated source, documentation and images; no dataset, checkpoints, binaries, environment, caches, generated build output or private absolute paths staged. |
| Published evidence | Final checkpoint hash verified; 256-ray configuration and selected-view means read from the original CSVs. |
| Current workstation imagery | Actual local browser-rendered UI captured in both languages. No training was launched for screenshot capture. |

The source-only checkout used the existing installed Python interpreter, rather than a newly installed CPU environment. A fresh CPU wheel installation, additional operating systems, a standalone Windows installer and native desktop relaunch are not claimed.

The scientific model, ray, sampling, renderer and training logic were not refactored. Historical run files were only read for evidence export. The website's volume points, rays and vector cover are editorial geometry; reconstructed RGB plates and metrics are recorded outputs.

The source edition keeps the existing model and research conventions. Website and Release identities will be added to the publication record after their live verification.
