# Dataset provenance: NeRF synthetic Lego

- Source: original NeRF project example-data archive, [`nerf_example_data.zip`](https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip).
- Retrieved: 2026-09-24 (Asia/Dubai), over HTTPS. The server returned HTTP 200, `Content-Type: application/zip`, `Content-Length: 370385516`, and `Last-Modified: Thu, 21 Jan 2021 15:40:04 GMT`.
- Complete downloaded ZIP: 370,385,516 bytes; SHA256 `CE4E94E031C099A19EF04CFB6C71F1E47225D97D365BE610B476E379A386C25F`. This hash identifies the bytes downloaded here; no independently published upstream checksum was checked.
- Extracted subset: only `nerf_synthetic/lego/` into `data/nerf_synthetic/lego/`. No other scene or LLFF data was extracted. Two macOS `.DS_Store` metadata files in that subtree were omitted.
- Files retained: 3 transform JSON files and 800 PNGs. The JSON files reference 100 train, 100 validation and 200 test color images (400 total). The test folder additionally contains 200 depth and 200 normal PNGs; these are **not** referenced as color views by the transforms and should not be counted as additional camera frames.
- Checked after extraction: each referenced color PNG exists exactly once; each frame has a finite numeric 4×4 `transform_matrix`; each split has `camera_angle_x = 0.6911112070083618`; all 800 PNG headers report 800×800, 8-bit, PNG color type 6 (RGBA). ZIP members were selected and copied with path traversal, symlink, encryption and duplicate-name checks; reading the selected files also checked their ZIP CRCs.

The PNGs contain an alpha channel. Any future RGB compositing policy must be recorded as part of experiment configuration and applied identically to every bandwidth condition. Camera matrices and frame paths are kept in their source files without convention changes.
