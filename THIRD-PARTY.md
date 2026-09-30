# Method, data and dependencies

NeRF: Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi and Ren Ng, *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*, ECCV 2020. [Project](https://www.matthewtancik.com/nerf) · [Paper](https://arxiv.org/abs/2003.08934) · [Original code](https://github.com/bmild/nerf).

The Python pipeline independently implements the documented mathematics and conventions. Differences are disclosed in the README and implementation notes.

Lego comes from the authors' NeRF synthetic archive: [data provenance](data/PROVENANCE.md). Small reconstruction examples derive from that scene. Reference imagery belongs to its creators and is included as attributed research illustration, without being relicensed as original artwork. The dataset itself is excluded.

Workstation screenshots are actual local captures from 30 September 2026; hardware readouts are dated snapshots. The SVG cover is original editorial artwork, not a computed density field. Portfolio spatial particles are illustrative.

Dependencies retain their package licences. Major components: PyTorch / torchvision (BSD-style), NumPy (BSD), Pillow (HPND), Matplotlib (PSF-based), scikit-image (BSD), LPIPS (BSD), FastAPI (MIT), React (MIT), Vite (MIT), Three.js (MIT), Recharts (MIT), Framer Motion (MIT), Tabler Icons (MIT) and Electron (MIT plus Chromium notices). No Python wheels, Node modules or Electron binaries are redistributed.
