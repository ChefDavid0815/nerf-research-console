# NeRF Research Console 1.0.0 / A world, made of numbers.

An independently implemented PyTorch vanilla NeRF pipeline with a bilingual local research workstation. The first source edition connects camera rays, positional encoding, coarse/fine sampling, differentiable rendering and recoverable training to Train, Runs, Analyze and System workspaces.

Deep black, restrained phosphor green and precise engineering type follow the application into a [dedicated exhibition](https://chefzc.dev/nerf.html), a [research log](https://chefzc.dev/post.html?article=nerf) and the [School Lab](https://chefzc.dev/school-gallery.html#project-nerf).

This is a **source release**. Clone the repository and follow the Chinese or English README. Python 3.12 and Node.js 22 are the verified environment; the CUDA setup targets PyTorch 2.8.0 / CUDA 12.8. CPU requirements are provided as an alternative.

The recorded Lego evidence uses a 50,000-step continuation with **256 rays/batch**. Final validation means: 27.333 dB PSNR, 0.88850 SSIM, 0.08770 LPIPS across views 0/1/2. Final test means: 26.055 dB, 0.89022, 0.09240 across views 0/1/2/66/133. These are selected-view evaluations, not a full-dataset benchmark or a bandwidth-study conclusion.

Datasets, model checkpoints, full local run archives and Electron binaries are intentionally excluded. The desktop packaging command creates a workspace-bound folder, not a standalone installer. Opening the console does not start training or an encoding sweep.

Source publication preparation: 188 passing Python checks in the research workspace, 185 in the source-only checkout with three optional history integrations skipped; frontend 13/13 and both production builds passed. See the verification document for the additional symlink skip, installation scope and remaining compatibility work.

---

让数学，在屏幕里长出一个世界。首个源码版本公开 NeRF 管线与双语本地研究工作台，保留真实配置、指标与重建记录。数据集、checkpoint 与依赖工作区的桌面二进制不分发；没有将选定视角的评估写成完整 benchmark，也不会自动启动研究实验。

Made by ChefZC.
