# NeRF positional-encoding bandwidth experiment - Research prototype and local console

**研究问题**： “To what extent does positional encoding bandwidth affect the reconstruction accuracy of Neural Radiance Fields across scenes of varying spatial-frequency complexity?”

本仓库已建立数据、相机射线、显式正弦位置编码、vanilla NeRF MLP、coarse/fine 采样与体渲染，以及可恢复训练链路。Step 3 已在官方 Lego 上完成 500 次迭代的**工程 smoke run**；这不是正式 EE 实验，也没有进行 L sweep、跨场景比较或研究结论。未来实验会把 `position_encoding_L` 作为主要自变量，在场景复杂度不同的数据上训练受控的模型，并以重建质量作因变量。

## 目录

```text
src/nerf_step1/        Blender 数据、相机、NeRF 模型、采样、渲染、预览与训练记录
configs/               受控 baseline 与独立的工程 smoke 配置
data/                  官方数据及来源记录（大文件不纳入版本控制）
tests/                 自动测试；仅 Step 1 数据测试使用临时微型夹具
scripts/               Step 1 / Step 2 验证、Step 3 训练和报告入口
runs/                  验证/训练的配置、环境、指标、checkpoint 与预览
artifacts/             数学核对、诊断图、smoke 曲线与完成报告
```

## 安装

本机使用 Python **3.12** 的独立虚拟环境。`requirements.txt` 固定直接依赖，`requirements-lock.txt` 记录本次安装的全部依赖版本；PyTorch 固定为 2.8.0 的 CUDA 12.8 wheel。安装需要足够的磁盘空间及网络流量。PyTorch 官方列出该版本的 [CUDA 12.8 安装方式](https://pytorch.org/get-started/previous-versions/)。

在 PowerShell、项目根目录执行：

```powershell
# 用已安装的 Python 3.12 创建虚拟环境；若 py 未注册 3.12，
# 先用 py -0p 找到解释器，再用该 python.exe 执行 -m venv .venv。
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

本机的 Python 3.12 解释器未注册到 `py -3.12`；实际创建环境时使用了
`<your-python-3.12-installation>/python.exe -m venv .venv。
系统默认的 Python 3.14 与此项目环境不是同一个解释器。

## 官方 Blender 数据

标准布局如下：

```text
data/nerf_synthetic/lego/
  transforms_train.json
  transforms_val.json
  transforms_test.json
  train/r_0.png ...
  val/r_0.png ...
  test/r_0.png ...
```

作者的 [示例数据下载脚本](https://github.com/bmild/nerf/blob/master/download_example_data.sh) 提供 [Lego 示例压缩包](https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip)；完整 NeRF synthetic 数据集入口见[原仓库 README](https://github.com/bmild/nerf/blob/master/README.md)。把解压后的 `lego` 文件夹放在上述位置。项目不会自动生成或替换研究数据。若场景缺失，验证脚本会明确失败并在 `runs/` 留下失败状态。下载时的来源和校验值记录在 `data/` 的来源文件中。

加载器分别暴露 `train`、`val`、`test`，逐一核对图像路径、尺寸和相机矩阵；图片按需读取。Blender 原图是 RGBA。当前配置的 `white_background: true` 把 RGBA 显式合成为 RGB：`rgb × alpha + white × (1 - alpha)`。`image_resolution: original` 保留原始尺寸；没有重采样。相机 `transform_matrix` 以 float64 保留 JSON 的 camera-to-world 数值和帧顺序，不做轴翻转、居中或缩放。三个 split 的内参必须一致；否则会报错，因为此阶段不实现逐视角内参。

## 射线约定

`src/nerf_step1/rays.py` 单独实现原版 NeRF 的 OpenGL 相机约定：局部 +X 向右、+Y 向上、**−Z 朝前**。图像从左上角开始编号，像素 `(x, y)` 使用整数索引，不额外加 `0.5`；主点为 `(W/2, H/2)`。标准 Blender 元数据的水平视角 `camera_angle_x` 为弧度，`fx = fy = 0.5 × W / tan(camera_angle_x/2)`，这是原版加载器的方形像素假设。世界方向由相机旋转矩阵变换，射线起点是相机平移。依据见原版 [Blender 加载器](https://github.com/bmild/nerf/blob/master/load_blender.py)和[射线函数](https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py)。

默认射线方向**不归一化**，匹配原版 NeRF 的参数化；`normalize_directions=True` 只在需要单位方向时显式使用。将来的采样边界必须与选定方向长度一致。相机图使用单位向量便于比较朝向。图中把世界原点作为参考点；正向点积只能检查相机大体朝向，不能单独证明每个图像像素几何配准。

## Step 1 验证

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe scripts\validate_step1.py --config configs\baseline.yaml
```

测试使用**仅用于软件测试**的微型临时图片和矩阵，不产生研究观测值。第二条命令实际读取官方 Lego 的三个 split、每组的图像与姿态、生成中心射线，并保存：

- `artifacts/environment_report.md`：机器和运行环境记录。
- `artifacts/step1_validation_*/camera_rays_lego.png`：训练相机位置、朝向和少量射线的 3D 图。
- `artifacts/step1_validation_*/dataset_views_lego.png`：真实数据图像及 split、索引、位置、朝向。
- `runs/step1_validation_*/`：本次配置副本、Python/PyTorch/CUDA 环境快照、`run.log` 与 `status.json`。

检查 `status.json` 的 `status`、已解码视图数量、数据内容 SHA256、split 数量和相机朝世界原点的余弦值，同时人工查看两张图。一个运行记录对应一次验证，图也存入唯一目录，不覆盖旧记录。未来训练脚本可以复用 `run_logging.py` 记录同样的配置与环境。`white_background: false` 在当前加载器中表示黑底 alpha 合成；这与原版非白底分支直接舍弃 alpha 的行为不同，未来若使用该配置必须明确写入实验方法。

`batch_size: 4096` 是本研究要求的控制值，**并非**原作者 Blender 论文配置文件中 `N_rand: 1024` 的逐项复刻。其他网络和采样参数也只作为未来训练的初始配置，当前未执行。原版 Blender 论文配置可在 [paper_configs/blender_config.txt](https://github.com/bmild/nerf/blob/master/paper_configs/blender_config.txt) 对照。

## Step 2：位置编码与未训练的 NeRF MLP

`src/nerf_step1/positional_encoding.py` 对每个输入分量使用
`[p, sin(2^0 πp), cos(2^0 πp), …, sin(2^(L−1) πp), cos(2^(L−1) πp)]`。
原始坐标总是保留，所以三维输出是 `3 + 6L`：`L=0` 是原始 xyz，默认位置
`L=10` 是 **63** 维，默认 viewing direction `L=4` 是 **27** 维。频带由构造时的
`L` 产生；研究运行中不能改变 `L`。一组实验应只改变 `position_encoding_L`，
并把其他模型设置、场景和 seed 保存在每次运行的完整配置快照中。

这一公式遵从本研究明确指定的 `π`。原始 [NeRF 论文](https://arxiv.org/pdf/2003.08934)
使用 `π` 表述映射，而作者[发布的 TensorFlow embedder](https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py)
使用 `2^k` 而没有 `π`。发布代码保留原始输入，因此这里的 63/27 维与论文架构图
只标 Fourier 通道的 60/24 维标签也不同。这些约定必须在未来 EE 方法部分披露。

`src/nerf_step1/model.py` 的 `VanillaNeRF.from_config(config)` 从配置构造位置与
方向的独立编码器、八层默认 256 通道的 ReLU 位置分支、密度与 feature 分支，以及
一层默认 128 通道的 viewing-direction 颜色分支。默认 skip index 是 4（从零开始）：
**第 5 个位置层 ReLU 后**把编码位置重新拼接，因此第 6 层接受拼接后的输入。
`forward(positions, directions)` 接受两个 `[N,3]` 张量，返回 `[N,3]` 的
`rgb` 和 `[N,1]` 的 `sigma`；模型内部完成编码，RGB 限定在 `[0,1]`，密度非负。
输出激活在作者原版代码中位于 renderer；此项目把约束放在模型的输出接口以匹配
Step 2 要求。方向编码采用调用方提供的 viewing direction；为符合原版用法，
未来从 Step 1 的未归一化射线接入时应先得到单位 viewing direction，不能修改
现有射线生成约定。

运行完整单元测试及诊断：

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe scripts\validate_step2.py --config configs\baseline.yaml
```

第二条命令会再次运行完整 pytest，执行配置模型的 CPU/CUDA forward/backward
sanity check，并保存完整配置和环境快照到唯一的 `runs/step2_validation_*/`。
它会生成以下可复核产物：

- `artifacts/positional_encoding_frequencies.png`：`L=1,2,4,6,10` 的实际编码曲线，展示完整 `p∈[-1,1]` 区间和高频局部放大。
- `artifacts/positional_encoding_report.md`：公式、维度、实际频带、测试结果和实现约定。
- `artifacts/nerf_architecture.txt`：每层输入输出 shape、skip、密度/颜色分支与参数总量。
- `artifacts/model_parameter_report.md`：仅改变位置 `L=0,2,4,6,8,10,12,15` 时的实际模型参数量。

这些 Step 2 检查只验证当时尚未训练的模型接口和数学运算；后续 Step 3
才首次建立训练闭环。Step 2 图表本身不能用于重建精度结论。

## Step 3：coarse/fine 体渲染与工程 smoke run

`configs/baseline.yaml` 保留原有 L=10/4、64/128 samples、4096 rays、
5e-4 学习率、8×256/128 网络和 500000 iterations；新增 near=2、far=6、
显式密度初始偏置、学习率衰减及运行设置。`configs/smoke.yaml` 是单独的
工程验证配置：同一研究编码与网络结构，batch 256、500 iterations、
100×100 preview。不会自动运行正式 baseline 或 L sweep。

采样沿 Step 1 原版未归一化射线 `o + td`。`t` 是射线参数，体渲染用
`(t_(i+1)-t_i) × ||d||` 转成世界距离；只有输入 MLP 的 viewing direction
被归一化。最后一个间距按原作者实现设为 `1e10 × ||d||`，白底残差与
Blender RGBA 合成方式一致。位置编码仍是研究指定的含 `π` 公式。
coarse 权重决定 fine inverse-CDF 抽样，合并后输入第二个独立的 Step 2
`VanillaNeRF`。总 loss 是 coarse/fine RGB MSE 之和，PSNR 来自 fine MSE。

在项目根目录运行（默认使用 `smoke.yaml`）：

```powershell
& .\.venv\Scripts\python.exe scripts\train_nerf.py --config configs\smoke.yaml --stop-after 50 --device cuda
& .\.venv\Scripts\python.exe scripts\train_nerf.py --resume runs\<上一步输出的 run_id> --device cuda
& .\.venv\Scripts\python.exe -m pytest -q
```

`--stop-after` 只模拟一次可恢复的中断，不改变快照中的总迭代数；恢复时
检查完整配置、加载 coarse/fine 参数、Adam 与 scheduler 状态及随机状态，
从下一次 iteration 继续。每个 run 保存 `config_snapshot.yaml`、
`environment_snapshot.json`、`run_manifest.json`、`metrics.csv`、
`checkpoints/` 和 `renders/`。预览使用验证集第一个视角、原图对齐的
整数像素和分块 deterministic rendering；每个 iteration 单独保存预测、
ground truth、绝对差异和对照图，已有预览不会被覆盖。

本机 Step 3 smoke run 的可复查结果见
`artifacts/training_pipeline_report.md`、`volume_rendering_validation.md`、
`hierarchical_sampling_validation.md`、`smoke_training_curve.png` 和
`smoke_preview.png`。图像只有 500 次迭代，能看到模糊的 Lego 轮廓，
不代表正式收敛质量或 EE 研究数据。后续收敛与全图评估见 Step 4。

## Step 4：Lego baseline 收敛与全图评估

Step 4 从上述 Step 3 smoke run 的 `iter_000500.pt` **续训**。该 checkpoint 的不可变配置使用
256 条训练射线一批；`configs/baseline.yaml` 的 4096 条射线是另一份配置，不能在同一次
checkpoint 恢复中悄悄替换。位置与方向编码带宽仍为 10/4，coarse/fine 网络、密度偏置、
采样、白底、Adam、学习率调度及随机状态均从 checkpoint 继续。迭代上限和 checkpoint
输出时点作为单独的 Step 4 运行控制记录在 `baseline_extension.json`，不改写原配置快照。
以下是本次执行记录；50,000 次已完成，不能在同一 run 上重复执行相同续训命令。

```powershell
& .\.venv\Scripts\python.exe scripts\train_nerf.py --resume runs\step3_smoke_20260924T165838Z_f3bbd1f5 --device cuda --extend-to 50000 --stage-checkpoints 1000,2000,5000,10000,15000,20000,25000,30000,35000,40000,45000,50000
& .\.venv\Scripts\python.exe scripts\evaluate_nerf.py --run runs\step3_smoke_20260924T165838Z_f3bbd1f5 --checkpoint 50000 --split val --views 0 1 2 --device cuda --render-chunk-size 1024
& .\.venv\Scripts\python.exe scripts\report_step4.py --run runs\step3_smoke_20260924T165838Z_f3bbd1f5 --selected-iteration 50000 --pytest-xml runs\step3_smoke_20260924T165838Z_f3bbd1f5\pytest_step4.xml
```

`evaluate_nerf.py` 每次只评估一个明确的 split，按原始 800×800 像素完整渲染指定相机视角。
对 train、val、test 分别调用它；不要并行运行多个评估进程写同一 CSV。结果进入
`evaluation_results.csv`、`evaluation_summary.csv` 和按 checkpoint/split/view 命名的图像。
PSNR 使用 RGB MSE；SSIM 使用固定版本的 scikit-image；LPIPS 使用官方 AlexNet v0.1
权重与明确的 `[0,1]` 到 `[-1,1]` 归一化。指标在浮点 RGB 上计算，PNG 只用于展示。
`evaluation/metric_metadata.json` 保存精确版本、参数与权重 SHA256；后续追加结果前会校验
指标实现未漂移。`evaluation/checkpoint_provenance.json` 记录每个被评估 checkpoint 的 SHA256；
早于此机制的评估明确标为按当前文件追补。相同 run 的评估结果写入由跨进程锁串行化。
报告脚本只从已经保存的训练、评估和 checkpoint 证据生成图表与分析。
本阶段不扫描其他位置编码带宽，也不把工程性能当作研究问题的结论。

最终 Step 4 产物位于 `artifacts/baseline_convergence_report.md`、
`artifacts/evaluation_metric_report.md`、`artifacts/baseline_performance_report.md`、
`artifacts/baseline_metrics_curve.png`、`artifacts/baseline_reconstruction_progress.png`、
`artifacts/novel_view_validation.png` 与 `artifacts/hierarchical_sampling_after_training.png`。
逐视角原始指标及拆分汇总位于上述 run 的 `evaluation_results.csv` 和
`evaluation_summary.csv`。50,000 次是本次用户指定的停止点；是否达到实用收敛、
以及未来固定预算是否可用，按报告中的实际验证曲线和批量配置差异判断。

## Step 4.5：本地 NeRF Research Console

`console_frontend/` 是 React、TypeScript、Vite 研究界面；`src/nerf_console/` 是 FastAPI 和独立进程实验管理器；`src/nerf_console_diagnostics/` 只读取已有研究产物与 Step 1/2 数据。科学模型、采样、损失、渲染及全图评估仍由原 Python 链路执行。详细架构、REST/WS 契约与操作边界见 [docs/research_console.md](docs/research_console.md)。

首次启动前，在项目根目录安装新增 Python 依赖，然后安装前端依赖：

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
Set-Location .\console_frontend
npm install
```

之后分别开两个 PowerShell 终端：

```powershell
# 后端：项目根目录
$env:PYTHONPATH='src'
& .\.venv\Scripts\python.exe -m uvicorn nerf_console.api:app --host 127.0.0.1 --port 8000
```

```powershell
# 前端：项目根目录
Set-Location .\console_frontend
npm run dev
```

浏览器打开 `http://127.0.0.1:5173`。默认只在本机运行，不发布到云端。控制台启动本身不会进行训练或 L sweep。

界面支持中文与 English 切换，首次按浏览器语言选择，并在本机保存偏好。切换语言不修改科研数据。

“运行记录”已纳入既有的 50,000 次 Lego 运行，不重训。该运行的科学配置快照仍是 Step 3 的 **500 次预算、256 rays/batch**；实际延长到 50,000 次的来源是 `baseline_extension.json` 与 `status.json`。独立的 `console_migration_manifest.json` 标记导入字段来源与缺失项，不覆盖原始 manifest。`configs/baseline.yaml` 的 500,000 次、4096 rays/batch 用于未来**新实验**，不能套在既有 checkpoint 上。

创建新实验时先检查完整摘要，启动后研究参数锁定。停止请求在完整训练迭代边界保存 checkpoint 并写 `termination_reason: user_requested`；恢复会校验配置与 checkpoint。训练曲线由持久 `metrics.csv` 重载，WebSocket 只传实时变化。回放用真实已存指标测试界面，必须显示 “REPLAY MODE - NOT LIVE SCIENTIFIC TRAINING”，不会启动科学训练或追加训练 CSV。

现有 50k run 没有训练时保存的逐 ray 层级采样数值；诊断页可从指定 checkpoint 在 CPU 上只读重算所选像素的 coarse/fine 深度、密度、alpha、权重和透射率，并对照两个 checkpoint。结果标为按需计算，原有验证图仍单独展示。GPU 遥测依赖本地进程权限：本次检查中曾返回权限不足，重启后恢复真实读数；界面不把空值显示成零。Step 4.5 不自动开始 Step 5 的正式实验。

现有训练器尚无周期性全图验证调度；新实验页会将验证间隔标为不可配置，`preview_interval` 仅生成诊断预览。需要全图 PSNR、SSIM、LPIPS 时，对明确的 checkpoint、split 和视角单独运行评估。
