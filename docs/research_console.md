# NeRF Research Console V1

本控制台是 Steps 1-4 已验证的 Vanilla NeRF Python 研究链路的本地操作界面。React 只负责输入、状态和可视化；模型、射线、采样、渲染、训练和评估继续由 `src/nerf_step1/` 与既有脚本负责。控制台不自动启动位置编码带宽扫描，也不自动推进 EE 正式实验。

## 架构和数据归属

```text
console_frontend/ (React + TypeScript + Vite)
  -> REST /api 与 WebSocket /ws
src/nerf_console/ (FastAPI + ExperimentManager)
  -> 独立训练或评估进程
scripts/train_nerf.py, scripts/evaluate_nerf.py
  -> runs/<run_id>/ 的冻结配置、指标、checkpoint、图像和日志
```

实验的权威记录是磁盘上的冻结配置与实际产物。前端刷新后重新从 CSV 和文件目录读取历史，WebSocket 只传递当前事件，不能替代持久记录。所有会写入研究产物的操作都由后端处理；浏览器不能直接修改 run 目录。控制台是本地单用户工具，不提供远程认证或云部署。

## 研究配置与锁定

新实验采用完整配置快照。场景、数据集、位置/方向编码带宽、模型结构、采样、密度初始偏置、随机种子、优化器、学习率、batch size 与训练预算在创建 run 时冻结。恢复只读取原快照与 checkpoint，并通过既有兼容性检查；要改变研究参数，必须创建新的 run。`config_sha256` 是意外篡改检查，不是数字签名。

位置编码遵循本项目已有约定：原始三维输入保留，`D = 3 + 6L`；频带是 `sin(2^k πx)` 与 `cos(2^k πx)`，`k = 0…L-1`。相机仍沿 Step 1 已验证的 camera-to-world 矩阵和 OpenGL `-Z` 前向约定；前端只展示后端导出的几何数据。

### 已有 50,000 次 Lego 运行

`runs/step3_smoke_20260924T165838Z_f3bbd1f5/` 是从 Step 3 的 500 次工程运行继续到 50,000 次的既有运行，不会被导入过程重训。它的冻结配置为 **256 rays/batch**、位置 L=10、方向 L=4；`configs/baseline.yaml` 的 4096 rays/batch 是另一份配置。原始 `run_manifest.json` 的训练预算仍是 500；实际延长的里程碑和停止点记录在 `baseline_extension.json` 与 `status.json`。控制台必须同时呈现这两层来源，不能把配置快照改写为 50,000 或 4096。

该运行有真实训练 CSV、checkpoint、预览及逐视角全图评估。训练 CSV 中的 `psnr` 是训练 batch PSNR；验证/测试 PSNR、SSIM、LPIPS 来自 `evaluation_results.csv` 与 `evaluation_summary.csv`，两者不能混用。预览 PNG 是诊断图，不能当作全分辨率评估图。

## 状态、回放和缺失数据

界面区分运行中的 `LIVE`、已完成、已停止、失败与 `REPLAY`。回放仅按已经保存的指标重放，用于检查图表与界面，明确标注 **REPLAY MODE - NOT LIVE SCIENTIFIC TRAINING**。没有保存的测量值显示“不可用”，不会生成随机值、估计值或默认 0。

现有 50k run 没有逐 ray 的 coarse/fine depth、density、alpha、weight 与 transmittance **历史数值文件**。采样页可选择 checkpoint、split、视角和像素，由后端加载对应冻结模型，在 CPU 上按原 Python 采样与渲染代码做一次只读、确定性的单射线推理。它返回真实 coarse/fine 深度、密度、alpha、weight、transmittance 与检查点 SHA256，支持两个 checkpoint 对照。结果明确标为**按需重算**，不是训练时保存的测量，也不会从已有验证图反推数值。系统 GPU 利用率、温度和功耗依赖设备权限；本次检查曾遇 `nvidia-smi` 权限不足，重启后恢复真实读数。不可用时界面明确显示缺失。已有环境快照中的 GPU 名称和 CUDA 版本是**当时采集值**，不是实时测量。

现有训练器没有按固定间隔自动执行全分辨率验证的调度。`preview_interval` 仅控制诊断预览图，不能当成验证间隔；PSNR、SSIM、LPIPS 的全图验证需要在指定 checkpoint 上单独运行评估。新实验默认值和界面将 `validation_interval` 明确标为不可配置，后端若收到此参数会拒绝，而非静默忽略。

## 停止、恢复、失败

停止请求需要在完整训练 iteration 边界生效，并在可能时保存一个与该 iteration 对应的 checkpoint，再记录 `termination_reason: user_requested`。请求送达和真正停止是不同状态。恢复前必须核对冻结配置与 checkpoint，不能悄悄修改研究参数或重启失败运行。异常时保留部分 metrics、checkpoint、配置、日志及 traceback，界面显示失败；不会静默清空或重新开始。

评估针对一个明确的 checkpoint、split 和 view index 集合。既有评估脚本校验 metric metadata 与 checkpoint provenance，同一 run 的评估写入串行化。重复评估的真实结果由后端返回，前端不预设成功。

## 视觉与动效原则

控制台采用近黑石墨色工作台、绿色机器信号、少量青色辅助状态；紧凑仪表和大尺寸重建图像分出信息层级。启动序列、页面进入、数据刷新与状态变化有明确触发，环境代码雨保持低透明度。减弱动态偏好下停止连续背景运动，保留全部页面、数值、按钮和错误状态。

代码雨使用独立 Canvas 绘制数字和符号，运行中可增强强度，启动序列中的亮度更高；窗口不可见时暂停绘制，系统设置为减弱动态时保留静帧。它是装饰层，不是 GPU、训练或科研测量数据。

## 本地启动

在两个 PowerShell 终端中分别从项目根目录运行。首次使用先按根目录 `README.md` 安装 Python 和 Node 依赖。

```powershell
$env:PYTHONPATH='src'
& .\.venv\Scripts\python.exe -m uvicorn nerf_console.api:app --host 127.0.0.1 --port 8000
```

```powershell
Set-Location .\console_frontend
npm run dev
```

打开 `http://127.0.0.1:5173`。Vite 把 `/api` 与 `/ws` 转发至本机 `8000`；后端默认仅绑定回环地址。关闭两个终端即停止控制台服务，但已启动的训练工作进程以自己的状态和停止请求为准，不能把关闭网页当作已停止训练。

界面提供中文和 English 切换。首次打开按浏览器语言选择，随后将用户选择保存在本机浏览器；语言切换不改变实验配置、指标或原始日志。

## API 结构

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/api/health`、`/api/system` | 后端状态、Python/PyTorch/CUDA 与可用硬件读数 |
| GET | `/api/config/defaults`、`/api/scenes` | 新实验默认值与可用场景 |
| GET/POST | `/api/runs` | 列出历史 run、创建冻结 run；`auto_start` 可决定是否立即启动 |
| GET | `/api/runs/{id}` | 单次运行的配置、状态、环境、最新指标、可用性与迁移来源 |
| GET | `/api/runs/{id}/metrics` | 从持久 CSV 读取训练曲线，支持 `start`、`limit` |
| GET | `/api/runs/{id}/checkpoints`、`/reconstructions`、`/evaluations` | checkpoint、诊断预览、全图重建与评估 |
| GET | `/api/runs/{id}/sampling`、`/artifacts`、`/logs`、`/metadata` | 采样可用性、产物、日志和来源信息 |
| GET | `/api/runs/{id}/sampling/ray` | 按 checkpoint、split、view_index、x、y 只读重算真实单射线诊断；不写科研产物 |
| GET | `/api/runs/{id}/files/{relative}` | 白名单范围内的本地研究文件，只读提供 |
| POST | `/api/runs/{id}/start`、`/stop`、`/resume` | 训练进程控制；停止请求与已停止分开 |
| POST | `/api/runs/{id}/evaluate` | 对指定 checkpoint、split、view indices 启动后台全图评估 |
| POST | `/api/runs/{id}/replay` | 从已有 CSV 发送演示事件，不训练模型 |
| GET | `/api/scenes/lego/cameras`、`/api/positional-encoding/{L}` | Python 生成的相机/射线几何和 Step 2 编码约定 |
| WS | `/ws/runs/{id}` | 当前控制台进程中的实时事件 |

WebSocket 事件为包含 `seq`、`type`、`run_id`、`timestamp_utc` 的 JSON，并按事件附带 iteration、metrics、checkpoint 或文件 URL。主要事件有 `training_started`、`iteration_update`、`metric_update`、`checkpoint_saved`、`preview_ready`、`evaluation_started`、`evaluation_completed`、`training_completed`、`training_failed`、`replay_started` 和 `replay_completed`。重新打开页面时应先通过 REST 取得权威历史，再订阅 WebSocket；WebSocket 缓冲不承诺保存全部历史。

## 在界面中操作

1. 打开“新实验”，确认 Lego 数据、全部编码/模型/采样/训练参数和最终摘要。默认模板来自 `configs/baseline.yaml`，是未来新实验配置，不是已完成 50k run 的配置。
2. 点击启动后后端创建唯一 run 目录，保存不可变配置和环境，再用独立进程调用既有训练链路。查看“活动运行”和“指标”页；刷新页面仍可从 CSV 重建历史。
3. 停止时先显示“正在停止”，等进程在完整 iteration 边界保存 checkpoint 并确认状态。恢复时沿原 checkpoint 与原始冻结配置继续，不能修改研究变量。
4. 在“运行记录”选择已有 50k run，可查看真实评估、预览、checkpoint 时间线、日志与迁移来源。点击“回放”时页面必须明确显示回放标签；回放不会写入研究 metrics。
5. 在重建页选择 checkpoint、数据划分和视角；评估值只显示对应的全图结果。相机页从后端加载位置/射线，位置编码页展示 Step 2 公式和频带。采样页选择同一像素与两个 checkpoint 后手动重算单射线数值；原运行缺少历史采样文件这一点仍有明确标记。

## 验证边界

构建与 API 测试可证明控制台代码与本地联通。浏览历史运行、真实图像及评估属于已有数据读取验证。回放不是实际新训练；本次另做了两条独立的三次迭代受控运行：CPU run 验证启动、停止和恢复，CUDA run 验证真实 GPU 训练、检查点与一次 800×800 val 全图评估。它们不是 EE 数据或 L sweep，也没有重跑 50k baseline。GPU 利用率等权限受限字段不能因 API 返回空而宣称已监控；恢复读数后仍须区分瞬时值和历史值。
