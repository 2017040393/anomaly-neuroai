## NeuroAI Project

这是一个面向工业视觉异常检测 / NeuroAI 起步实验的最小项目模板。第一阶段目标不是复现 SOTA，而是先把训练、评估、推理 demo 这条链路跑通，并保留清晰的扩展位，方便后续接入 PatchCore、PaDiM、EfficientAD、DRAEM 等方法。

### 当前能力

- 使用 `uv` 管理环境与依赖
- 使用 `pyproject.toml`
- 使用 `src/` 目录结构
- 提供 synthetic dataset，方便本地快速验证训练流程
- 支持 `mlp` 和 `snn_mlp` 两种最小模型
- 提供训练、评估脚本
- 提供一个基于 feature 输入的 Gradio demo
- 第二阶段支持 MVTec AD 数据读取
- 第二阶段支持 `PatchCore` 和 `PaDiM` anomaly detection wrapper
- 第二阶段支持 anomaly training / inference / latency benchmark / visualization / Gradio demo

### 项目结构

```text
neuroai-project/
├─ configs/
├─ data/
├─ demo/
├─ checkpoints/
├─ results/
├─ src/
│  ├─ data/
│  ├─ models/
│  ├─ training/
│  └─ utils/
└─ tests/
```

### 安装

```bash
uv venv
uv pip install -e ".[dev]"
```

如果你还想尝试事件数据或导出能力，可以额外安装：

```bash
uv pip install -e ".[dev,event,export]"
```

### 训练

快速 smoke/debug：

```bash
uv run python -m src.training.train --config configs/local_debug.yaml
```

本地小规模训练：

```bash
uv run python -m src.training.train --config configs/local_train.yaml
```

云端训练配置示例：

```bash
uv run python -m src.training.train --config configs/cloud_train.yaml
```

### 评估

```bash
uv run python -m src.training.eval --config configs/local_train.yaml --ckpt checkpoints/local/best.pt
```

如果不显式传 `--ckpt`，评估脚本会默认读取配置中 `checkpoint_dir` 下的 `best.pt`。

### Demo

先训练出一个 checkpoint，再启动 demo：

```bash
uv run python demo/app.py
```

输入框使用逗号分隔的 feature 数值，例如：

```text
0.1, -0.4, 1.2, 0.8, ...
```

### 第二阶段：MVTec AD 运行命令

先把 MVTec AD 数据集解压到：

```text
data/raw/mvtec_ad/
```

例如：

```text
data/raw/mvtec_ad/bottle/train/good/
data/raw/mvtec_ad/bottle/test/good/
data/raw/mvtec_ad/bottle/test/broken_large/
data/raw/mvtec_ad/bottle/ground_truth/broken_large/
```

#### PatchCore 本地快速验证

训练：

```bash
uv run python -m src.training.train_anomaly --config configs/mvtec_patchcore_local.yaml
```

推理：

```bash
uv run python -m src.inference.predict_anomaly --config configs/mvtec_patchcore_local.yaml
```

延迟测试：

```bash
uv run python -m src.inference.latency_benchmark --config configs/mvtec_patchcore_local.yaml
```

绘制指标图：

```bash
uv run python -m src.visualization.plot_metrics --metrics-csv results/runs/mvtec_patchcore_bottle_local/metrics.csv
```

#### PaDiM 本地快速验证

训练：

```bash
uv run python -m src.training.train_anomaly --config configs/mvtec_padim_local.yaml
```

推理：

```bash
uv run python -m src.inference.predict_anomaly --config configs/mvtec_padim_local.yaml
```

延迟测试：

```bash
uv run python -m src.inference.latency_benchmark --config configs/mvtec_padim_local.yaml
```

绘制指标图：

```bash
uv run python -m src.visualization.plot_metrics --metrics-csv results/runs/mvtec_padim_bottle_local/metrics.csv
```

#### 云端配置示例

PatchCore：

```bash
uv run python -m src.training.train_anomaly --config configs/mvtec_patchcore.yaml
uv run python -m src.inference.predict_anomaly --config configs/mvtec_patchcore.yaml
uv run python -m src.inference.latency_benchmark --config configs/mvtec_patchcore.yaml
```

PaDiM：

```bash
uv run python -m src.training.train_anomaly --config configs/mvtec_padim.yaml
uv run python -m src.inference.predict_anomaly --config configs/mvtec_padim.yaml
uv run python -m src.inference.latency_benchmark --config configs/mvtec_padim.yaml
```

#### 第二阶段 Demo

先训练出本地 checkpoint，再启动 MVTec demo：

```bash
uv run python demo/app_mvtec.py
```

#### 常见输出位置

PatchCore 本地：

```text
results/runs/mvtec_patchcore_bottle_local/
checkpoints/local/mvtec_patchcore/best.pt
results/runs/mvtec_patchcore_bottle_local/predictions/predictions.csv
results/runs/mvtec_patchcore_bottle_local/visualizations/
```

PaDiM 本地：

```text
results/runs/mvtec_padim_bottle_local/
checkpoints/local/mvtec_padim/best.pt
results/runs/mvtec_padim_bottle_local/predictions/predictions.csv
results/runs/mvtec_padim_bottle_local/visualizations/
```

### 说明

当前仓库只实现了第一阶段最小骨架：

- 数据是 synthetic feature classification，用来验证工程链路
- 模型是最小 MLP / SNN baseline
- demo 是 feature 级别推理，不是图像 UI

当前已进入第二阶段最小扩展：

- 支持 MVTec AD 数据读取与 anomaly detection 基础流程
- 支持 PatchCore / PaDiM wrapper
- 支持 anomaly map 与 overlay 保存
- 支持最小 MVTec Gradio demo

后续还可以继续扩展：

- EfficientAD
- DRAEM
- 更完整的 anomaly visualization
- 更多工业视觉数据集与部署流程
