from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import torch
from PIL import Image

from src.data.anomaly_transforms import build_mvtec_transforms
from src.inference.predict_anomaly import load_anomaly_model
from src.utils.config import load_yaml_config
from src.utils.device import get_device
from src.visualization.save_anomaly_maps import anomaly_map_to_heatmap, blend_overlay, tensor_to_uint8_image

CONFIG_OPTIONS = {
    "PatchCore Local": PROJECT_ROOT / "configs" / "mvtec_patchcore_local.yaml",
    "PaDiM Local": PROJECT_ROOT / "configs" / "mvtec_padim_local.yaml",
}
RUNTIME_CACHE: dict[str, dict] = {}


def load_runtime(config_label: str) -> dict:
    if config_label in RUNTIME_CACHE:
        return RUNTIME_CACHE[config_label]

    config_path = CONFIG_OPTIONS[config_label]
    cfg = load_yaml_config(config_path)
    device = get_device(str(cfg.get("device", "cpu")))
    model, checkpoint, checkpoint_path = load_anomaly_model(cfg, None, device)
    _, eval_transform, _ = build_mvtec_transforms(cfg)

    runtime = {
        "cfg": cfg,
        "device": device,
        "model": model,
        "eval_transform": eval_transform,
        "checkpoint_path": checkpoint_path,
        "image_threshold": checkpoint.get("thresholds", {}).get("image_threshold"),
        "normalization": str(cfg["data"].get("normalization", "imagenet")),
        "category": str(cfg["data"].get("category", "unknown")),
    }
    RUNTIME_CACHE[config_label] = runtime
    return runtime


def predict_image(image: Image.Image | None, config_label: str):
    if image is None:
        return "请先上传一张图像。", None, None

    try:
        runtime = load_runtime(config_label)
    except Exception as exc:
        return f"模型加载失败：{exc}", None, None

    image = image.convert("RGB")
    input_tensor = runtime["eval_transform"](image).unsqueeze(0).to(runtime["device"])

    with torch.inference_mode():
        output = runtime["model"].predict(input_tensor)

    score = float(output.image_scores[0].item())
    anomaly_map = output.anomaly_maps[0].detach().cpu()
    threshold = runtime["image_threshold"]
    if threshold is None:
        label_text = "未提供阈值，仅返回分数。"
    else:
        label_text = "anomalous" if score >= float(threshold) else "normal"

    original_image = tensor_to_uint8_image(
        input_tensor[0].detach().cpu(),
        normalization=runtime["normalization"],
    )
    heatmap = anomaly_map_to_heatmap(anomaly_map)
    overlay = blend_overlay(original_image, heatmap, alpha=0.4)

    status = (
        f"模型: {config_label}\n"
        f"类别: {runtime['category']}\n"
        f"checkpoint: {runtime['checkpoint_path']}\n"
        f"预测: {label_text}\n"
        f"score: {score:.6f}\n"
        f"threshold: {threshold if threshold is not None else 'None'}"
    )
    return status, heatmap, overlay


with gr.Blocks(title="MVTec AD Demo") as app:
    gr.Markdown(
        """
        # MVTec AD Demo

        上传一张图像，选择 PatchCore 或 PaDiM 本地配置，返回 anomaly score、热力图和 overlay。
        """
    )
    config_dropdown = gr.Dropdown(
        choices=list(CONFIG_OPTIONS.keys()),
        value="PatchCore Local",
        label="Model Config",
    )
    image_input = gr.Image(type="pil", label="Input Image")
    run_button = gr.Button("Predict")
    status_output = gr.Textbox(label="Prediction Summary", lines=6)
    heatmap_output = gr.Image(label="Anomaly Heatmap")
    overlay_output = gr.Image(label="Overlay")
    run_button.click(
        fn=predict_image,
        inputs=[image_input, config_dropdown],
        outputs=[status_output, heatmap_output, overlay_output],
    )


if __name__ == "__main__":
    app.launch()
