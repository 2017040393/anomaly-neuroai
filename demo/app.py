from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import torch

from src.models.build_model import build_model
from src.utils.config import load_yaml_config
from src.utils.device import get_device

CONFIG_PATH = PROJECT_ROOT / "configs" / "local_train.yaml"


def load_runtime():
    cfg = load_yaml_config(CONFIG_PATH)
    device = get_device(str(cfg.get("device", "cpu")))
    model = build_model(cfg).to(device)

    checkpoint_path = PROJECT_ROOT / cfg["save"]["checkpoint_dir"] / "best.pt"
    status_message = "checkpoint loaded"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        status_message = (
            f"checkpoint missing: {checkpoint_path}. "
            "Run `uv run python -m src.training.train --config configs/local_train.yaml` first."
        )
    model.eval()
    return cfg, device, model, status_message


CFG, DEVICE, MODEL, MODEL_STATUS = load_runtime()


def predict_from_features(raw_text: str):
    if MODEL_STATUS != "checkpoint loaded":
        return MODEL_STATUS, {}

    parts = [part.strip() for part in raw_text.split(",") if part.strip()]
    if not parts:
        return "请输入逗号分隔的 feature 数值。", {}

    try:
        values = [float(part) for part in parts]
    except ValueError:
        return "输入中包含无法解析为浮点数的内容。", {}

    expected_dim = int(CFG["data"]["num_features"])
    if len(values) != expected_dim:
        return f"输入维度不正确，期望 {expected_dim} 个值，实际收到 {len(values)} 个。", {}

    inputs = torch.tensor([values], dtype=torch.float32, device=DEVICE)
    with torch.inference_mode():
        logits = MODEL(inputs)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    predicted_class = int(torch.argmax(probs).item())
    prob_dict = {f"class_{index}": float(score) for index, score in enumerate(probs.tolist())}
    return f"预测类别: class_{predicted_class}", prob_dict


example = ",".join(["0.0"] * int(CFG["data"]["num_features"]))

with gr.Blocks(title="NeuroAI Synthetic Feature Demo") as app:
    gr.Markdown(
        """
        # NeuroAI Synthetic Feature Demo

        输入逗号分隔的 feature 数值，返回预测类别与 softmax 概率。
        """
    )
    with gr.Row():
        feature_input = gr.Textbox(
            label="Feature Input",
            lines=4,
            placeholder=example,
        )
    status_box = gr.Textbox(label="Prediction")
    prob_box = gr.JSON(label="Softmax Probabilities")
    run_button = gr.Button("Predict")
    gr.Examples(examples=[[example]], inputs=feature_input)

    run_button.click(fn=predict_from_features, inputs=feature_input, outputs=[status_box, prob_box])


if __name__ == "__main__":
    app.launch()
