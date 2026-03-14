from __future__ import annotations

from typing import Any

from src.models.mlp import MLPClassifier


def build_model(cfg: dict[str, Any]):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    input_dim = int(data_cfg["num_features"])
    num_classes = int(data_cfg["num_classes"])
    hidden_dim = int(model_cfg.get("hidden_dim", 64))

    model_name = str(model_cfg["name"]).lower()
    if model_name == "mlp":
        return MLPClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    if model_name == "snn_mlp":
        from src.models.snn import SNNClassifier

        time_steps = int(model_cfg.get("time_steps", 4))
        return SNNClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            time_steps=time_steps,
        )

    raise ValueError(f"Unsupported model '{model_name}'. Expected one of ['mlp', 'snn_mlp'].")
