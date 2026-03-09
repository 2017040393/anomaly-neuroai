from __future__ import annotations

import torch

from src.data.datasets import SyntheticClassificationDataset
from src.models.build_model import build_model


def test_build_model_creates_mlp() -> None:
    cfg = {
        "data": {"num_features": 8, "num_classes": 3},
        "model": {"name": "mlp", "hidden_dim": 16},
    }

    model = build_model(cfg)
    inputs = torch.randn(4, 8)
    outputs = model(inputs)

    assert outputs.shape == (4, 3)


def test_synthetic_dataset_returns_expected_shapes() -> None:
    dataset = SyntheticClassificationDataset(
        num_samples=10,
        num_features=6,
        num_classes=2,
        rng_seed=123,
    )

    features, label = dataset[0]
    assert features.shape == (6,)
    assert features.dtype == torch.float32
    assert label.dtype == torch.int64
