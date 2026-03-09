from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.datasets import SyntheticClassificationDataset


def build_dataloaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    """Build train/validation dataloaders from a config dictionary."""

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    seed = int(cfg.get("seed", 42))

    num_features = int(data_cfg["num_features"])
    num_classes = int(data_cfg["num_classes"])
    train_size = int(data_cfg["train_size"])
    val_size = int(data_cfg["val_size"])

    class_separation = float(data_cfg.get("class_separation", 3.0))
    noise_std = float(data_cfg.get("noise_std", 0.8))

    center_rng = np.random.default_rng(seed)
    class_centers = center_rng.normal(
        loc=0.0,
        scale=class_separation,
        size=(num_classes, num_features),
    ).astype(np.float32)

    train_dataset = SyntheticClassificationDataset(
        num_samples=train_size,
        num_features=num_features,
        num_classes=num_classes,
        rng_seed=seed,
        class_centers=class_centers,
        class_separation=class_separation,
        noise_std=noise_std,
    )
    val_dataset = SyntheticClassificationDataset(
        num_samples=val_size,
        num_features=num_features,
        num_classes=num_classes,
        rng_seed=seed + 1,
        class_centers=class_centers,
        class_separation=class_separation,
        noise_std=noise_std,
    )

    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = str(cfg.get("device", "cpu")).lower() == "cuda" and torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
