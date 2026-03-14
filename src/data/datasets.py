from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticClassificationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Synthetic feature dataset used to validate the training pipeline."""

    def __init__(
        self,
        num_samples: int,
        num_features: int,
        num_classes: int,
        rng_seed: int = 42,
        class_centers: Optional[np.ndarray] = None,
        class_separation: float = 3.0,
        noise_std: float = 0.8,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")
        if num_features <= 0:
            raise ValueError("num_features must be a positive integer.")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")

        rng = np.random.default_rng(rng_seed)

        if class_centers is None:
            class_centers = rng.normal(
                loc=0.0,
                scale=class_separation,
                size=(num_classes, num_features),
            ).astype(np.float32)
        else:
            class_centers = np.asarray(class_centers, dtype=np.float32)
            if class_centers.shape != (num_classes, num_features):
                raise ValueError(
                    f"class_centers must have shape ({num_classes}, {num_features}), got {class_centers.shape}."
                )

        labels = rng.integers(0, num_classes, size=num_samples, dtype=np.int64)
        noise = rng.normal(loc=0.0, scale=noise_std, size=(num_samples, num_features)).astype(np.float32)
        features = class_centers[labels] + noise

        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]
