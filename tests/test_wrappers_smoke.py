from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from src.models.padim_wrapper import PaDiMWrapper
from src.models.patchcore_wrapper import PatchCoreWrapper


class RandomImageDataset(Dataset):
    def __init__(self, num_samples: int = 4, image_size: int = 64) -> None:
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        return {"image": torch.rand(3, self.image_size, self.image_size)}


def test_patchcore_wrapper_fit_and_predict_smoke() -> None:
    cfg = {
        "seed": 42,
        "model": {
            "name": "patchcore",
            "backbone": "resnet18",
            "feature_layers": ["layer1", "layer2"],
            "pretrained": False,
            "patch_stride": 1,
            "coreset_sampling_ratio": 0.5,
            "num_neighbors": 3,
            "gaussian_sigma": 0.0,
            "projection_dim": 16,
            "distance_chunk_size": 256,
        },
    }

    model = PatchCoreWrapper.from_config(cfg).to(torch.device("cpu"))
    loader = DataLoader(RandomImageDataset(), batch_size=2)
    model.fit(loader)

    batch = next(iter(loader))
    prediction = model.predict(batch["image"])
    assert prediction.image_scores.shape == (2,)
    assert prediction.anomaly_maps.shape == (2, 1, 64, 64)


def test_padim_wrapper_fit_and_predict_smoke() -> None:
    cfg = {
        "seed": 42,
        "model": {
            "name": "padim",
            "backbone": "resnet18",
            "feature_layers": ["layer1", "layer2"],
            "pretrained": False,
            "embedding_dim": 192,
            "reduced_dim": 16,
            "gaussian_sigma": 0.0,
            "covariance_epsilon": 0.01,
        },
    }

    model = PaDiMWrapper.from_config(cfg).to(torch.device("cpu"))
    loader = DataLoader(RandomImageDataset(), batch_size=2)
    model.fit(loader)

    batch = next(iter(loader))
    prediction = model.predict(batch["image"])
    assert prediction.image_scores.shape == (2,)
    assert prediction.anomaly_maps.shape == (2, 1, 64, 64)
