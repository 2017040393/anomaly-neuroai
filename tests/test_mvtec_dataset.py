from __future__ import annotations

import numpy as np
from PIL import Image

from src.data.mvtec import MVTecADDataset, build_mvtec_datasets


def _save_rgb(path, color: int) -> None:
    image = np.full((32, 32, 3), fill_value=color, dtype=np.uint8)
    Image.fromarray(image).save(path)


def _save_mask(path) -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    Image.fromarray(mask).save(path)


def _build_fake_mvtec_tree(root) -> None:
    category_root = root / "bottle"
    (category_root / "train" / "good").mkdir(parents=True)
    (category_root / "test" / "good").mkdir(parents=True)
    (category_root / "test" / "broken_large").mkdir(parents=True)
    (category_root / "ground_truth" / "broken_large").mkdir(parents=True)

    _save_rgb(category_root / "train" / "good" / "000.png", color=96)
    _save_rgb(category_root / "test" / "good" / "001.png", color=128)
    _save_rgb(category_root / "test" / "broken_large" / "002.png", color=192)
    _save_mask(category_root / "ground_truth" / "broken_large" / "002_mask.png")


def test_mvtec_dataset_reads_labels_and_masks(tmp_path) -> None:
    _build_fake_mvtec_tree(tmp_path)

    train_dataset = MVTecADDataset(root_dir=tmp_path, category="bottle", split="train")
    test_dataset = MVTecADDataset(root_dir=tmp_path, category="bottle", split="test")

    assert len(train_dataset) == 1
    assert len(test_dataset) == 2

    train_sample = train_dataset[0]
    assert tuple(train_sample["image"].shape) == (3, 32, 32)
    assert int(train_sample["label"].item()) == 0
    assert float(train_sample["mask"].sum().item()) == 0.0

    anomaly_sample = next(
        sample
        for sample in (test_dataset[index] for index in range(len(test_dataset)))
        if sample["defect_type"] != "good"
    )
    assert int(anomaly_sample["label"].item()) == 1
    assert float(anomaly_sample["mask"].sum().item()) > 0.0


def test_build_mvtec_datasets_respects_limits(tmp_path) -> None:
    _build_fake_mvtec_tree(tmp_path)
    cfg = {
        "seed": 42,
        "device": "cpu",
        "data": {
            "root_dir": str(tmp_path),
            "category": "bottle",
            "train_split": "train",
            "test_split": "test",
            "mask_dir_name": "ground_truth",
            "limit_train_samples": 1,
            "limit_test_samples": 1,
        },
        "train": {"batch_size": 1, "num_workers": 0},
        "inference": {"batch_size": 1},
    }

    train_dataset, test_dataset = build_mvtec_datasets(cfg)
    assert len(train_dataset) == 1
    assert len(test_dataset) == 1
