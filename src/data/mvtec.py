from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MVTEC_AD_CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)


@dataclass(frozen=True)
class MVTecSample:
    image_path: Path
    label: int
    defect_type: str
    split: str
    mask_path: Path | None


def _pil_to_float_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = array[..., None]
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor / 255.0


def _ensure_image_tensor(image: torch.Tensor | np.ndarray | Image.Image) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = image.detach().clone()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return tensor.float()
    if isinstance(image, np.ndarray):
        tensor = torch.from_numpy(image)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[0] not in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        return tensor.float()
    return _pil_to_float_tensor(image)


def _ensure_mask_tensor(mask: torch.Tensor | np.ndarray | Image.Image) -> torch.Tensor:
    if isinstance(mask, torch.Tensor):
        tensor = mask.detach().clone().float()
    elif isinstance(mask, np.ndarray):
        tensor = torch.from_numpy(mask).float()
    else:
        tensor = _pil_to_float_tensor(mask)

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3 and tensor.shape[0] != 1:
        tensor = tensor[:1]

    if tensor.max().item() > 1.0:
        tensor = tensor / 255.0
    return (tensor > 0.5).float()


def _resize_mask_to_match(mask: Image.Image, spatial_size: tuple[int, int]) -> Image.Image:
    height, width = spatial_size
    if mask.size != (width, height):
        mask = mask.resize((width, height), resample=Image.NEAREST)
    return mask


def _list_available_categories(root_dir: Path) -> list[str]:
    if not root_dir.exists():
        return []
    return sorted(path.name for path in root_dir.iterdir() if path.is_dir())


def _limit_samples(samples: list[MVTecSample], limit: int | None, seed: int) -> list[MVTecSample]:
    if limit is None or limit <= 0 or limit >= len(samples):
        return samples

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(samples), size=limit, replace=False))
    return [samples[index] for index in indices.tolist()]


class MVTecADDataset(Dataset[dict[str, Any]]):
    """Dataset reader for the standard MVTec AD directory layout."""

    def __init__(
        self,
        root_dir: str | Path,
        category: str,
        split: str,
        transform: Callable[[Image.Image], Any] | None = None,
        mask_transform: Callable[[Image.Image], Any] | None = None,
        mask_dir_name: str = "ground_truth",
        limit_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.category = category
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform
        self.mask_dir_name = mask_dir_name

        if split not in {"train", "test"}:
            raise ValueError(f"Unsupported split '{split}'. Expected 'train' or 'test'.")
        if not self.root_dir.exists():
            raise FileNotFoundError(f"MVTec root directory not found: {self.root_dir}")

        self.category_dir = self.root_dir / self.category
        if not self.category_dir.exists():
            available = ", ".join(_list_available_categories(self.root_dir)) or "none"
            raise FileNotFoundError(
                f"MVTec category directory not found: {self.category_dir}. "
                f"Available categories: {available}"
            )

        split_dir = self.category_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.samples = self._discover_samples()
        self.samples = _limit_samples(self.samples, limit_samples, seed)

        if not self.samples:
            raise RuntimeError(
                f"No image files found under {split_dir}. "
                "Expected the standard MVTec AD directory layout."
            )

    def _discover_samples(self) -> list[MVTecSample]:
        split_dir = self.category_dir / self.split
        samples: list[MVTecSample] = []

        for defect_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            defect_type = defect_dir.name
            image_paths = sorted(
                path for path in defect_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )

            for image_path in image_paths:
                is_anomaly = self.split == "test" and defect_type != "good"
                mask_path = self._resolve_mask_path(defect_type, image_path) if is_anomaly else None
                samples.append(
                    MVTecSample(
                        image_path=image_path,
                        label=int(is_anomaly),
                        defect_type=defect_type,
                        split=self.split,
                        mask_path=mask_path,
                    )
                )
        return samples

    def _resolve_mask_path(self, defect_type: str, image_path: Path) -> Path:
        mask_dir = self.category_dir / self.mask_dir_name / defect_type
        if not mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found for defect '{defect_type}': {mask_dir}")

        candidate = mask_dir / f"{image_path.stem}_mask.png"
        if candidate.exists():
            return candidate

        matches = sorted(mask_dir.glob(f"{image_path.stem}_mask.*"))
        if matches:
            return matches[0]

        raise FileNotFoundError(
            f"Mask file not found for image {image_path.name} in directory {mask_dir}."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]

        image = Image.open(sample.image_path).convert("RGB")
        if self.transform is not None:
            image_tensor = _ensure_image_tensor(self.transform(image))
        else:
            image_tensor = _pil_to_float_tensor(image)

        spatial_size = tuple(int(dim) for dim in image_tensor.shape[-2:])
        if sample.mask_path is not None:
            mask_image = Image.open(sample.mask_path).convert("L")
            if self.mask_transform is not None:
                mask_tensor = _ensure_mask_tensor(self.mask_transform(mask_image))
            else:
                mask_tensor = _ensure_mask_tensor(_resize_mask_to_match(mask_image, spatial_size))
        else:
            empty_mask = Image.new("L", (spatial_size[1], spatial_size[0]), color=0)
            if self.mask_transform is not None:
                mask_tensor = _ensure_mask_tensor(self.mask_transform(empty_mask))
            else:
                mask_tensor = _ensure_mask_tensor(empty_mask)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": torch.tensor(sample.label, dtype=torch.long),
            "is_anomaly": torch.tensor(sample.label, dtype=torch.long),
            "category": self.category,
            "defect_type": sample.defect_type,
            "image_path": str(sample.image_path),
            "mask_path": str(sample.mask_path) if sample.mask_path is not None else "",
            "image_name": sample.image_path.name,
            "split": sample.split,
        }


def build_mvtec_datasets(
    cfg: dict[str, Any],
    train_transform: Callable[[Image.Image], Any] | None = None,
    eval_transform: Callable[[Image.Image], Any] | None = None,
    mask_transform: Callable[[Image.Image], Any] | None = None,
) -> tuple[MVTecADDataset, MVTecADDataset]:
    data_cfg = cfg["data"]
    seed = int(cfg.get("seed", 42))

    train_dataset = MVTecADDataset(
        root_dir=data_cfg["root_dir"],
        category=data_cfg["category"],
        split=str(data_cfg.get("train_split", "train")),
        transform=train_transform,
        mask_transform=mask_transform,
        mask_dir_name=str(data_cfg.get("mask_dir_name", "ground_truth")),
        limit_samples=data_cfg.get("limit_train_samples"),
        seed=seed,
    )
    test_dataset = MVTecADDataset(
        root_dir=data_cfg["root_dir"],
        category=data_cfg["category"],
        split=str(data_cfg.get("test_split", "test")),
        transform=eval_transform if eval_transform is not None else train_transform,
        mask_transform=mask_transform,
        mask_dir_name=str(data_cfg.get("mask_dir_name", "ground_truth")),
        limit_samples=data_cfg.get("limit_test_samples"),
        seed=seed + 1,
    )
    return train_dataset, test_dataset


def build_mvtec_dataloaders(
    cfg: dict[str, Any],
    train_transform: Callable[[Image.Image], Any] | None = None,
    eval_transform: Callable[[Image.Image], Any] | None = None,
    mask_transform: Callable[[Image.Image], Any] | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_dataset, test_dataset = build_mvtec_datasets(
        cfg=cfg,
        train_transform=train_transform,
        eval_transform=eval_transform,
        mask_transform=mask_transform,
    )

    train_cfg = cfg["train"]
    inference_cfg = cfg.get("inference", {})
    pin_memory = str(cfg.get("device", "cpu")).lower() == "cuda" and torch.cuda.is_available()
    num_workers = int(train_cfg.get("num_workers", 0))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(inference_cfg.get("batch_size", train_cfg["batch_size"])),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader
