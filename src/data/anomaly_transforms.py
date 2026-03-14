from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_normalization_stats(normalization: str | None) -> tuple[tuple[float, ...], tuple[float, ...]]:
    name = (normalization or "none").lower()
    if name == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    if name in {"none", "identity"}:
        return (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    raise ValueError(f"Unsupported normalization '{normalization}'. Expected one of ['imagenet', 'none'].")


def build_image_transform(
    image_size: int,
    center_crop: int | None = None,
    normalization: str | None = "imagenet",
    train: bool = False,
) -> Callable:
    if image_size <= 0:
        raise ValueError("image_size must be a positive integer.")
    if center_crop is not None and center_crop <= 0:
        raise ValueError("center_crop must be a positive integer when provided.")

    mean, std = get_normalization_stats(normalization)
    steps: list[Callable] = [
        transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
    ]
    if center_crop is not None:
        steps.append(transforms.CenterCrop(center_crop))
    if train:
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transforms.Compose(steps)


def build_mask_transform(image_size: int, center_crop: int | None = None) -> Callable:
    if image_size <= 0:
        raise ValueError("image_size must be a positive integer.")
    if center_crop is not None and center_crop <= 0:
        raise ValueError("center_crop must be a positive integer when provided.")

    steps: list[Callable] = [
        transforms.Resize(image_size, interpolation=InterpolationMode.NEAREST),
    ]
    if center_crop is not None:
        steps.append(transforms.CenterCrop(center_crop))
    steps.append(transforms.PILToTensor())
    return transforms.Compose(steps)


def build_mvtec_transforms(cfg: dict[str, Any]) -> tuple[Callable, Callable, Callable]:
    data_cfg = cfg["data"]
    image_size = int(data_cfg.get("image_size", 256))
    center_crop = data_cfg.get("center_crop")
    center_crop = int(center_crop) if center_crop is not None else None
    normalization = str(data_cfg.get("normalization", "imagenet"))

    train_transform = build_image_transform(
        image_size=image_size,
        center_crop=center_crop,
        normalization=normalization,
        train=False,
    )
    eval_transform = build_image_transform(
        image_size=image_size,
        center_crop=center_crop,
        normalization=normalization,
        train=False,
    )
    mask_transform = build_mask_transform(image_size=image_size, center_crop=center_crop)
    return train_transform, eval_transform, mask_transform


def denormalize_image_tensor(
    image_tensor: torch.Tensor,
    normalization: str | None = "imagenet",
) -> torch.Tensor:
    mean, std = get_normalization_stats(normalization)
    mean_tensor = torch.tensor(mean, dtype=image_tensor.dtype, device=image_tensor.device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=image_tensor.dtype, device=image_tensor.device).view(-1, 1, 1)

    if image_tensor.ndim == 4:
        mean_tensor = mean_tensor.unsqueeze(0)
        std_tensor = std_tensor.unsqueeze(0)
    elif image_tensor.ndim != 3:
        raise ValueError(f"image_tensor must have shape [C,H,W] or [N,C,H,W], got {tuple(image_tensor.shape)}.")

    denormalized = image_tensor * std_tensor + mean_tensor
    return denormalized.clamp(0.0, 1.0)
