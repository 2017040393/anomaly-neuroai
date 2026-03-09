from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image

from src.data.anomaly_transforms import denormalize_image_tensor


def normalize_anomaly_map(anomaly_map: torch.Tensor | np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if isinstance(anomaly_map, torch.Tensor):
        array = anomaly_map.detach().cpu().float().numpy()
    else:
        array = np.asarray(anomaly_map, dtype=np.float32)

    array = np.squeeze(array).astype(np.float32)
    array = array - float(array.min())
    scale = float(array.max())
    if scale > eps:
        array = array / scale
    return array.clip(0.0, 1.0)


def tensor_to_uint8_image(image_tensor: torch.Tensor, normalization: str = "imagenet") -> np.ndarray:
    image = denormalize_image_tensor(image_tensor.detach().cpu(), normalization=normalization)
    image = image.permute(1, 2, 0).numpy()
    return np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)


def anomaly_map_to_heatmap(anomaly_map: torch.Tensor | np.ndarray, cmap: str = "jet") -> np.ndarray:
    normalized = normalize_anomaly_map(anomaly_map)
    heatmap = colormaps.get_cmap(cmap)(normalized)[..., :3]
    return np.clip(heatmap * 255.0, 0.0, 255.0).astype(np.uint8)


def blend_overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    image_float = image.astype(np.float32)
    heatmap_float = heatmap.astype(np.float32)
    blended = (1.0 - alpha) * image_float + alpha * heatmap_float
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def save_single_anomaly_map(
    image_tensor: torch.Tensor,
    anomaly_map: torch.Tensor | np.ndarray,
    output_dir: str | Path,
    image_name: str,
    normalization: str = "imagenet",
    image_score: float | None = None,
    mask_tensor: torch.Tensor | None = None,
    save_overlay: bool = True,
    save_maps: bool = True,
    overlay_alpha: float = 0.4,
    cmap: str = "jet",
) -> dict[str, str]:
    output_path = Path(output_dir)
    image_dir = output_path / "images"
    map_dir = output_path / "maps"
    overlay_dir = output_path / "overlays"
    mask_dir = output_path / "masks"

    image_dir.mkdir(parents=True, exist_ok=True)
    if save_maps:
        map_dir.mkdir(parents=True, exist_ok=True)
    if save_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    if mask_tensor is not None:
        mask_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(image_name).stem
    image_rgb = tensor_to_uint8_image(image_tensor, normalization=normalization)
    heatmap_rgb = anomaly_map_to_heatmap(anomaly_map, cmap=cmap)

    image_path = image_dir / f"{stem}_image.png"
    Image.fromarray(image_rgb).save(image_path)

    map_path = ""
    if save_maps:
        raw_map = (normalize_anomaly_map(anomaly_map) * 255.0).astype(np.uint8)
        map_path = str(map_dir / f"{stem}_anomaly_map.png")
        Image.fromarray(raw_map, mode="L").save(map_path)

    overlay_path = ""
    if save_overlay:
        overlay_rgb = blend_overlay(image_rgb, heatmap_rgb, alpha=overlay_alpha)
        overlay_path = str(overlay_dir / f"{stem}_overlay.png")
        Image.fromarray(overlay_rgb).save(overlay_path)

    mask_path = ""
    if mask_tensor is not None:
        mask_array = normalize_anomaly_map(mask_tensor) * 255.0
        mask_path = str(mask_dir / f"{stem}_mask.png")
        Image.fromarray(mask_array.astype(np.uint8), mode="L").save(mask_path)

    metadata = {
        "saved_image_path": str(image_path),
        "saved_map_path": map_path,
        "saved_overlay_path": overlay_path,
        "saved_mask_path": mask_path,
    }
    if image_score is not None:
        metadata["saved_image_score"] = f"{float(image_score):.6f}"
    return metadata


def save_batch_anomaly_maps(
    batch: dict[str, Any],
    anomaly_maps: torch.Tensor,
    output_dir: str | Path,
    normalization: str = "imagenet",
    image_scores: list[float] | None = None,
    save_overlay: bool = True,
    save_maps: bool = True,
    overlay_alpha: float = 0.4,
    cmap: str = "jet",
) -> list[dict[str, str]]:
    images = batch["image"].detach().cpu()
    masks = batch.get("mask")
    masks_cpu = masks.detach().cpu() if isinstance(masks, torch.Tensor) else None
    image_names = list(batch["image_name"])

    rows: list[dict[str, str]] = []
    for index, image_name in enumerate(image_names):
        score = None if image_scores is None else image_scores[index]
        mask_tensor = None if masks_cpu is None else masks_cpu[index]
        rows.append(
            save_single_anomaly_map(
                image_tensor=images[index],
                anomaly_map=anomaly_maps[index],
                output_dir=output_dir,
                image_name=image_name,
                normalization=normalization,
                image_score=score,
                mask_tensor=mask_tensor,
                save_overlay=save_overlay,
                save_maps=save_maps,
                overlay_alpha=overlay_alpha,
                cmap=cmap,
            )
        )
    return rows
