from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.random_projection import SparseRandomProjection
from torch import nn
from torchvision import models as tv_models
from torchvision.models.feature_extraction import create_feature_extractor


@dataclass
class PatchCorePrediction:
    image_scores: torch.Tensor
    anomaly_maps: torch.Tensor
    patch_scores: torch.Tensor


def _build_torchvision_backbone(model_name: str, pretrained: bool) -> nn.Module:
    if not hasattr(tv_models, model_name):
        raise ValueError(f"Unsupported torchvision backbone '{model_name}'.")

    builder = getattr(tv_models, model_name)
    if pretrained:
        try:
            weights_enum = tv_models.get_model_weights(model_name)
            return builder(weights=weights_enum.DEFAULT)
        except Exception:
            try:
                return builder(weights="DEFAULT")
            except Exception:
                return builder(pretrained=True)

    try:
        return builder(weights=None)
    except Exception:
        return builder(pretrained=False)


def _extract_images(batch: Any) -> torch.Tensor:
    if isinstance(batch, dict):
        if "image" not in batch:
            raise KeyError("Batch dictionary must contain an 'image' key.")
        images = batch["image"]
    elif isinstance(batch, (list, tuple)):
        if not batch:
            raise ValueError("Received an empty batch.")
        images = batch[0]
    else:
        images = batch

    if not isinstance(images, torch.Tensor):
        raise TypeError(f"Expected images to be a torch.Tensor, got {type(images)!r}.")
    return images


class PatchCoreWrapper(nn.Module):
    """Minimal PatchCore-style wrapper built on top of torchvision backbones."""

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        feature_layers: list[str] | tuple[str, ...] = ("layer2", "layer3"),
        pretrained: bool = True,
        patch_stride: int = 1,
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
        gaussian_sigma: float = 4.0,
        projection_dim: int = 128,
        distance_chunk_size: int = 2048,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if patch_stride <= 0:
            raise ValueError("patch_stride must be a positive integer.")
        if not 0.0 < coreset_sampling_ratio <= 1.0:
            raise ValueError("coreset_sampling_ratio must be in the range (0, 1].")
        if num_neighbors <= 0:
            raise ValueError("num_neighbors must be a positive integer.")
        if gaussian_sigma < 0:
            raise ValueError("gaussian_sigma must be non-negative.")
        if projection_dim <= 0:
            raise ValueError("projection_dim must be a positive integer.")
        if distance_chunk_size <= 0:
            raise ValueError("distance_chunk_size must be a positive integer.")

        self.backbone_name = backbone
        self.feature_layers = list(feature_layers)
        self.pretrained = pretrained
        self.patch_stride = patch_stride
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.num_neighbors = num_neighbors
        self.gaussian_sigma = gaussian_sigma
        self.projection_dim = projection_dim
        self.distance_chunk_size = distance_chunk_size
        self.seed = seed

        backbone_model = _build_torchvision_backbone(model_name=backbone, pretrained=pretrained)
        for parameter in backbone_model.parameters():
            parameter.requires_grad = False

        return_nodes = {layer_name: layer_name for layer_name in self.feature_layers}
        self.feature_extractor = create_feature_extractor(backbone_model, return_nodes=return_nodes)
        self.feature_extractor.eval()

        self.memory_bank: torch.Tensor | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "PatchCoreWrapper":
        model_cfg = cfg["model"]
        return cls(
            backbone=str(model_cfg.get("backbone", "wide_resnet50_2")),
            feature_layers=list(model_cfg.get("feature_layers", ["layer2", "layer3"])),
            pretrained=bool(model_cfg.get("pretrained", True)),
            patch_stride=int(model_cfg.get("patch_stride", 1)),
            coreset_sampling_ratio=float(model_cfg.get("coreset_sampling_ratio", 0.1)),
            num_neighbors=int(model_cfg.get("num_neighbors", 9)),
            gaussian_sigma=float(model_cfg.get("gaussian_sigma", 4.0)),
            projection_dim=int(model_cfg.get("projection_dim", 128)),
            distance_chunk_size=int(model_cfg.get("distance_chunk_size", 2048)),
            seed=int(cfg.get("seed", 42)),
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "name": "patchcore",
            "backbone": self.backbone_name,
            "feature_layers": self.feature_layers,
            "pretrained": self.pretrained,
            "patch_stride": self.patch_stride,
            "coreset_sampling_ratio": self.coreset_sampling_ratio,
            "num_neighbors": self.num_neighbors,
            "gaussian_sigma": self.gaussian_sigma,
            "projection_dim": self.projection_dim,
            "distance_chunk_size": self.distance_chunk_size,
            "seed": self.seed,
        }

    def _model_device(self) -> torch.device:
        return next(self.feature_extractor.parameters()).device

    def _build_embedding_map(self, images: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            feature_maps = self.feature_extractor(images)

        base_feature = feature_maps[self.feature_layers[0]]
        if base_feature.ndim != 4:
            raise RuntimeError("Backbone feature maps must have shape [B, C, H, W].")

        resized_maps: list[torch.Tensor] = []
        target_size = base_feature.shape[-2:]
        for layer_name in self.feature_layers:
            feature_map = feature_maps[layer_name]
            if feature_map.shape[-2:] != target_size:
                feature_map = F.interpolate(
                    feature_map,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            resized_maps.append(feature_map)

        embedding_map = torch.cat(resized_maps, dim=1)
        if self.patch_stride > 1:
            embedding_map = F.avg_pool2d(
                embedding_map,
                kernel_size=self.patch_stride,
                stride=self.patch_stride,
            )
        return F.normalize(embedding_map, p=2, dim=1)

    def _flatten_embedding_map(self, embedding_map: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = embedding_map.shape
        return embedding_map.permute(0, 2, 3, 1).reshape(batch_size * height * width, channels)

    def _project_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings_cpu = embeddings.detach().cpu()
        if embeddings_cpu.ndim != 2:
            raise ValueError(
                f"embeddings must have shape [N, C], got {tuple(embeddings_cpu.shape)}."
            )

        num_samples, num_features = embeddings_cpu.shape
        if num_samples <= 1 or num_features <= self.projection_dim:
            return embeddings_cpu

        projector = SparseRandomProjection(
            n_components=min(self.projection_dim, num_features),
            dense_output=True,
            random_state=self.seed,
        )
        projected = projector.fit_transform(embeddings_cpu.numpy())
        return torch.from_numpy(np.asarray(projected, dtype=np.float32))

    def _greedy_coreset_indices(self, projected_embeddings: torch.Tensor, target_size: int) -> torch.Tensor:
        num_embeddings = projected_embeddings.shape[0]
        if target_size >= num_embeddings:
            return torch.arange(num_embeddings, dtype=torch.long)

        generator = torch.Generator()
        generator.manual_seed(self.seed)
        selected_indices = torch.empty(target_size, dtype=torch.long)
        first_index = int(torch.randint(num_embeddings, (1,), generator=generator).item())
        selected_indices[0] = first_index

        min_distances = torch.cdist(
            projected_embeddings,
            projected_embeddings[first_index : first_index + 1],
        ).squeeze(1)

        for index in range(1, target_size):
            next_index = int(torch.argmax(min_distances).item())
            selected_indices[index] = next_index
            distances = torch.cdist(
                projected_embeddings,
                projected_embeddings[next_index : next_index + 1],
            ).squeeze(1)
            min_distances = torch.minimum(min_distances, distances)

        return selected_indices

    def _subsample_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must have shape [N, C], got {tuple(embeddings.shape)}.")

        num_embeddings = embeddings.shape[0]
        target_size = max(1, int(num_embeddings * self.coreset_sampling_ratio))
        if target_size >= num_embeddings:
            return embeddings

        projected = self._project_embeddings(embeddings)
        selected_indices = self._greedy_coreset_indices(projected, target_size)
        return embeddings[selected_indices]

    def _score_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.memory_bank is None or self.memory_bank.numel() == 0:
            raise RuntimeError("PatchCore memory bank is empty. Call `fit(...)` before inference.")

        memory_bank = self.memory_bank.to(embeddings.device)
        nearest_k = min(self.num_neighbors, memory_bank.shape[0])
        score_chunks: list[torch.Tensor] = []

        for start in range(0, embeddings.shape[0], self.distance_chunk_size):
            stop = start + self.distance_chunk_size
            chunk = embeddings[start:stop]
            distances = torch.cdist(chunk, memory_bank)
            nearest_distances = torch.topk(
                distances,
                k=nearest_k,
                largest=False,
                dim=1,
            ).values
            score_chunks.append(nearest_distances.mean(dim=1))

        return torch.cat(score_chunks, dim=0)

    def _apply_gaussian_smoothing(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        if self.gaussian_sigma <= 0:
            return anomaly_map

        radius = max(1, int(round(self.gaussian_sigma * 3)))
        kernel_size = radius * 2 + 1
        coords = torch.arange(kernel_size, dtype=anomaly_map.dtype, device=anomaly_map.device)
        coords = coords - radius
        kernel_1d = torch.exp(-(coords**2) / (2 * self.gaussian_sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, kernel_size, kernel_size)

        padded = F.pad(anomaly_map, (radius, radius, radius, radius), mode="reflect")
        return F.conv2d(padded, kernel_2d)

    @torch.inference_mode()
    def extract_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        embedding_map = self._build_embedding_map(images)
        return self._flatten_embedding_map(embedding_map)

    @torch.inference_mode()
    def fit(self, train_loader) -> None:
        self.eval()
        device = self._model_device()
        embedding_chunks: list[torch.Tensor] = []

        for batch in train_loader:
            images = _extract_images(batch).to(device)
            embeddings = self.extract_embeddings(images)
            embedding_chunks.append(embeddings.cpu())

        if not embedding_chunks:
            raise RuntimeError("Received an empty train_loader while building the PatchCore memory bank.")

        full_embedding_bank = torch.cat(embedding_chunks, dim=0)
        sampled_bank = self._subsample_embeddings(full_embedding_bank)
        self.memory_bank = sampled_bank.to(device)

    @torch.inference_mode()
    def predict(self, images: torch.Tensor) -> PatchCorePrediction:
        self.eval()
        embedding_map = self._build_embedding_map(images)
        patch_embeddings = self._flatten_embedding_map(embedding_map)
        patch_scores = self._score_embeddings(patch_embeddings)

        batch_size = images.shape[0]
        height, width = embedding_map.shape[-2:]
        patch_scores = patch_scores.view(batch_size, 1, height, width)

        anomaly_maps = F.interpolate(
            patch_scores,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        anomaly_maps = self._apply_gaussian_smoothing(anomaly_maps)

        flattened_maps = anomaly_maps.flatten(start_dim=1)
        image_score_k = min(self.num_neighbors, flattened_maps.shape[1])
        image_scores = torch.topk(flattened_maps, k=image_score_k, dim=1).values.mean(dim=1)

        return PatchCorePrediction(
            image_scores=image_scores,
            anomaly_maps=anomaly_maps,
            patch_scores=patch_scores,
        )

    def forward(self, images: torch.Tensor) -> PatchCorePrediction:
        return self.predict(images)

    def export_state(self) -> dict[str, Any]:
        return {
            "model_name": "patchcore",
            "model_config": self.get_config(),
            "feature_extractor_state_dict": self.feature_extractor.state_dict(),
            "memory_bank": None if self.memory_bank is None else self.memory_bank.detach().cpu(),
        }

    def load_exported_state(self, state: dict[str, Any], strict: bool = True) -> None:
        feature_state = state.get("feature_extractor_state_dict")
        if feature_state is None:
            raise KeyError("Checkpoint state is missing 'feature_extractor_state_dict'.")
        self.feature_extractor.load_state_dict(feature_state, strict=strict)

        memory_bank = state.get("memory_bank")
        if memory_bank is not None:
            self.memory_bank = memory_bank.to(self._model_device())
        else:
            self.memory_bank = None

    def save(self, path: str | Path, extra_state: dict[str, Any] | None = None) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        state = self.export_state()
        if extra_state:
            state.update(extra_state)
        torch.save(state, checkpoint_path)

    @classmethod
    def _from_saved_model_config(cls, model_config: dict[str, Any]) -> "PatchCoreWrapper":
        return cls(
            backbone=str(model_config.get("backbone", "wide_resnet50_2")),
            feature_layers=list(model_config.get("feature_layers", ["layer2", "layer3"])),
            pretrained=bool(model_config.get("pretrained", True)),
            patch_stride=int(model_config.get("patch_stride", 1)),
            coreset_sampling_ratio=float(model_config.get("coreset_sampling_ratio", 0.1)),
            num_neighbors=int(model_config.get("num_neighbors", 9)),
            gaussian_sigma=float(model_config.get("gaussian_sigma", 4.0)),
            projection_dim=int(model_config.get("projection_dim", 128)),
            distance_chunk_size=int(model_config.get("distance_chunk_size", 2048)),
            seed=int(model_config.get("seed", 42)),
        )

    @classmethod
    def load_from_checkpoint(
        cls,
        path: str | Path,
        cfg: dict[str, Any] | None = None,
        device: torch.device | None = None,
    ) -> "PatchCoreWrapper":
        checkpoint = torch.load(path, map_location=device or "cpu")
        if cfg is not None:
            model = cls.from_config(cfg)
        else:
            model = cls._from_saved_model_config(checkpoint["model_config"])
        if device is not None:
            model = model.to(device)
        model.load_exported_state(checkpoint)
        model.eval()
        return model


def build_patchcore_wrapper(cfg: dict[str, Any]) -> PatchCoreWrapper:
    return PatchCoreWrapper.from_config(cfg)
