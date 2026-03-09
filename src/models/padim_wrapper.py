from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor

from src.models.patchcore_wrapper import _build_torchvision_backbone, _extract_images


@dataclass
class PaDiMPrediction:
    image_scores: torch.Tensor
    anomaly_maps: torch.Tensor
    patch_scores: torch.Tensor


class PaDiMWrapper(nn.Module):
    """Minimal PaDiM-style wrapper for industrial anomaly detection."""

    def __init__(
        self,
        backbone: str = "resnet18",
        feature_layers: list[str] | tuple[str, ...] = ("layer1", "layer2", "layer3"),
        pretrained: bool = True,
        embedding_dim: int | None = None,
        reduced_dim: int = 100,
        gaussian_sigma: float = 4.0,
        covariance_epsilon: float = 0.01,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if reduced_dim <= 0:
            raise ValueError("reduced_dim must be a positive integer.")
        if gaussian_sigma < 0:
            raise ValueError("gaussian_sigma must be non-negative.")
        if covariance_epsilon <= 0:
            raise ValueError("covariance_epsilon must be positive.")

        self.backbone_name = backbone
        self.feature_layers = list(feature_layers)
        self.pretrained = pretrained
        self.embedding_dim = embedding_dim
        self.reduced_dim = reduced_dim
        self.gaussian_sigma = gaussian_sigma
        self.covariance_epsilon = covariance_epsilon
        self.seed = seed

        backbone_model = _build_torchvision_backbone(model_name=backbone, pretrained=pretrained)
        for parameter in backbone_model.parameters():
            parameter.requires_grad = False

        return_nodes = {layer_name: layer_name for layer_name in self.feature_layers}
        self.feature_extractor = create_feature_extractor(backbone_model, return_nodes=return_nodes)
        self.feature_extractor.eval()

        self.mean_embeddings: torch.Tensor | None = None
        self.inv_covariances: torch.Tensor | None = None
        self.selected_indices: torch.Tensor | None = None
        self.feature_map_size: tuple[int, int] | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "PaDiMWrapper":
        model_cfg = cfg["model"]
        return cls(
            backbone=str(model_cfg.get("backbone", "resnet18")),
            feature_layers=list(model_cfg.get("feature_layers", ["layer1", "layer2", "layer3"])),
            pretrained=bool(model_cfg.get("pretrained", True)),
            embedding_dim=model_cfg.get("embedding_dim"),
            reduced_dim=int(model_cfg.get("reduced_dim", 100)),
            gaussian_sigma=float(model_cfg.get("gaussian_sigma", 4.0)),
            covariance_epsilon=float(model_cfg.get("covariance_epsilon", 0.01)),
            seed=int(cfg.get("seed", 42)),
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "name": "padim",
            "backbone": self.backbone_name,
            "feature_layers": self.feature_layers,
            "pretrained": self.pretrained,
            "embedding_dim": self.embedding_dim,
            "reduced_dim": self.reduced_dim,
            "gaussian_sigma": self.gaussian_sigma,
            "covariance_epsilon": self.covariance_epsilon,
            "seed": self.seed,
        }

    def _model_device(self) -> torch.device:
        return next(self.feature_extractor.parameters()).device

    def _build_embedding_map(self, images: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            feature_maps = self.feature_extractor(images)

        reference = feature_maps[self.feature_layers[0]]
        if reference.ndim != 4:
            raise RuntimeError("Backbone feature maps must have shape [B, C, H, W].")

        target_size = reference.shape[-2:]
        resized_maps: list[torch.Tensor] = []
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

        return torch.cat(resized_maps, dim=1)

    def _ensure_selected_indices(self, total_channels: int) -> torch.Tensor:
        if self.selected_indices is None:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            target_dim = min(self.reduced_dim, total_channels)
            permutation = torch.randperm(total_channels, generator=generator)
            self.selected_indices = permutation[:target_dim].sort().values.cpu()
        return self.selected_indices

    def _select_reduced_channels(self, embedding_map: torch.Tensor) -> torch.Tensor:
        indices = self._ensure_selected_indices(embedding_map.shape[1]).to(embedding_map.device)
        return embedding_map[:, indices, :, :]

    def _flatten_embeddings(self, embedding_map: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = embedding_map.shape
        return embedding_map.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)

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
    def fit(self, train_loader) -> None:
        self.eval()
        device = self._model_device()
        embedding_batches: list[torch.Tensor] = []

        for batch in train_loader:
            images = _extract_images(batch).to(device)
            embedding_map = self._build_embedding_map(images)
            embedding_map = self._select_reduced_channels(embedding_map)
            flattened = self._flatten_embeddings(embedding_map)
            embedding_batches.append(flattened.cpu())

        if not embedding_batches:
            raise RuntimeError("Received an empty train_loader while fitting PaDiM statistics.")

        train_embeddings = torch.cat(embedding_batches, dim=0)
        num_samples, num_positions, channels = train_embeddings.shape
        height, width = embedding_map.shape[-2:]
        self.feature_map_size = (height, width)

        mean_embeddings = train_embeddings.mean(dim=0)
        eye = torch.eye(channels, dtype=train_embeddings.dtype)
        inv_covariances: list[torch.Tensor] = []

        denominator = max(num_samples - 1, 1)
        for position in range(num_positions):
            position_embeddings = train_embeddings[:, position, :]
            centered = position_embeddings - mean_embeddings[position]
            covariance = (centered.T @ centered) / denominator
            covariance = covariance + self.covariance_epsilon * eye
            try:
                inv_covariance = torch.linalg.inv(covariance)
            except RuntimeError:
                inv_covariance = torch.linalg.pinv(covariance)
            inv_covariances.append(inv_covariance)

        self.mean_embeddings = mean_embeddings.to(device)
        self.inv_covariances = torch.stack(inv_covariances, dim=0).to(device)

    @torch.inference_mode()
    def predict(self, images: torch.Tensor) -> PaDiMPrediction:
        if self.mean_embeddings is None or self.inv_covariances is None or self.feature_map_size is None:
            raise RuntimeError("PaDiM statistics are empty. Call `fit(...)` before inference.")

        self.eval()
        embedding_map = self._build_embedding_map(images)
        embedding_map = self._select_reduced_channels(embedding_map)
        embeddings = self._flatten_embeddings(embedding_map)

        height, width = self.feature_map_size
        if embeddings.shape[1] != height * width:
            raise RuntimeError(
                "Feature map size mismatch between training and inference. "
                f"Expected {self.feature_map_size}, got {embedding_map.shape[-2:]}."
            )

        mean_embeddings = self.mean_embeddings.to(images.device)
        inv_covariances = self.inv_covariances.to(images.device)
        diff = embeddings - mean_embeddings.unsqueeze(0)

        mahalanobis = torch.einsum("bhc,hcd,bhd->bh", diff, inv_covariances, diff)
        mahalanobis = mahalanobis.clamp_min(0.0).sqrt()

        patch_scores = mahalanobis.view(images.shape[0], 1, height, width)
        anomaly_maps = F.interpolate(
            patch_scores,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        anomaly_maps = self._apply_gaussian_smoothing(anomaly_maps)
        image_scores = anomaly_maps.flatten(start_dim=1).max(dim=1).values

        return PaDiMPrediction(
            image_scores=image_scores,
            anomaly_maps=anomaly_maps,
            patch_scores=patch_scores,
        )

    def forward(self, images: torch.Tensor) -> PaDiMPrediction:
        return self.predict(images)

    def export_state(self) -> dict[str, Any]:
        return {
            "model_name": "padim",
            "model_config": self.get_config(),
            "feature_extractor_state_dict": self.feature_extractor.state_dict(),
            "mean_embeddings": None if self.mean_embeddings is None else self.mean_embeddings.detach().cpu(),
            "inv_covariances": None if self.inv_covariances is None else self.inv_covariances.detach().cpu(),
            "selected_indices": None if self.selected_indices is None else self.selected_indices.detach().cpu(),
            "feature_map_size": self.feature_map_size,
        }

    def load_exported_state(self, state: dict[str, Any], strict: bool = True) -> None:
        feature_state = state.get("feature_extractor_state_dict")
        if feature_state is None:
            raise KeyError("Checkpoint state is missing 'feature_extractor_state_dict'.")
        self.feature_extractor.load_state_dict(feature_state, strict=strict)

        device = self._model_device()
        mean_embeddings = state.get("mean_embeddings")
        inv_covariances = state.get("inv_covariances")
        selected_indices = state.get("selected_indices")

        self.mean_embeddings = None if mean_embeddings is None else mean_embeddings.to(device)
        self.inv_covariances = None if inv_covariances is None else inv_covariances.to(device)
        self.selected_indices = None if selected_indices is None else selected_indices.cpu()
        feature_map_size = state.get("feature_map_size")
        self.feature_map_size = tuple(feature_map_size) if feature_map_size is not None else None

    def save(self, path: str | Path, extra_state: dict[str, Any] | None = None) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        state = self.export_state()
        if extra_state:
            state.update(extra_state)
        torch.save(state, checkpoint_path)


def build_padim_wrapper(cfg: dict[str, Any]) -> PaDiMWrapper:
    return PaDiMWrapper.from_config(cfg)
