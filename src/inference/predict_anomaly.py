from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm.auto import tqdm

from src.data.anomaly_transforms import build_mvtec_transforms
from src.data.mvtec import build_mvtec_dataloaders
from src.models.padim_wrapper import PaDiMWrapper
from src.models.patchcore_wrapper import PatchCoreWrapper
from src.utils.config import load_yaml_config, parse_args
from src.utils.device import get_device
from src.utils.seed import set_seed
from src.visualization.save_anomaly_maps import save_batch_anomaly_maps


def build_anomaly_model(cfg: dict[str, Any], device: torch.device):
    model_name = str(cfg["model"]["name"]).lower()
    if model_name == "patchcore":
        return PatchCoreWrapper.from_config(cfg).to(device)
    if model_name == "padim":
        return PaDiMWrapper.from_config(cfg).to(device)
    raise ValueError(f"Unsupported anomaly model '{model_name}'. Expected one of ['patchcore', 'padim'].")


def resolve_checkpoint_path(cfg: dict[str, Any], ckpt_path: Path | None) -> Path:
    if ckpt_path is not None:
        return ckpt_path
    return Path(cfg["save"]["checkpoint_dir"]) / "best.pt"


def load_anomaly_model(
    cfg: dict[str, Any],
    ckpt_path: Path | None,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], Path]:
    checkpoint_path = resolve_checkpoint_path(cfg, ckpt_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Run `train_anomaly.py` first or provide --ckpt."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_anomaly_model(cfg, device)
    model.load_exported_state(checkpoint)
    model.eval()
    return model, checkpoint, checkpoint_path


def prepare_mvtec_loaders(cfg: dict[str, Any]):
    train_transform, eval_transform, mask_transform = build_mvtec_transforms(cfg)
    return build_mvtec_dataloaders(
        cfg=cfg,
        train_transform=train_transform,
        eval_transform=eval_transform,
        mask_transform=mask_transform,
    )


def safe_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    scores = np.asarray(scores).astype(np.float32).reshape(-1)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def compute_optimal_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    scores = np.asarray(scores).astype(np.float32).reshape(-1)

    if scores.size == 0:
        raise ValueError("Cannot compute a threshold from empty scores.")
    if np.unique(labels).size < 2:
        return float(np.quantile(scores, 0.95))

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return float(np.median(scores))

    f1 = (2.0 * precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-8)
    best_index = int(np.nanargmax(f1))
    return float(thresholds[best_index])


def compute_anomaly_metrics(
    image_labels: np.ndarray,
    image_scores: np.ndarray,
    pixel_masks: np.ndarray | None = None,
    pixel_scores: np.ndarray | None = None,
    image_threshold: float | None = None,
    pixel_threshold: float | None = None,
) -> dict[str, Any]:
    image_labels = np.asarray(image_labels).astype(np.int64).reshape(-1)
    image_scores = np.asarray(image_scores).astype(np.float32).reshape(-1)

    metrics: dict[str, Any] = {
        "num_images": int(image_labels.size),
        "num_anomalous_images": int(image_labels.sum()),
        "num_normal_images": int((image_labels == 0).sum()),
        "image_auroc": safe_roc_auc(image_labels, image_scores),
    }

    if image_threshold is not None:
        image_predictions = (image_scores >= image_threshold).astype(np.int64)
        metrics.update(
            {
                "image_threshold": float(image_threshold),
                "image_accuracy": float(accuracy_score(image_labels, image_predictions)),
                "image_precision": float(precision_score(image_labels, image_predictions, zero_division=0)),
                "image_recall": float(recall_score(image_labels, image_predictions, zero_division=0)),
                "image_f1": float(f1_score(image_labels, image_predictions, zero_division=0)),
            }
        )

    if pixel_masks is not None and pixel_scores is not None:
        flat_masks = np.asarray(pixel_masks).astype(np.int64).reshape(-1)
        flat_scores = np.asarray(pixel_scores).astype(np.float32).reshape(-1)
        metrics["pixel_auroc"] = safe_roc_auc(flat_masks, flat_scores)

        if pixel_threshold is not None:
            pixel_predictions = (flat_scores >= pixel_threshold).astype(np.int64)
            metrics.update(
                {
                    "pixel_threshold": float(pixel_threshold),
                    "pixel_accuracy": float(accuracy_score(flat_masks, pixel_predictions)),
                    "pixel_precision": float(precision_score(flat_masks, pixel_predictions, zero_division=0)),
                    "pixel_recall": float(recall_score(flat_masks, pixel_predictions, zero_division=0)),
                    "pixel_f1": float(f1_score(flat_masks, pixel_predictions, zero_division=0)),
                }
            )
    else:
        metrics["pixel_auroc"] = float("nan")

    return metrics


def write_metrics_csv(path: str | Path, metrics: dict[str, Any]) -> None:
    metrics_path = Path(path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)


def run_anomaly_inference(
    model,
    data_loader,
    device: torch.device,
    normalization: str = "imagenet",
    image_threshold: float | None = None,
    pixel_threshold: float | None = None,
    predictions_output_path: str | Path | None = None,
    visualization_dir: str | Path | None = None,
    save_maps: bool = False,
    save_overlay: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()

    rows: list[dict[str, Any]] = []
    image_score_values: list[float] = []
    image_label_values: list[int] = []
    pixel_score_batches: list[np.ndarray] = []
    pixel_mask_batches: list[np.ndarray] = []

    start_time = perf_counter()
    progress = tqdm(data_loader, desc="predict_anomaly", leave=False)
    for batch in progress:
        images = batch["image"].to(device)
        outputs = model.predict(images)

        image_scores = outputs.image_scores.detach().cpu().numpy().astype(np.float32)
        anomaly_maps = outputs.anomaly_maps.detach().cpu().numpy().astype(np.float32)
        labels = batch["label"].detach().cpu().numpy().astype(np.int64)
        masks = batch["mask"].detach().cpu().numpy().astype(np.int64)

        if visualization_dir is not None and (save_maps or save_overlay):
            artifact_rows = save_batch_anomaly_maps(
                batch=batch,
                anomaly_maps=torch.from_numpy(anomaly_maps),
                output_dir=visualization_dir,
                normalization=normalization,
                image_scores=image_scores.tolist(),
                save_overlay=save_overlay,
                save_maps=save_maps,
            )
        else:
            artifact_rows = [{} for _ in range(len(image_scores))]

        image_names = list(batch["image_name"])
        defect_types = list(batch["defect_type"])
        image_paths = list(batch["image_path"])
        mask_paths = list(batch["mask_path"])

        for index, score in enumerate(image_scores.tolist()):
            row = {
                "image_name": image_names[index],
                "image_path": image_paths[index],
                "mask_path": mask_paths[index],
                "defect_type": defect_types[index],
                "label": int(labels[index]),
                "image_score": float(score),
                "pred_label": "",
            }
            row.update(artifact_rows[index])
            rows.append(row)

        image_score_values.extend(float(value) for value in image_scores.tolist())
        image_label_values.extend(int(value) for value in labels.tolist())
        pixel_score_batches.append(anomaly_maps)
        pixel_mask_batches.append(masks)

        progress.set_postfix(score=f"{float(np.mean(image_scores)):.4f}")

    elapsed_time = perf_counter() - start_time

    image_scores_array = np.asarray(image_score_values, dtype=np.float32)
    image_labels_array = np.asarray(image_label_values, dtype=np.int64)
    pixel_scores_array = np.concatenate(pixel_score_batches, axis=0) if pixel_score_batches else None
    pixel_masks_array = np.concatenate(pixel_mask_batches, axis=0) if pixel_mask_batches else None

    if image_threshold is None:
        image_threshold = compute_optimal_threshold(image_labels_array, image_scores_array)
    if pixel_threshold is None and pixel_scores_array is not None and pixel_masks_array is not None:
        pixel_threshold = compute_optimal_threshold(pixel_masks_array, pixel_scores_array)

    metrics = compute_anomaly_metrics(
        image_labels=image_labels_array,
        image_scores=image_scores_array,
        pixel_masks=pixel_masks_array,
        pixel_scores=pixel_scores_array,
        image_threshold=image_threshold,
        pixel_threshold=pixel_threshold,
    )
    metrics["inference_time_sec"] = float(elapsed_time)

    if image_threshold is not None:
        for row in rows:
            row["pred_label"] = int(float(row["image_score"]) >= image_threshold)

    if predictions_output_path is not None:
        predictions_path = Path(predictions_output_path)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(predictions_path, index=False)

    return metrics, rows


def main() -> None:
    args = parse_args("Run anomaly inference on MVTec AD.")
    cfg = load_yaml_config(args.config)

    set_seed(int(cfg.get("seed", 42)))
    device = get_device(str(cfg.get("device", "cpu")))
    _, test_loader = prepare_mvtec_loaders(cfg)
    model, checkpoint, checkpoint_path = load_anomaly_model(cfg, args.ckpt, device)

    thresholds = checkpoint.get("thresholds", {})
    image_threshold = thresholds.get("image_threshold", cfg.get("inference", {}).get("image_threshold"))
    pixel_threshold = thresholds.get("pixel_threshold", cfg.get("inference", {}).get("pixel_threshold"))

    run_dir = Path(cfg.get("output_dir", "results/runs")) / str(cfg["experiment_name"])
    predictions_output_path = Path(cfg["save"]["prediction_dir"]) / "predictions.csv"
    metrics_output_path = run_dir / "prediction_metrics.csv"

    metrics, _ = run_anomaly_inference(
        model=model,
        data_loader=test_loader,
        device=device,
        normalization=str(cfg["data"].get("normalization", "imagenet")),
        image_threshold=image_threshold,
        pixel_threshold=pixel_threshold,
        predictions_output_path=predictions_output_path,
        visualization_dir=cfg["save"].get("visualization_dir"),
        save_maps=bool(cfg.get("inference", {}).get("save_maps", True)),
        save_overlay=bool(cfg.get("inference", {}).get("save_overlay", True)),
    )
    metrics["checkpoint_path"] = str(checkpoint_path)
    write_metrics_csv(metrics_output_path, metrics)

    print(
        f"Inference complete | image_auroc={metrics['image_auroc']:.4f} "
        f"| pixel_auroc={metrics['pixel_auroc']:.4f} "
        f"| checkpoint={checkpoint_path}"
    )


if __name__ == "__main__":
    main()
