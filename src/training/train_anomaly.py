from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from src.inference.predict_anomaly import (
    build_anomaly_model,
    prepare_mvtec_loaders,
    run_anomaly_inference,
    write_metrics_csv,
)
from src.utils.config import load_yaml_config, parse_args
from src.utils.device import get_device
from src.utils.seed import set_seed


def _validate_quantile(value: float, name: str) -> float:
    quantile = float(value)
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {quantile}.")
    return quantile


@torch.inference_mode()
def calibrate_thresholds_from_train_split(
    model,
    data_loader,
    device: torch.device,
    image_quantile: float,
    pixel_quantile: float,
) -> tuple[float, float, dict[str, Any]]:
    model.eval()

    image_quantile = _validate_quantile(image_quantile, "image_quantile")
    pixel_quantile = _validate_quantile(pixel_quantile, "pixel_quantile")

    image_score_batches: list[np.ndarray] = []
    pixel_score_batches: list[np.ndarray] = []

    start_time = perf_counter()
    progress = tqdm(data_loader, desc="calibrate_thresholds", leave=False)
    for batch in progress:
        images = batch["image"].to(device)
        outputs = model.predict(images)

        image_scores = outputs.image_scores.detach().cpu().numpy().astype(np.float32)
        anomaly_maps = outputs.anomaly_maps.detach().cpu().numpy().astype(np.float32)

        image_score_batches.append(image_scores.reshape(-1))
        pixel_score_batches.append(anomaly_maps.reshape(-1))
        progress.set_postfix(score=f"{float(np.mean(image_scores)):.4f}")

    calibration_time = perf_counter() - start_time
    if not image_score_batches:
        raise RuntimeError("Received an empty train_loader while calibrating anomaly thresholds.")

    image_scores_array = np.concatenate(image_score_batches, axis=0)
    pixel_scores_array = np.concatenate(pixel_score_batches, axis=0)

    image_threshold = float(np.quantile(image_scores_array, image_quantile))
    pixel_threshold = float(np.quantile(pixel_scores_array, pixel_quantile))
    calibration_metrics = {
        "train_num_images": int(image_scores_array.size),
        "train_image_score_mean": float(image_scores_array.mean()),
        "train_image_score_std": float(image_scores_array.std()),
        "train_image_score_max": float(image_scores_array.max()),
        "train_image_score_p95": float(np.quantile(image_scores_array, 0.95)),
        "train_pixel_score_mean": float(pixel_scores_array.mean()),
        "train_pixel_score_std": float(pixel_scores_array.std()),
        "train_pixel_score_max": float(pixel_scores_array.max()),
        "train_pixel_score_p99": float(np.quantile(pixel_scores_array, 0.99)),
        "train_calibration_time_sec": float(calibration_time),
        "train_image_threshold_quantile": float(image_quantile),
        "train_pixel_threshold_quantile": float(pixel_quantile),
    }
    return image_threshold, pixel_threshold, calibration_metrics


def main() -> None:
    args = parse_args("Fit an anomaly detector on MVTec AD.")
    cfg = load_yaml_config(args.config)

    set_seed(int(cfg.get("seed", 42)))
    device = get_device(str(cfg.get("device", "cpu")))
    train_loader, test_loader = prepare_mvtec_loaders(cfg)
    model = build_anomaly_model(cfg, device)
    run_dir = Path(cfg.get("output_dir", "results/runs")) / str(cfg["experiment_name"])
    metrics_path = run_dir / "metrics.csv"
    predictions_output_path = Path(cfg["save"]["prediction_dir"]) / "predictions.csv"
    visualization_dir = cfg["save"].get("visualization_dir")

    train_cfg = cfg.get("train", {})
    inference_cfg = cfg.get("inference", {})
    configured_epochs = int(train_cfg.get("epochs", 1))
    image_threshold = inference_cfg.get("image_threshold")
    pixel_threshold = inference_cfg.get("pixel_threshold")

    if configured_epochs != 1:
        print(
            f"Anomaly model '{cfg['model']['name']}' uses one-pass fitting; "
            f"recording train.epochs={configured_epochs} without iterative optimization."
        )

    fit_start = perf_counter()
    model.fit(train_loader)
    fit_time = perf_counter() - fit_start

    train_image_threshold, train_pixel_threshold, calibration_metrics = calibrate_thresholds_from_train_split(
        model=model,
        data_loader=train_loader,
        device=device,
        image_quantile=float(train_cfg.get("image_threshold_quantile", 0.99)),
        pixel_quantile=float(train_cfg.get("pixel_threshold_quantile", 0.999)),
    )
    image_threshold_source = "config" if image_threshold is not None else "train_quantile"
    pixel_threshold_source = "config" if pixel_threshold is not None else "train_quantile"
    image_threshold = train_image_threshold if image_threshold is None else float(image_threshold)
    pixel_threshold = train_pixel_threshold if pixel_threshold is None else float(pixel_threshold)

    metrics, _ = run_anomaly_inference(
        model=model,
        data_loader=test_loader,
        device=device,
        normalization=str(cfg["data"].get("normalization", "imagenet")),
        image_threshold=image_threshold,
        pixel_threshold=pixel_threshold,
        predictions_output_path=predictions_output_path,
        visualization_dir=visualization_dir,
        save_maps=bool(inference_cfg.get("save_maps", True)),
        save_overlay=bool(inference_cfg.get("save_overlay", True)),
    )
    metrics.update(calibration_metrics)
    metrics["configured_epochs"] = int(configured_epochs)
    metrics["effective_epochs"] = 1
    metrics["fit_time_sec"] = float(fit_time)
    metrics["total_train_time_sec"] = float(fit_time + calibration_metrics["train_calibration_time_sec"])
    metrics["image_threshold_source"] = image_threshold_source
    metrics["pixel_threshold_source"] = pixel_threshold_source
    write_metrics_csv(metrics_path, metrics)

    checkpoint_dir = Path(cfg["save"]["checkpoint_dir"])
    checkpoint_name = "best.pt" if bool(cfg["save"].get("save_best", True)) else "last.pt"
    checkpoint_path = checkpoint_dir / checkpoint_name
    model.save(
        checkpoint_path,
        extra_state={
            "config": cfg,
            "metrics": metrics,
            "train_calibration": calibration_metrics,
            "thresholds": {
                "image_threshold": metrics.get("image_threshold"),
                "pixel_threshold": metrics.get("pixel_threshold"),
            },
        },
    )

    print(
        f"Finished anomaly training | image_auroc={metrics['image_auroc']:.4f} "
        f"| pixel_auroc={metrics['pixel_auroc']:.4f} "
        f"| image_threshold={image_threshold:.6f} "
        f"| pixel_threshold={pixel_threshold:.6f} "
        f"| fit_time={fit_time:.2f}s | checkpoint={checkpoint_path}"
    )


if __name__ == "__main__":
    main()
