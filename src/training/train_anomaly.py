from __future__ import annotations

from pathlib import Path
from time import perf_counter

from src.inference.predict_anomaly import (
    build_anomaly_model,
    prepare_mvtec_loaders,
    run_anomaly_inference,
    write_metrics_csv,
)
from src.utils.config import load_yaml_config, parse_args
from src.utils.device import get_device
from src.utils.seed import set_seed


def main() -> None:
    args = parse_args("Fit an anomaly detector on MVTec AD.")
    cfg = load_yaml_config(args.config)

    set_seed(int(cfg.get("seed", 42)))
    device = get_device(str(cfg.get("device", "cpu")))
    train_loader, test_loader = prepare_mvtec_loaders(cfg)
    model = build_anomaly_model(cfg, device)

    fit_start = perf_counter()
    model.fit(train_loader)
    fit_time = perf_counter() - fit_start

    run_dir = Path(cfg.get("output_dir", "results/runs")) / str(cfg["experiment_name"])
    metrics_path = run_dir / "metrics.csv"
    predictions_output_path = Path(cfg["save"]["prediction_dir"]) / "predictions.csv"
    visualization_dir = cfg["save"].get("visualization_dir")

    metrics, _ = run_anomaly_inference(
        model=model,
        data_loader=test_loader,
        device=device,
        normalization=str(cfg["data"].get("normalization", "imagenet")),
        image_threshold=cfg.get("inference", {}).get("image_threshold"),
        pixel_threshold=cfg.get("inference", {}).get("pixel_threshold"),
        predictions_output_path=predictions_output_path,
        visualization_dir=visualization_dir,
        save_maps=bool(cfg.get("inference", {}).get("save_maps", True)),
        save_overlay=bool(cfg.get("inference", {}).get("save_overlay", True)),
    )
    metrics["fit_time_sec"] = float(fit_time)
    write_metrics_csv(metrics_path, metrics)

    checkpoint_dir = Path(cfg["save"]["checkpoint_dir"])
    checkpoint_name = "best.pt" if bool(cfg["save"].get("save_best", True)) else "last.pt"
    checkpoint_path = checkpoint_dir / checkpoint_name
    model.save(
        checkpoint_path,
        extra_state={
            "config": cfg,
            "metrics": metrics,
            "thresholds": {
                "image_threshold": metrics.get("image_threshold"),
                "pixel_threshold": metrics.get("pixel_threshold"),
            },
        },
    )

    print(
        f"Finished anomaly training | image_auroc={metrics['image_auroc']:.4f} "
        f"| pixel_auroc={metrics['pixel_auroc']:.4f} "
        f"| fit_time={fit_time:.2f}s | checkpoint={checkpoint_path}"
    )


if __name__ == "__main__":
    main()
