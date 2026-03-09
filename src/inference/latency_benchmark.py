from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from src.inference.predict_anomaly import load_anomaly_model, prepare_mvtec_loaders
from src.utils.config import load_yaml_config, parse_args
from src.utils.device import get_device
from src.utils.seed import set_seed


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args("Benchmark anomaly inference latency.")
    cfg = load_yaml_config(args.config)

    set_seed(int(cfg.get("seed", 42)))
    device = get_device(str(cfg.get("device", "cpu")))
    _, test_loader = prepare_mvtec_loaders(cfg)
    model, _, checkpoint_path = load_anomaly_model(cfg, args.ckpt, device)

    try:
        batch = next(iter(test_loader))
    except StopIteration as exc:
        raise RuntimeError("Test dataloader is empty. Cannot run latency benchmark.") from exc

    images = batch["image"].to(device)
    warmup_runs = int(cfg.get("benchmark", {}).get("warmup_runs", 10))
    timed_runs = int(cfg.get("benchmark", {}).get("timed_runs", 50))

    with torch.inference_mode():
        for _ in range(warmup_runs):
            _ = model.predict(images)
        synchronize_if_needed(device)

        latencies_ms: list[float] = []
        for _ in range(timed_runs):
            start = perf_counter()
            _ = model.predict(images)
            synchronize_if_needed(device)
            latencies_ms.append((perf_counter() - start) * 1000.0)

    latencies = np.asarray(latencies_ms, dtype=np.float64)
    mean_ms = float(latencies.mean())
    std_ms = float(latencies.std(ddof=0))
    p95_ms = float(np.percentile(latencies, 95))
    fps = float(images.shape[0] / (mean_ms / 1000.0)) if mean_ms > 0 else float("inf")

    run_dir = Path(cfg.get("output_dir", "results/runs")) / str(cfg["experiment_name"])
    output_path = run_dir / "latency_benchmark.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "checkpoint_path": str(checkpoint_path),
        "device": device.type,
        "batch_size": int(images.shape[0]),
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        "mean_latency_ms": mean_ms,
        "std_latency_ms": std_ms,
        "p95_latency_ms": p95_ms,
        "throughput_fps": fps,
    }
    pd.DataFrame([row]).to_csv(output_path, index=False)

    print(
        f"Latency benchmark | mean={mean_ms:.2f}ms | std={std_ms:.2f}ms "
        f"| p95={p95_ms:.2f}ms | throughput={fps:.2f} fps | output={output_path}"
    )


if __name__ == "__main__":
    main()
