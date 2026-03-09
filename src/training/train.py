from __future__ import annotations

import csv
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Adam
from tqdm.auto import tqdm

from src.data.loaders import build_dataloaders
from src.models.build_model import build_model
from src.training.metrics import accuracy_from_logits
from src.utils.config import load_yaml_config, parse_args
from src.utils.device import get_device
from src.utils.seed import set_seed


def _write_metrics_header(metrics_path: Path) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "elapsed_time_sec",
            ],
        )
        writer.writeheader()


def _append_metrics(metrics_path: Path, row: dict[str, Any]) -> None:
    with metrics_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        writer.writerow(row)


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        try:
            return torch.amp.autocast(device_type=device.type, enabled=True)
        except (AttributeError, TypeError):
            return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


def _build_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler(device="cuda", enabled=enabled)
    except (AttributeError, TypeError):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,
    use_amp: bool,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(loader, desc="train", leave=False)
    for features, targets in progress:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, use_amp):
            logits = model(features)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_samples += batch_size

        progress.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = total_loss / max(total_samples, 1)
    epoch_acc = total_correct / max(total_samples, 1)
    return epoch_loss, epoch_acc


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(loader, desc="eval", leave=False)
    for features, targets in progress:
        features = features.to(device)
        targets = targets.to(device)

        logits = model(features)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_samples += batch_size
        progress.set_postfix(acc=f"{accuracy_from_logits(logits, targets):.4f}")

    epoch_loss = total_loss / max(total_samples, 1)
    epoch_acc = total_correct / max(total_samples, 1)
    return epoch_loss, epoch_acc


def save_checkpoint(path: Path, model: nn.Module, cfg: dict[str, Any], epoch: int, val_acc: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "val_acc": val_acc,
        },
        path,
    )


def main() -> None:
    args = parse_args("Train a minimal NeuroAI baseline.")
    cfg = load_yaml_config(args.config)

    set_seed(int(cfg.get("seed", 42)))
    device = get_device(str(cfg.get("device", "cpu")))
    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=float(cfg["train"]["lr"]))
    amp_enabled = bool(cfg["train"].get("use_amp", False)) and device.type == "cuda"
    scaler = _build_grad_scaler(enabled=amp_enabled)

    experiment_name = str(cfg["experiment_name"])
    run_dir = Path(cfg.get("output_dir", "results/runs")) / experiment_name
    metrics_path = run_dir / "metrics.csv"
    _write_metrics_header(metrics_path)

    checkpoint_dir = Path(cfg["save"]["checkpoint_dir"])
    save_best = bool(cfg["save"].get("save_best", True))
    best_val_acc = float("-inf")

    epochs = int(cfg["train"]["epochs"])
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=amp_enabled,
        )
        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )
        elapsed_time = time.perf_counter() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "train_acc": f"{train_acc:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "val_acc": f"{val_acc:.6f}",
            "elapsed_time_sec": f"{elapsed_time:.3f}",
        }
        _append_metrics(metrics_path, row)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"elapsed={elapsed_time:.2f}s"
        )

        if save_best and val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(checkpoint_dir / "best.pt", model, cfg, epoch, val_acc)

    if not save_best:
        save_checkpoint(checkpoint_dir / "last.pt", model, cfg, epochs, val_acc)


if __name__ == "__main__":
    main()
