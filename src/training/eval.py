from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.data.loaders import build_dataloaders
from src.models.build_model import build_model
from src.training.train import evaluate
from src.utils.config import load_yaml_config, parse_args
from src.utils.device import get_device
from src.utils.seed import set_seed


def resolve_checkpoint_path(config: dict, ckpt_path: Path | None) -> Path:
    if ckpt_path is not None:
        return ckpt_path
    return Path(config["save"]["checkpoint_dir"]) / "best.pt"


def main() -> None:
    args = parse_args("Evaluate a trained NeuroAI baseline.")
    cfg = load_yaml_config(args.config)

    set_seed(int(cfg.get("seed", 42)))
    device = get_device(str(cfg.get("device", "cpu")))

    _, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)

    checkpoint_path = resolve_checkpoint_path(cfg, args.ckpt)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Run training first or provide --ckpt.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = evaluate(model=model, loader=val_loader, criterion=criterion, device=device)
    print(f"Validation results | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")


if __name__ == "__main__":
    main()
