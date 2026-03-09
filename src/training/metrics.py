from __future__ import annotations

import torch


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    correct = (predictions == targets).sum().item()
    total = targets.numel()
    return float(correct / total) if total > 0 else 0.0
