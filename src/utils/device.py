from __future__ import annotations

import torch


def get_device(device_name: str) -> torch.device:
    requested = str(device_name).lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
