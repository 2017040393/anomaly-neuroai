from __future__ import annotations

import torch
from torch import nn

try:
    from spikingjelly.activation_based import functional, layer, neuron
except ImportError as exc:  # pragma: no cover - handled when snn is explicitly requested
    functional = None
    layer = None
    neuron = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class SNNClassifier(nn.Module):
    """Minimal spiking MLP for synthetic feature classification."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, time_steps: int) -> None:
        if IMPORT_ERROR is not None:
            raise ImportError(
                "spikingjelly is required to build `snn_mlp`. Install project dependencies first."
            ) from IMPORT_ERROR
        if time_steps <= 0:
            raise ValueError("time_steps must be a positive integer.")

        super().__init__()
        self.time_steps = time_steps
        self.fc1 = layer.Linear(input_dim, hidden_dim)
        self.lif = neuron.LIFNode()
        self.fc2 = layer.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        functional.reset_net(self)
        outputs = []
        for _ in range(self.time_steps):
            hidden = self.fc1(x)
            spikes = self.lif(hidden)
            outputs.append(self.fc2(spikes))
        functional.reset_net(self)
        return torch.stack(outputs, dim=0).mean(dim=0)
