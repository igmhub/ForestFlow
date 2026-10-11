"""Small fully connected deterministic student networks."""

from dataclasses import dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class MLPConfig:
    input_dim: int
    output_dim: int
    hidden_dim: int = 64
    hidden_layers: int = 3
    activation: str = "silu"

    def __post_init__(self):
        if min(self.input_dim, self.output_dim, self.hidden_dim, self.hidden_layers) < 1:
            raise ValueError("MLP dimensions and hidden_layers must be positive")
        if self.activation not in {"silu", "tanh"}:
            raise ValueError("activation must be 'silu' or 'tanh'")


def build_mlp(config: MLPConfig) -> nn.Sequential:
    """Build an MLP with a linear output in transformed coefficient space."""
    activation = nn.SiLU if config.activation == "silu" else nn.Tanh
    layers: list[nn.Module] = [nn.Linear(config.input_dim, config.hidden_dim), activation()]
    for _ in range(config.hidden_layers - 1):
        layers.extend((nn.Linear(config.hidden_dim, config.hidden_dim), activation()))
    layers.append(nn.Linear(config.hidden_dim, config.output_dim))
    return nn.Sequential(*layers)
