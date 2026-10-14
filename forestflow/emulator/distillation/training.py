"""Training utilities for deterministic ForestFlow student networks."""

from dataclasses import dataclass
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import DatasetSplit
from .network import MLPConfig, build_mlp


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization settings for a deterministic student network."""
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    batch_size: int = 256
    max_epochs: int = 1000
    patience: int = 75
    seed: int = 0


@dataclass
class TrainingResult:
    """Best checkpoint and learning curves from one student training run."""
    model: torch.nn.Module
    best_epoch: int
    best_validation_mse: float
    training_loss: list[float]
    validation_loss: list[float]


def compare_candidates(configurations: list[MLPConfig], train: DatasetSplit,
                       validation: DatasetSplit, training: TrainingConfig = TrainingConfig()):
    """Train configurations and return results keyed by their immutable configs."""
    return {config: train_student(config, train, validation, training) for config in configurations}


def train_student(config: MLPConfig, train: DatasetSplit, validation: DatasetSplit,
                  training: TrainingConfig = TrainingConfig()) -> TrainingResult:
    """Train in transformed output space and restore the best validation state."""
    if train.input_labels != validation.input_labels or train.output_labels != validation.output_labels:
        raise ValueError("training and validation labels must match")
    if train.inputs.shape[1] != config.input_dim or train.targets.shape[1] != config.output_dim:
        raise ValueError("dataset dimensions do not match MLPConfig")
    torch.manual_seed(training.seed)
    model = build_mlp(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate,
                                  weight_decay=training.weight_decay)
    generator = torch.Generator().manual_seed(training.seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(train.inputs), torch.from_numpy(train.targets)),
                        batch_size=training.batch_size, shuffle=True, generator=generator)
    xval = torch.from_numpy(validation.inputs)
    yval = torch.from_numpy(validation.targets)
    best_state, best_loss, best_epoch, stale = None, float("inf"), -1, 0
    train_history, validation_history = [], []
    for epoch in range(training.max_epochs):
        model.train()
        losses = []
        for xbatch, ybatch in loader:
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(xbatch), ybatch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            validation_loss = torch.nn.functional.mse_loss(model(xval), yval).item()
        train_history.append(float(np.mean(losses)))
        validation_history.append(validation_loss)
        if validation_loss < best_loss:
            best_state, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), validation_loss, epoch, 0
        else:
            stale += 1
            if stale >= training.patience:
                break
    model.load_state_dict(best_state)
    return TrainingResult(model, best_epoch, best_loss, train_history, validation_history)
