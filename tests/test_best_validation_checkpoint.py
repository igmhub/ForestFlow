from types import MethodType

import numpy as np
import torch

from forestflow.P3D_cINN import P3DEmulator


class _ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def apply(self, function):
        return self


def test_training_saves_and_restores_best_validation_checkpoint(tmp_path):
    trainer = object.__new__(P3DEmulator)
    trainer.input_labels = ["input"]
    trainer.output_labels = ["output"]
    model = _ScalarModel()
    training_weights = iter([1.0, 2.0, 3.0])
    validation_losses = iter([3.0, 1.0, 2.0])

    trainer._define_cINN_Arinyo = MethodType(
        lambda self, *args, **kwargs: model, trainer
    )
    trainer._prepare_training_data = MethodType(
        lambda self, data: (torch.zeros((10, 1)), torch.zeros((10, 1))),
        trainer,
    )
    trainer._create_data_loaders = MethodType(
        lambda self, *args, **kwargs: ([None], [None]), trainer
    )
    trainer._setup_optimizer = MethodType(
        lambda self, *args, **kwargs: torch.optim.SGD(model.parameters(), lr=0.1),
        trainer,
    )

    def train_epoch(self, optimizer, loader):
        value = next(training_weights)
        model.weight.data.fill_(value)
        return value

    trainer._train_epoch = MethodType(train_epoch, trainer)
    trainer._compute_validation_loss = MethodType(
        lambda self, loader: next(validation_losses), trainer
    )
    trainer._log_training_progress = MethodType(
        lambda self, *args, **kwargs: None, trainer
    )

    save_path = str(tmp_path / "emulator")
    trainer._train_emulator(
        {"input_par": {}, "output_par": {}},
        nepochs=3,
        save_path=save_path,
        use_val_set=True,
    )

    assert trainer.best_epoch == 1
    assert trainer.best_validation_loss == 1.0
    assert model.weight.item() == 2.0

    metadata = np.load(save_path + "_metadata.npy", allow_pickle=True).item()
    assert metadata["best_epoch"] == 1
    assert metadata["best_validation_loss"] == 1.0
    state = torch.load(save_path + ".pt", weights_only=True)
    assert state["weight"].item() == 2.0
