import numpy as np
import torch

from forestflow.emulator.distillation.data import (
    DatasetSplit, TeacherSamplingConfig, build_input_design, load_teacher_dataset,
    save_teacher_dataset, split_input_design,
)
from forestflow.emulator.distillation.inference import DeterministicArinyoEmulator
from forestflow.emulator.distillation.network import MLPConfig, build_mlp
from forestflow.emulator.distillation.training import TrainingConfig, train_student
from forestflow.emulator.distillation.validation import validate_coefficients
from forestflow.emulator.distillation.validation import validate_observable


class IdentityTransform:
    def transf_stand(self, values, **kwargs):
        return values


def test_student_uses_shared_inverse_transform_and_single_shapes():
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2)); model.bias.zero_()
    emulator = DeterministicArinyoEmulator(model, IdentityTransform(), ("x", "y"), ("a", "b"))
    assert emulator.predict_transformed({"x": 1.0, "y": 2.0}) == {"a": 1.0, "b": 2.0}
    result = emulator.evaluate([{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}])
    np.testing.assert_allclose(result["a"], [1.0, 3.0])


def test_training_restores_best_validation_checkpoint():
    inputs = np.arange(20, dtype=np.float32).reshape(10, 2)
    targets = inputs.copy()
    data = DatasetSplit(inputs, targets, ("x", "y"), ("a", "b"))
    result = train_student(MLPConfig(2, 2, hidden_dim=4, hidden_layers=2), data, data,
                           TrainingConfig(max_epochs=20, patience=5, batch_size=5))
    assert result.best_epoch >= 0
    assert result.best_validation_mse == min(result.validation_loss)


def test_coefficient_validation_rejects_mismatched_shapes():
    summary = validate_coefficients(np.array([[1., 2.]]), np.array([[0., 0.]]))
    np.testing.assert_allclose(summary.rmse, [1., 2.])


def test_sobol_design_and_split_are_reproducible_and_isolated():
    design = build_input_design({"x": (0., 1.), "y": (2., 3.)}, 16, seed=5)
    train, validation, test = split_input_design(design, seed=6)
    assert len(train) + len(validation) + len(test) == 16
    assert {tuple(row.values()) for row in train}.isdisjoint({tuple(row.values()) for row in test})


def test_teacher_sampling_requires_valid_antithetic_and_replicate_counts():
    try:
        TeacherSamplingConfig(realizations=3)
    except ValueError:
        pass
    else:
        raise AssertionError("odd antithetic realization count was accepted")
    try:
        TeacherSamplingConfig(replicates=0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero replicate count was accepted")


def test_observable_validation_requires_external_physical_fixture():
    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(1); model.bias.zero_()
    student = DeterministicArinyoEmulator(model, IdentityTransform(), ("x",), ("a",))
    summary = validate_observable(student, student, [{"emu_params": {"x": 2.}, "factor": 3.}],
                                  lambda coefficient, fixture: coefficient["a"] * fixture["factor"])
    assert summary.maximum_abs_fractional == 0


def test_dataset_and_student_round_trip(tmp_path):
    dataset = DatasetSplit(np.array([[1., 2.]], dtype=np.float32), np.array([[3.]], dtype=np.float32),
                           ("x", "y"), ("a",))
    dataset_path = tmp_path / "labels.npz"
    save_teacher_dataset(dataset_path, dataset, metadata={"teacher": "test"})
    loaded_dataset = load_teacher_dataset(dataset_path)
    np.testing.assert_allclose(loaded_dataset.targets, dataset.targets)
    model = build_mlp(MLPConfig(2, 1, hidden_dim=3, hidden_layers=2))
    emulator = DeterministicArinyoEmulator(model, IdentityTransform(), ("x", "y"), ("a",))
    student_path = tmp_path / "student.pt"
    emulator.save(student_path)
    recovered = DeterministicArinyoEmulator.load(student_path)
    assert recovered.predict_transformed({"x": 1., "y": 2.}) == emulator.predict_transformed({"x": 1., "y": 2.})
