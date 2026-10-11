# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: lace
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 1. Train a deterministic emulator of `forest_mpg_fix`
#
# This notebook distils the *transformed-space mean* produced by the
# stochastic ForestFlow flow into a small deterministic MLP. The result is a
# faster approximation to `forest_mpg_fix`; it does not provide a new physical
# simulation emulator or improve ForestFlow's physical precision.
#
# Run this notebook from the ForestFlow repository root. It writes only under
# `notebooks/deterministic_distillation/artifacts/`.

# %%
from pathlib import Path
import json
import numpy as np

from forestflow.emulator.p3d_cinn import P3DEmulator
from forestflow.emulator.distillation.data import (
    TeacherSamplingConfig,
    build_input_design,
    calibrate_teacher_sampling,
    generate_teacher_dataset,
    save_teacher_dataset,
    split_input_design,
)
from forestflow.emulator.distillation.inference import DeterministicArinyoEmulator
from forestflow.emulator.distillation.network import MLPConfig
from forestflow.emulator.distillation.training import TrainingConfig, compare_candidates

ARTIFACTS = Path("notebooks/deterministic_distillation/artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load the teacher and define a physically valid design
#
# The numbers below are deliberately left as a user-owned configuration. Derive
# them from the supported archive/inference domain for the analysis being
# accelerated. A rectangular min/max box can include unsupported combinations;
# provide `validity` when needed. Keep trajectory points together in one split.

# %%
teacher = P3DEmulator(key="forest_mpg_fix")
print("inputs:", teacher.input_labels)
print("outputs:", teacher.output_labels)

# Replace these bounds with a reviewed scientific domain before a production run.
# The smoke configuration is for code-path verification only.
SMOKE = True
bounds = {
    "Delta2_p": (0.25, 0.45),
    "n_p": (-2.45, -2.15),
    "mF": (0.15, 0.35),
    "sigT_Mpc": (0.07, 0.14),
    "gamma": (1.0, 1.4),
    "kF_Mpc": (9.0, 20.0),
}
number_of_inputs = 64 if SMOKE else 16_384
design = build_input_design(bounds, number_of_inputs, seed=100)
train_inputs, validation_inputs, test_inputs = split_input_design(
    design, validation_fraction=0.125, test_fraction=0.125, seed=101
)

with (ARTIFACTS / "input_splits.json").open("w") as stream:
    json.dump(
        {"train": train_inputs, "validation": validation_inputs, "test": test_inputs},
        stream, indent=2,
    )

# %% [markdown]
# ## Establish teacher-label precision first
#
# The label uncertainty must be materially below the student error budget. This
# measures Monte Carlo integration noise in transformed coefficient space; it is
# not ForestFlow's calibrated simulation-emulation uncertainty. Use several
# independent scrambled sequences. Do this on representative interior and
# boundary points before choosing the production realization count.

# %%
pilot_inputs = train_inputs[: min(8, len(train_inputs))]
pilot_counts = (32, 128) if SMOKE else (1024, 4096, 16384)
convergence = calibrate_teacher_sampling(
    teacher,
    pilot_inputs,
    [TeacherSamplingConfig(realizations=count, seed=102, replicates=3) for count in pilot_counts],
)
for result in convergence:
    print(result.realizations, dict(zip(teacher.output_labels, result.replicate_rmse)))

# %% [markdown]
# Select the count only after inspecting these results and the corresponding
# P1D/P3D impact. The setting below is intentionally small in smoke mode.

# %%
sampling = TeacherSamplingConfig(
    realizations=32 if SMOKE else 4096,
    seed=103,
    replicates=1 if SMOKE else 2,
)
train = generate_teacher_dataset(teacher, train_inputs, sampling)
validation = generate_teacher_dataset(teacher, validation_inputs, sampling)
test = generate_teacher_dataset(teacher, test_inputs, sampling)

metadata = {
    "teacher_key": "forest_mpg_fix",
    "sampling": sampling.__dict__,
    "warning": "smoke labels are not suitable for a scientific production model" if SMOKE else None,
}
save_teacher_dataset(ARTIFACTS / "train_labels.npz", train, metadata=metadata)
save_teacher_dataset(ARTIFACTS / "validation_labels.npz", validation, metadata=metadata)
save_teacher_dataset(ARTIFACTS / "test_labels.npz", test, metadata=metadata)

# %% [markdown]
# ## Train the three candidate architectures
#
# All outputs are fitted jointly. The MSE is in the teacher's transformed and
# standardized space. The best checkpoint is selected by validation loss; do
# not use the untouched test split to choose an architecture.

# %%
candidates = [
    MLPConfig(input_dim=len(teacher.input_labels), output_dim=len(teacher.output_labels), hidden_dim=32, hidden_layers=2),
    MLPConfig(input_dim=len(teacher.input_labels), output_dim=len(teacher.output_labels), hidden_dim=64, hidden_layers=3),
    MLPConfig(input_dim=len(teacher.input_labels), output_dim=len(teacher.output_labels), hidden_dim=128, hidden_layers=3),
]
training = TrainingConfig(max_epochs=50 if SMOKE else 1000, patience=15 if SMOKE else 75, seed=104)
results = compare_candidates(candidates, train, validation, training)
for config, result in results.items():
    print(config, "epoch", result.best_epoch, "validation MSE", result.best_validation_mse)

# %% [markdown]
# Freeze the architecture selected from validation, then save its exact model,
# transform, label order, and provenance. The next notebooks load this artifact.

# %%
selected_config = min(results, key=lambda config: results[config].best_validation_mse)
selected_result = results[selected_config]
student = DeterministicArinyoEmulator(
    selected_result.model, teacher.transf_data, teacher.input_labels, teacher.output_labels
)
student_path = ARTIFACTS / "forest_mpg_fix_student.pt"
student.save(student_path, metadata={**metadata, "architecture": selected_config.__dict__, "best_epoch": selected_result.best_epoch})
print("saved", student_path, "with", selected_config)
