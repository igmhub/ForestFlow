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
# # Distilling `forest_mpg_fix` to a deterministic MLP
#
# This tutorial produces labels from the transformed mean of the ForestFlow
# teacher and trains a deterministic student. It measures additional error with
# respect to the teacher; it is not a new simulation-accuracy validation.

# %%
from forestflow.emulator.p3d_cinn import P3DEmulator
from forestflow.emulator.distillation.data import (
    TeacherSamplingConfig,
    build_input_design,
    generate_teacher_dataset,
    split_input_design,
)
from forestflow.emulator.distillation.inference import DeterministicArinyoEmulator
from forestflow.emulator.distillation.network import MLPConfig
from forestflow.emulator.distillation.training import TrainingConfig, compare_candidates
from forestflow.emulator.distillation.validation import validate_coefficients

# %% [markdown]
# Use bounds approved for the intended analysis. These deliberately compact
# bounds provide an execution example only; a production design should combine
# archive and inference-trajectory points, documented domain rules, and a
# separate boundary/stress set.

# %%
bounds = {
    "Delta2_p": (0.25, 0.45), "n_p": (-2.45, -2.15), "mF": (0.65, 0.85),
    "sigT_Mpc": (0.07, 0.14), "gamma": (1.0, 1.4), "kF_Mpc": (9.0, 20.0),
}
teacher = P3DEmulator(key="forest_mpg_fix")
inputs = build_input_design(bounds, count=256, seed=5)
train_inputs, validation_inputs, test_inputs = split_input_design(inputs, seed=6)

# %% [markdown]
# Establish the realization count using an independent-scramble convergence
# study before trusting it. The small value below is only suitable for a wiring
# smoke test; use a converged value such as 4096 only after measuring it for the
# target domain.

# %%
sampling = TeacherSamplingConfig(realizations=32, seed=7)
train = generate_teacher_dataset(teacher, train_inputs, sampling)
validation = generate_teacher_dataset(teacher, validation_inputs, sampling)
test = generate_teacher_dataset(teacher, test_inputs, sampling)

# %%
candidates = [
    MLPConfig(6, 8, hidden_dim=32, hidden_layers=2),
    MLPConfig(6, 8, hidden_dim=64, hidden_layers=3),
    MLPConfig(6, 8, hidden_dim=128, hidden_layers=3),
]
results = compare_candidates(candidates, train, validation,
                             TrainingConfig(max_epochs=1000, patience=75, seed=8))
[(config, result.best_validation_mse) for config, result in results.items()]

# %% [markdown]
# Select a candidate using validation only. The final test set remains untouched
# until this choice has been frozen. In a production run, compare against
# independent, higher-realization teacher labels and validate physical P1D/P3D
# with matched linear-theory fixtures.

# %%
selected_config = min(results, key=lambda config: results[config].best_validation_mse)
selected = results[selected_config]
student = DeterministicArinyoEmulator(
    selected.model, teacher.transf_data, teacher.input_labels, teacher.output_labels,
)
student_transformed = student.predict_transformed(test_inputs)
import numpy as np
student_values = np.column_stack([student_transformed[name] for name in teacher.output_labels])
summary = validate_coefficients(student_values, test.targets)
dict(zip(teacher.output_labels, summary.rmse))

# %%
# Persist a selected, fully validated model together with provenance metadata.
# student.save("forest_mpg_fix_deterministic.pt", metadata={"teacher_key": "forest_mpg_fix"})
