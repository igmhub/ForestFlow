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
# # 3. Validate a deterministic student against `forest_mpg_fix`
#
# This notebook measures the student’s additional error relative to its frozen
# ForestFlow teacher. Keep this separate from simulation-based validation of the
# teacher itself. Never use the final test split to select architecture, loss,
# or training duration.

# %%
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

from forestflow.emulator.p3d_cinn import P3DEmulator
from forestflow.emulator.distillation.data import (
    TeacherSamplingConfig, generate_teacher_dataset, load_teacher_dataset,
)
from forestflow.emulator.distillation.inference import DeterministicArinyoEmulator
from forestflow.emulator.distillation.validation import validate_coefficients

ARTIFACTS = Path("notebooks/deterministic_distillation/artifacts")
teacher = P3DEmulator(key="forest_mpg_fix")
student = DeterministicArinyoEmulator.load(ARTIFACTS / "forest_mpg_fix_student.pt")
with (ARTIFACTS / "input_splits.json").open() as stream:
    test_inputs = json.load(stream)["test"]

# %% [markdown]
# ## Use independent, better-converged teacher references
#
# The training labels may share stochastic teacher noise with validation labels.
# Regenerate test references with a different seed and a realization count shown
# by the convergence study to be adequate. Do not reuse this result for model
# selection after inspecting it.

# %%
reference_sampling = TeacherSamplingConfig(
    realizations=4096,
    seed=9001,
    sampler="sobol_antithetic",
    statistic="mean",
    draw_policy="nested",
    replicates=2,
)
reference = generate_teacher_dataset(teacher, test_inputs, reference_sampling)
student_transformed = student.predict_transformed(test_inputs)
student_values = np.column_stack([student_transformed[name] for name in teacher.output_labels])
summary = validate_coefficients(student_values, reference.targets)

for index, name in enumerate(teacher.output_labels):
    print(f"{name:8s} bias={summary.bias[index]:+.3e}, RMSE={summary.rmse[index]:.3e}, p99(|error|)={summary.percentile_99_abs[index]:.3e}")

# %%
fig, axis = plt.subplots(figsize=(8, 3.5))
axis.bar(teacher.output_labels, summary.rmse)
axis.set_ylabel("transformed-space RMSE")
axis.set_title("Student error relative to independent ForestFlow references")
fig.tight_layout()

# %% [markdown]
# ## Check physical coefficients and observable impact
#
# The inverse transform is shared with ForestFlow, so transformed errors need to
# be interpreted after that inverse mapping. For P1D/P3D, supply a fixture that
# contains the same cosmology, redshift, linear theory and grids for each input.
# A compressed conditioning vector alone does not contain enough information to
# reconstruct arbitrary linear theory. The template below makes this requirement
# explicit.

# %%
from forestflow.emulator.distillation.validation import validate_observable

def p1d_observable(arinyo_coefficients, fixture):
    """Replace with a fixed ArinyoModel P1D evaluation for this fixture."""
    return fixture["model"].P1D_Mpc(
        fixture["linear"], fixture["z"], fixture["k_iMpc"], arinyo_coefficients
    )

# Example structure, to populate from matched archive/cosmology records:
# fixtures = [{"emu_params": test_inputs[0], "model": arinyo_model,
#              "linear": linear_theory, "z": 3.0, "k_iMpc": k_grid}]
# p1d_summary = validate_observable(student, teacher, fixtures, p1d_observable)
# print(p1d_summary)

# %% [markdown]
# A production acceptance report should show residuals by redshift and k, plus
# RMS, p99, and maxima over fixed grids. It should separately report teacher
# reference uncertainty, test coverage, runtime, and any boundary failures.

# %% [markdown]
#

# %% [markdown]
#
