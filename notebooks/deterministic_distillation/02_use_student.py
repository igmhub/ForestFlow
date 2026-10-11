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
# # 2. Use the deterministic student emulator
#
# The student has the same conditioning inputs and physical Arinyo-coefficient
# output as ForestFlow, but no latent draws. Internally it normalizes inputs with
# the teacher transform, predicts transformed coefficients with one MLP forward
# pass, then uses that same transform's inverse.

# %%
from pathlib import Path
import time
import numpy as np

from forestflow.emulator.p3d_cinn import P3DEmulator
from forestflow.emulator.distillation.inference import DeterministicArinyoEmulator

ARTIFACTS = Path("notebooks/deterministic_distillation/artifacts")
student = DeterministicArinyoEmulator.load(ARTIFACTS / "forest_mpg_fix_student.pt")
teacher = P3DEmulator(key="forest_mpg_fix")
print(student.input_labels, student.output_labels)

# %% [markdown]
# Use physically consistent inputs inside the reviewed training domain. A model
# prediction outside that domain is an extrapolation, just as it is for the
# teacher. The first example is a single input and therefore returns scalars.

# %%
input_at_z = {
    "Delta2_p": 0.35,
    "n_p": -2.30,
    "mF": 0.23,
    "sigT_Mpc": 0.10,
    "gamma": 1.20,
    "kF_Mpc": 14.0,
}
transformed = student.predict_transformed(input_at_z)
physical = student.evaluate(input_at_z)
print("transformed:", transformed)
print("physical:", physical)

# %% [markdown]
# Batched calls return one value per input. Use the same input ordering and
# batch shape conventions as the ForestFlow teacher.

# %%
inputs_by_redshift = [input_at_z, {**input_at_z, "mF": 0.25}]
batch_coefficients = student.evaluate(inputs_by_redshift)
{name: np.asarray(values).shape for name, values in batch_coefficients.items()}

# %% [markdown]
# Compare latency fairly: warm both paths, include Python and transformations,
# and state the teacher realization count. The student only replaces the flow
# coefficient prediction; P1D projection and likelihood costs remain.

# %%
teacher_options = dict(Nrealizations=4096, seed=0, sampler="sobol_antithetic", statistic="mean", aggregation_space="transformed", draw_policy="nested")
student.evaluate(inputs_by_redshift)
teacher.evaluate(inputs_by_redshift, **teacher_options)

repetitions = 20
start = time.perf_counter()
for _ in range(repetitions):
    student.evaluate(inputs_by_redshift)
student_seconds = (time.perf_counter() - start) / repetitions

start = time.perf_counter()
for _ in range(repetitions):
    teacher.evaluate(inputs_by_redshift, **teacher_options)
teacher_seconds = (time.perf_counter() - start) / repetitions

print(f"student: {student_seconds:.5f} s; teacher: {teacher_seconds:.5f} s; coefficient speedup: {teacher_seconds / student_seconds:.1f}x")

# %%
