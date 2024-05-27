# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Errorbars on power spectra

# %%
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# %%
import forestflow
from forestflow.archive import GadgetArchive3D

# %%
path_program = os.path.dirname(forestflow.__path__[0]) + "/"
path_program
folder_lya_data = path_program + "/data/best_arinyo/"
Archive3D = GadgetArchive3D(
    base_folder=path_program[:-1],
    folder_data=folder_lya_data,
    average="both",
)

# %%
Archive3D.data[0].keys()

# %% [markdown]
# ### Get relevant P1D data from the CENTRAL simulation (z=3)

# %%
central_data = []
for d in Archive3D.data:
    if d["sim_label"] == "mpg_central":
        if d["val_scaling"] == 1.0:
            central_data.append(d)
print(len(central_data))
central_plus_z3 = []
central_minus_z3 = []
for d in central_data:
    if d["z"] == 3.0:
        if d["ind_phase"] == 0:
            central_minus_z3.append(d)
        else:
            central_plus_z3.append(d)
print(len(central_plus_z3), len(central_minus_z3))

# %%
central_z3 = {}
k_Mpc = central_plus_z3[0]["k_Mpc"]
sum_p1d = np.zeros_like(k_Mpc)
sum_mf = 0
for axis in range(3):
    mf_p = central_plus_z3[axis]["mF"]
    mf_m = central_minus_z3[axis]["mF"]
    mf = 0.5 * (mf_p + mf_m)
    p1d_p = central_plus_z3[axis]["p1d_Mpc"]
    p1d_m = central_minus_z3[axis]["p1d_Mpc"]
    p1d = (mf_p**2 * p1d_p + mf_m**2 * p1d_m) / mf**2 / 2.0
    central_z3[axis] = {"mF": mf, "p1d_Mpc": p1d}
    # compute average over axes
    sum_mf += mf
    sum_p1d += mf**2 * p1d
# normalize average
sum_mf = sum_mf / 3.0
sum_p1d = sum_p1d / sum_mf**2 / 3.0
central_z3["total"] = {"mF": sum_mf, "p1d_Mpc": sum_p1d}

# %%
for axis in range(3):
    plt.plot(
        k_Mpc,
        central_z3[axis]["p1d_Mpc"] / central_z3["total"]["p1d_Mpc"],
        label=axis,
    )
plt.xlim(0, 3)
plt.legend()

# %% [markdown]
# ### Get relevant P1D data from the SEED simulation (z=3)

# %%
seed_data = []
for d in Archive3D.data:
    if d["sim_label"] == "mpg_seed":
        if d["val_scaling"] == 1.0:
            seed_data.append(d)
print(len(seed_data))
seed_plus_z3 = []
seed_minus_z3 = []
for d in seed_data:
    if d["z"] == 3.0:
        if d["ind_phase"] == 0:
            seed_minus_z3.append(d)
        else:
            seed_plus_z3.append(d)
print(len(seed_plus_z3), len(seed_minus_z3))

# %%
seed_z3 = {}
seed_k_Mpc = seed_plus_z3[0]["k_Mpc"]
sum_p1d = np.zeros_like(k_Mpc)
sum_mf = 0
for axis in range(3):
    mf_p = seed_plus_z3[axis]["mF"]
    mf_m = seed_minus_z3[axis]["mF"]
    mf = 0.5 * (mf_p + mf_m)
    p1d_p = seed_plus_z3[axis]["p1d_Mpc"]
    p1d_m = seed_minus_z3[axis]["p1d_Mpc"]
    p1d = (mf_p**2 * p1d_p + mf_m**2 * p1d_m) / mf**2 / 2.0
    seed_z3[axis] = {"mF": mf, "p1d_Mpc": p1d}
    # compute average over axes
    sum_mf += mf
    sum_p1d += mf**2 * p1d
# normalize average
sum_mf = sum_mf / 3.0
sum_p1d = sum_p1d / sum_mf**2 / 3.0
seed_z3["total"] = {"mF": sum_mf, "p1d_Mpc": sum_p1d}

# %% [markdown]
# ### Compute best estimate of power (combining CENTRAL and SEED)

# %%
central_mf = central_z3["total"]["mF"]
seed_mf = seed_z3["total"]["mF"]
total_mf = 0.5 * (central_mf + seed_mf)
total_p1d = (
    0.5
    * (
        central_mf**2 * central_z3["total"]["p1d_Mpc"]
        + seed_mf**2 * seed_z3["total"]["p1d_Mpc"]
    )
    / total_mf**2
)

# %%
plt.plot(
    k_Mpc, central_z3["total"]["p1d_Mpc"] / total_p1d, label="central combined"
)
plt.plot(k_Mpc, seed_z3["total"]["p1d_Mpc"] / total_p1d, label="seed combined")
plt.xlim(0, 3)
plt.legend()

# %% [markdown]
# ### Compute scatter around best estimate (from axes and sims)

# %%
var_p1d = np.zeros_like(k_Mpc)
for axis in range(3):
    for sim in [central_z3, seed_z3]:
        var_p1d += (sim[axis]["p1d_Mpc"] - total_p1d) ** 2
rms_p1d = np.sqrt(var_p1d / 5)

# %%
for axis in range(3):
    plt.plot(
        k_Mpc,
        central_z3[axis]["p1d_Mpc"] / total_p1d,
        alpha=0.3,
        label="central " + str(axis),
    )
    plt.plot(
        k_Mpc,
        seed_z3[axis]["p1d_Mpc"] / total_p1d,
        alpha=0.3,
        label="seed " + str(axis),
    )
    plt.plot(k_Mpc, (total_p1d + rms_p1d) / total_p1d, ls=":", color="gray")
    plt.plot(k_Mpc, (total_p1d - rms_p1d) / total_p1d, ls=":", color="gray")
plt.plot(k_Mpc, total_p1d / total_p1d, ls="--", color="gray")
plt.xlim(0, 5)
plt.xlabel("kp [1/Mpc]")
plt.ylabel("P1D / <P1D>")
plt.legend()

# %%

# %%
