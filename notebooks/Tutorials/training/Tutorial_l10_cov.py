# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: lace
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Train full and l1O emulators

# %%
# %load_ext autoreload
# %autoreload 2

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

import forestflow
from forestflow.P3D_cINN import P3DEmulator

from forestflow.model_p3d_arinyo import ArinyoModel
from lace.cosmo import cosmology

from forestflow.set_training import Transf_data

# %% [markdown]
# ## Training data
#
# Get it and transform in, check out Tutorial_cook_input

# %%
# load training data
from forestflow.archive import GadgetArchive3D
Archive3D = GadgetArchive3D(addcentral=True)

# %% [markdown]
# #### Get data for training the emulator
#
# - input_par: cosmology and IGM
# - other_par: z, As, ns
# - output_par: Arinyo

# %%
from forestflow.set_training import get_training_data
zmax = 4.1 # improves the performance, the results of the Arinyo fit are noisy at z>4 (?!)
emu_data = get_training_data(Archive3D.training_data, zmax=zmax)

# %%
mpg_central = Archive3D.get_testing_data("mpg_central")

ztar = 3.0
for ii, sim in enumerate(mpg_central):
    if sim["z"] == ztar:
        ind_z3 = ii

mpg_central_z3 = mpg_central[ind_z3]

# %% [markdown]
# Standarize and modify input data

# %%
name_emu = "test"

# %%


save_file = os.path.join(
    os.path.dirname(forestflow.__path__[0]),
    "data",
    "emulator_models",
    name_emu + "_transf.npy",
)
transf_data = Transf_data(
    dict_all_params=emu_data, sim_model=mpg_central_z3, save_file=save_file
)

# %%
transf_file = os.path.join(
    os.path.dirname(forestflow.__path__[0]),
    "data",
    "emulator_models",
    name_emu + "_transf.npy",
)
transf_data = Transf_data(preload_file=transf_file)

# %% [markdown]
# Input (cosmo IGM)

# %%
ts_input = transf_data.transf_stand(
    emu_data["input_par"], type_stand="input", direct=True
)

fig, ax = plt.subplots(3, 2, figsize=(10, 8), sharex=True, sharey=True)
ax = ax.flatten()
for ii, par in enumerate(ts_input):
    ax[ii].hist(emu_data["input_par"][par], bins=20)
    ax[ii].hist(ts_input[par], bins=20, alpha=0.5)
    ax[ii].set_title(par)
plt.tight_layout()
ax[-1].set_xlim(-2, 2)

# %% [markdown]
# Output

# %%
ts_output = transf_data.transf_stand(
    emu_data["output_par"], type_stand="output", direct=True
)

fig, ax = plt.subplots(4, 2, figsize=(10, 8), sharex=True, sharey=True)
ax = ax.flatten()
for ii, par in enumerate(ts_output):
    ax[ii].hist(emu_data["output_par"][par], bins=20)
    ax[ii].hist(ts_output[par], bins=20, alpha=0.5)
    ax[ii].set_title(par)
plt.tight_layout()
ax[-1].set_xlim(-2, 2)

# %% [markdown]
# ## Train emulator
#
# Datasets:
# - input ts_input: cosmology + IGM physics. Tranform, standarize
# - ts_output: Arinyo parameters. Tranform, standarize
#

# %%
nepochs = 10 # 1000 better choice, 1 so it runs fast
use_val_set = True # use validation sample

input_training = {}
input_training["input_par"] = ts_input
input_training["output_par"] = ts_output


save_path = os.path.join(
    os.path.dirname(forestflow.__path__[0]), "data", "emulator_models", name_emu
)

emulator = P3DEmulator(
    training_data=input_training,
    train=True,
    nLayers_inn=6,
    nepochs=nepochs,
    batch_size=8,
    lr=1e-3,
    dims_int=12,
    use_val_set=use_val_set,
    save_path=save_path,
)

# %%
n = 0

plt.plot(-np.array(emulator.loss_arr)[n:])
plt.plot(-np.array(emulator.val_loss_arr)[n:])

# %% [markdown]
#

# %% [markdown]
# ## Load emulator
#
# Here to directly load the emulator

# %%
# name_emu = "test" # new trained above
name_emu = "forest_mpg" # default

emulator = P3DEmulator(key=name_emu)

# %% [markdown]
# ### Check precision for mpg-central (test set)

# %%

par_ari = {}
par_emu = {}
for par in emulator.output_labels:
    par_ari[par] = np.zeros(len(mpg_central))
    par_emu[par] = np.zeros(len(mpg_central))

zz = []
for ii, sim in enumerate(mpg_central):
    zz.append(mpg_central[ii]["z"])

    in_emu = {}
    for par in emulator.input_labels:
        in_emu[par] = sim[par]
    out_emu = emulator.evaluate(in_emu)

    for par in emulator.output_labels:
        par_emu[par][ii] = out_emu[par]
        par_ari[par][ii] = sim["Arinyo_min"][par]

zz = np.array(zz)

# %%
fig, ax = plt.subplots(4, 2, figsize=(10, 8), sharex=True)
ax = ax.flatten()

for ii, par in enumerate(emulator.output_labels):
    col = "C" + str(ii)
    ax[ii].plot(zz, par_ari[par], col, label=par)
    ax[ii].plot(zz, par_emu[par], col + "--")

    ax[ii].legend()

# %%
from forestflow.play_with_power import get_sim_power

# %%
cosmo_params_dict = mpg_central_z3["cosmo_params"]
fid_cosmo = cosmology.Cosmology(cosmo_params_dict=cosmo_params_dict)
model_Arinyo = ArinyoModel(fid_cosmo)


# %%
def check_p1d(emulator, Nrealizations=3000):
    ii0 = 0
    for ii in range(2, 11):
        sim = mpg_central[ii]
        print(ii, sim["z"])
        print()

        power_sim = get_sim_power(sim)
        x = power_sim["sim_k1d_Mpc"]
        p1d_data = power_sim["sim_p1d_Mpc"]

        par_ari = {}
        for par in emulator.output_labels:
            par_ari[par] = sim["Arinyo_min"][par]

        p1d_fit = model_Arinyo.P1D_Mpc(sim["z"], power_sim["sim_k1d_Mpc"], par_ari)

        in_emu = {}
        for par in emulator.input_labels:
            in_emu[par] = sim[par]
        out_emu = emulator.evaluate(in_emu, Nrealizations=Nrealizations)

        p1d_emu = model_Arinyo.P1D_Mpc(sim["z"], power_sim["sim_k1d_Mpc"], out_emu)

        # for par in emulator.output_labels:
        #     print(par, np.round(par_ari[par], 3), np.round(out_emu[par], 3))

        # plt.plot(x, x * p1d_data / np.pi, "C"+str(ii0) + ':')
        # plt.plot(x, x * p1d_fit / np.pi, "C"+str(ii0) + '-')
        # plt.plot(x, x * p1d_emu / np.pi, "C"+str(ii0) + '--'

        # plt.plot(x, p1d_data / p1d_fit - 1, "C"+str(ii0) + ':')
        plt.plot(
            x, p1d_emu / p1d_fit - 1, "C" + str(ii0) + "-", label=np.round(sim["z"], 2)
        )
        ii0 += 1

    plt.ylim(-0.02, 0.02)
    plt.legend()


check_p1d(emulator)

# %%
ii0 = 0
for ii in range(2, 11):
    sim = mpg_central[ii]
    print(ii, sim["z"])
    print()

    power_sim = get_sim_power(sim)
    x = power_sim["sim_k3d_Mpc"]
    p3d_data = power_sim["sim_p3d_Mpc"]
    mu3d = power_sim["sim_mu3d"]

    nk = 20
    nmu = 2
    k3D_compare = np.zeros((nk, nmu))
    k3D_compare[:, 0] = np.linspace(0.1, 5, nk)
    k3D_compare[:, 1] = np.linspace(0.1, 5, nk)
    mu3d_compare = np.zeros((nk, nmu))
    mu3d_compare[:, 0] = 0
    mu3d_compare[:, 1] = 1

    par_ari = {}
    for par in emulator.output_labels:
        par_ari[par] = sim["Arinyo_min"][par]

    p3d_fit = model_Arinyo.P3D_Mpc_k_mu(sim["z"], k3D_compare, mu3d_compare, par_ari)

    in_emu = {}
    for par in emulator.input_labels:
        in_emu[par] = sim[par]
    out_emu = emulator.evaluate(in_emu)

    p3d_emu = model_Arinyo.P3D_Mpc_k_mu(sim["z"], k3D_compare, mu3d_compare, out_emu)

    for par in emulator.output_labels:
        print(par, np.round(par_ari[par], 3), np.round(out_emu[par], 3))

    # mu_range = np.linspace(0, 1, 6)
    # # for imu in range(len(mu_range)-1):
    # for imu in range(1):
    #     _ = (mu3d > mu_range[imu]) & (mu3d <= mu_range[imu + 1])

    #     plt.plot(x[_], p3d_data[_], "C" + str(ii0) + ":")
    #     plt.plot(x[_], p3d_fit[_], "C" + str(ii0) + "-")
    #     plt.plot(x[_], p3d_emu[_], "C" + str(ii0) + "--")

    ls = ["-", "--"]
    for imu in range(2):
        plt.plot(k3D_compare[:, imu], p3d_fit[:, imu]/p3d_emu[:, imu]-1, "C" + str(ii0) + ls[imu])
    ii0 += 1
# plt.xscale("log")
# plt.yscale("log")

# %% [markdown]
# ### Train l1O emulators

# %%
zmax = 4.1 # better performance, the results of the Arinyo fit are noisy at z>4 (?!)

nepochs = 1000 # 1000 better choice, 1 so it runs fast
use_val_set = False # use validation sample

for isim, sim in enumerate(Archive3D.list_sim_cube):
    if isim < 25:
        continue
    print(sim)
    print()

    name_emu = "forest_mpg_l1O_" + str(isim)

    save_file_transf = os.path.join(
        os.path.dirname(forestflow.__path__[0]),
        "data",
        "emulator_models",
        "l1O",
        name_emu + "_transf.npy",
    )
    save_path_emu = os.path.join(
        os.path.dirname(forestflow.__path__[0]), "data", "emulator_models", "l1O", name_emu
    )

    # training data
    emu_data = get_training_data(Archive3D.training_data, zmax=zmax, drop_sim=sim)
    transf_data = Transf_data(
        dict_all_params=emu_data, save_file=save_file_transf, compute_fisher=False
    )
    ts_input = transf_data.transf_stand(
        emu_data["input_par"], type_stand="input", direct=True
    )
    ts_output = transf_data.transf_stand(
        emu_data["output_par"], type_stand="output", direct=True
    )
    input_training = {}
    input_training["input_par"] = ts_input
    input_training["output_par"] = ts_output

    emulator = P3DEmulator(
        training_data=input_training,
        train=True,
        nLayers_inn=6,
        nepochs=nepochs,
        batch_size=8,
        lr=1e-3,
        dims_int=12,
        use_val_set=use_val_set,
        save_path=save_path_emu,
    )

    # break

# %%
plt.plot(-np.array(emulator.loss_arr[25:]))

# %%
