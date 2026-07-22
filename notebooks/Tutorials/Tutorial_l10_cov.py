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

# %%

# %%
import h5py

data = {}
data["cosmo"] = {}
cosmo_labs = ["ombh2", "omch2", "ns", "As", "H0"]

file = "/home/jchaves/Proyectos/projects/lya/P3d_lya_ASTRID.hdf5"

with h5py.File(file, "r") as f:

    def load_dataset(name, obj):
        if isinstance(obj, h5py.Dataset):
            data[name] = obj[()]    # or obj[:] for array datasets

    f.visititems(load_dataset)

    data["z"] = f.attrs["z"]
    for par in cosmo_labs:
        if par == "H0":
            par2 = "hubble"
        else:
            par2 = par
        data["cosmo"][par] = f.attrs[par2]

    for att in f.attrs:
        print(att, f.attrs[att])

data["cosmo"]["w"] = -1.0
data["cosmo"]["mnu"] = 0.0
data["cosmo"]["nrun"] = 0.0
data["cosmo"]["omk"] = 0.0
    

# %%
250/0.5

# %%
for ii in range(10):
    lab = "mu = %.2f" % np.nanmean(data["mu"][:, ii])
    kk = data["k_hMpc"][:, ii]
    sig = kk**2*data["p3d_lya_Mpch"][:, ii]
    err = kk**2*data["p3d_lya_std_Mpch"][:, ii] * np.sqrt(2)
    plt.errorbar(kk, sig, err, label=lab)
plt.legend()
plt.xscale("log")
plt.yscale("log")

# %%
mpg_central_z3["mu3d"].shape

# %%
2 * np.pi/65.*768/3

# %%
2 * np.pi/250 * 500/3

# %%
mpg_central_z3["k3d_Mpc"][:,0]

# %%
np.geomspace(0.09, 23, 20)

# %%
np.geomspace(0.025, 21, 40)

# %%
data["k_hMpc"][:,0]

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
zmax = 4.1
emu_data = get_training_data(Archive3D.training_data)

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
from forestflow.set_training import Transf_data

save_file = os.path.join(
    os.path.dirname(forestflow.__path__[0]),
    "data",
    "emulator_models",
    "test_transf.npy",
)
transf_data = Transf_data(
    dict_all_params=emu_data, sim_model=mpg_central_z3, save_file=save_file
)

# %%
transf_file = os.path.join(
    os.path.dirname(forestflow.__path__[0]),
    "data",
    "emulator_models",
    "test_transf.npy",
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
tswn_output = transf_data.transf_stand_white_norm(
    emu_data["output_par"], type_stand="output", direct=True
)


fig, ax = plt.subplots(4, 2, figsize=(10, 8), sharex=True, sharey=True)
ax = ax.flatten()
for ii, par in enumerate(tswn_output):
    ax[ii].hist(emu_data["output_par"][par], bins=20)
    ax[ii].hist(tswn_output[par], bins=20, alpha=0.5)
    ax[ii].set_title(par)
plt.tight_layout()
ax[-1].set_xlim(-2, 2)

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
# - input stand_input_par: cosmology + IGM physics. Tranform, standarize
# - fisher_output_par: Arinyo parameters. Tranform, standarize, Fisher whitening, and global scaling. 
#

# %%
ts_output = transf_data.transf_stand(
    emu_data["output_par"], type_stand="output", direct=True
)

# %%
input_training = {}
input_training["input_par"] = ts_input
# input_training["output_par"] = tswn_output
input_training["output_par"] = ts_output


save_path = os.path.join(
    os.path.dirname(forestflow.__path__[0]), "data", "emulator_models", "test"
)

# %%
emulator = P3DEmulator(
    training_data=input_training,
    train=True,
    nLayers_inn=6,
    nepochs=300,
    batch_size=8,
    lr=10e-4,
    dims_int=12,
    # use_val_set=True,
    use_val_set=False,
    Nrealizations=5000,
    save_path=save_path,
)

# %%
Partial nepoch 325 

Full nepoch 500 -21.05 379s
Full nepoch 300 -19.37 170s

# %%
-21 1000 full
-20 val


batch
-16.8 batch 16
-16.2 batch 32
-15.67 batch 64

lr
-15.96 lr 1e-3
-15.67 lr 5e-4
-15.79 lr 3e-4
-13.85 lr 1e-4

nlayers
-16.55 n 6
-15.96 n 5

ndim
-15.03 n 8
-15.96 n 16
-14.97 n 32

n 6, batch 16, lr 1e-3 -17.37

n 6, batch 64, lr 1e-3, ndim 8 -15.98
n 6, batch 64, lr 1e-3, ndim 12 -16.97
n 6, batch 64, lr 1e-3, ndim 16 -16.55
n 6, batch 64, lr 1e-3, ndim 20 -16.45

n 7, batch 64, lr 1e-3, ndim 12 -16.09


n 6, batch 64, lr 20e-4, ndim 12 -17.1 30s
n 6, batch 64, lr 10e-4, ndim 12 -16.97
n 6, batch 64, lr 5e-4, ndim 12 -16.53

n 6, batch 12, lr 20e-4, ndim 12 -17.84 116s 300 epochs
n 6, batch 16, lr 20e-4, ndim 12 -17.78 142s 350 epochs
n 6, batch 8, lr 40e-4, ndim 12 -17.99 162s 200 epochs
n 6, batch 8, lr 10e-4, ndim 12 -18.33 199s 325 epochs

# %%
Produce plots with precision for validation data and training data

Check precision when:
- output o, t, ts, tsw, tswn
- input o t, ts

# %%
n = 50

plt.plot(-np.array(emulator.loss_arr)[n:])
plt.plot(-np.array(emulator.val_loss_arr)[n:])

# %% [markdown]
#

# %% [markdown]
# Epoch 800/3000, train loss -39.95, val loss -39.31, best -39.33, 225 s

# %% [markdown]
#

# %% [markdown]
# ## Load emulator
#
# Here to directly load the emulator

# %%
load_path = os.path.join(
    os.path.dirname(forestflow.__path__[0]), "data", "emulator_models", "test"
)

transf_file = os.path.join(
    os.path.dirname(forestflow.__path__[0]),
    "data",
    "emulator_models",
    "test_transf.npy",
)

emulator = P3DEmulator(
    Nrealizations=5000, model_path=load_path, transf_file=transf_file
)

# %%
500 everything within 3%
1000 everything within 2.5%, removing 3 outliers 1%

# %%
check_p1d(emulator)
plt.savefig("base_ts_new_500.png")

# %%

par_ari = {}
par_ari2 = {}
par_emu = {}
for par in emulator.output_labels:
    par_ari[par] = np.zeros(len(mpg_central))
    par_ari2[par] = np.zeros(len(mpg_central))
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
        if par not in sim["Arinyo_minz"]:
            continue
        elif par == "bias":
            sign = -1
        else:
            sign = 1
        par_ari2[par][ii] = sign * sim["Arinyo_minz"][par]

zz = np.array(zz)

# %%
fig, ax = plt.subplots(4, 2, figsize=(10, 8), sharex=True)
ax = ax.flatten()

for ii, par in enumerate(emulator.output_labels):
    col = "C" + str(ii)
    ax[ii].plot(zz, par_ari[par], col, label=par)
    ax[ii].plot(zz, par_emu[par], col + "--")
    # ax[ii].plot(zz, par_ari2[par], col + ":")

    ax[ii].legend()

# %%
from forestflow.play_with_power import get_sim_power

# %%
cosmo_params_dict = mpg_central_z3["cosmo_params"]
fid_cosmo = cosmology.Cosmology(cosmo_params_dict=cosmo_params_dict)
model_Arinyo = ArinyoModel(fid_cosmo)


# %%

def check_p1d(emulator):
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
        out_emu = emulator.evaluate(in_emu)

        p1d_emu = model_Arinyo.P1D_Mpc(sim["z"], power_sim["sim_k1d_Mpc"], out_emu)

        # for par in emulator.output_labels:
        #     print(par, np.round(par_ari[par], 3), np.round(out_emu[par], 3))

        # plt.plot(x, x * p1d_data / np.pi, "C"+str(ii0) + ':')
        # plt.plot(x, x * p1d_fit / np.pi, "C"+str(ii0) + '-')
        # plt.plot(x, x * p1d_emu / np.pi, "C"+str(ii0) + '--'
                
        # plt.plot(x, p1d_data / p1d_fit - 1, "C"+str(ii0) + ':')
        plt.plot(x, p1d_emu / p1d_fit - 1, "C"+str(ii0) + '-', label=np.round(sim["z"],2))
        ii0 += 1

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

# %%
p1d_data.shape

# %%
np.linspace(0, 1, 5)

# %%
full_emulator = P3DEmulator(
    model_path=os.path.join(
        os.path.dirname(forestflow.__path__[0]), "data", "emulator_models", "forest_mpg"
    )
)

# %% [markdown]
# ### Train l1O emulators

# %%
save_path = os.path.join(
    os.path.dirname(forestflow.__path__[0]), "data", "emulator_models", "test"
)
emulator = P3DEmulator(
    training_data=Archive3D.training_data,
    emu_input_names=Archive3D.emu_params,
    training_type="Arinyo_min",
    train=True,
    nLayers_inn=5,
    nepochs=1250,
    batch_size=16,
    lr=5e-4,
    dims_int=16,
    use_val_set=True,
    # use_val_set=False,
    Nrealizations=10000,
    save_path=save_path,
)

# %%
# takes forever
4 layers, batch 32 went to -24 for 1000, go longer!
4 layers, batch 16 went to -24.5 for 1000, go longer!
    
# sweet spot
dims_int=16
5 layers, batch 16 went to -30, 1000 is good
# anything better??? stop a little bit longer than when using
# the validation sample
dims_int=32
way worse
dims_int=8
worse

# bad
6 layers, batch 16 went to -24.84, stop

# %%
n = 25

plt.plot(-np.array(emulator.loss_arr)[n:])
plt.plot(-np.array(emulator.val_loss_arr)[n:])

# %%
emulator = P3DEmulator(
    model_path=os.path.join(
        os.path.dirname(forestflow.__path__[0]), "data", "emulator_models", "test"
    )
)

# %%

# %%
len(Archive3D.training_data)

# %%
for isim, sim in enumerate(Archive3D.list_sim_cube):
    if isim != 0:
        continue
    print(sim)
    print()
    save_path = os.path.join(
        os.path.dirname(forestflow.__path__[0]), "data", "emulator_models", "l1O", sim
    )
    emulator = P3DEmulator(
        training_data=Archive3D.training_data,
        emu_input_names=Archive3D.emu_params,
        training_type="Arinyo_min",
        train=True,
        drop_sim=sim,
        # nepochs=4000,
        nepochs=10,
        batch_size=20,
        step_size=200,
        weight_decay=0.01,
        Nrealizations=6000,
        save_path=save_path,
    )
    # break

# %%
plt.plot(np.log(-np.array(emulator.loss_arr)))

# %%
emulator.Arinyo_params

# %% [markdown]
# ## Stop

# %%
p1d_Mpc_sm from evaluating ForestFlow 
for the best-fitting parameters of the 
model, this is in the training data?

apply binning before comparing data?

out = data_for_l10(Archive3D)

# %%
# load data of the simulation we removed
sim_label = "mpg_central"
testing_data = []
for sim in Archive3D.training_data:
    if sim["sim_label"] == sim_label:
        testing_data.append(sim)
# evaluate emulator for this simulation

# store the results from the emulator, 
# the smooth result from Arinyo, and the actual
# result from the simulation



# %%
# Get power
kmax_1d_fit = 4
kmax_1d_plot = kmax_1d_fit + 1
kp_Mpc = 0.7

sim = Archive3D.training_data[0]
mask_1d = (sim['k_Mpc'] <= kmax_1d_plot) & (sim['k_Mpc'] > 0)
k1d_Mpc = sim['k_Mpc'][mask_1d]
p1d_Mpc = sim['p1d_Mpc'][mask_1d]

# %%
cosmo_params_dict = {}
for par in sim['cosmo_params']:
    if par != "omk":
        cosmo_params_dict[par] = sim['cosmo_params'][par]
    else:
        cosmo_params_dict[par] = 0.0

fid_cosmo = cosmology.Cosmology(cosmo_params_dict=cosmo_params_dict)
model_Arinyo = ArinyoModel(fid_cosmo)



# %%
# emulator = P3DEmulator(
#     model_path=os.path.join(
#         os.path.dirname(forestflow.__path__[0]), "data", "emulator_models", "l1O", "mpg_0"
#     )
# )

# %%
# emulator = full_emulator



for sim in testing_data:
    emu_params = {}
    for param in Archive3D.emu_params:
        emu_params[param] = sim[param]
    model3d_coeffs = emulator.predict_Arinyos(emu_params=emu_params)

    new_cosmo_params = {}
    for par in ["As", "ns"]:
        new_cosmo_params[par] = sim["cosmo_params"][par]

    p1d_emu = model_Arinyo.P1D_Mpc(
        sim["z"], k1d_Mpc, model3d_coeffs, new_cosmo_params=new_cosmo_params
    )
    # p1d_emu_l10 = model_Arinyo.P1D_Mpc(
    #     sim["z"], k1d_Mpc, model3d_coeffs, new_cosmo_params=new_cosmo_params
    # )

    p1d_fit = model_Arinyo.P1D_Mpc(
        sim["z"], k1d_Mpc, sim["Arinyo_min"], new_cosmo_params=new_cosmo_params
    )

    _mask_1d = (sim['k_Mpc'] <= kmax_1d_plot) & (sim['k_Mpc'] > 0)
    p1d_sim = sim["p1d_Mpc"][_mask_1d]

    plt.plot(k1d_Mpc, p1d_emu/p1d_fit, label=np.round(sim["z"],2))

    if sim["z"] == 3:
        break

plt.legend()

# %% [markdown]
#

# %%
plt.plot(k1d_Mpc, p1d_emu/p1d_fit)
# plt.plot(k1d_Mpc, p1d_emu_l10/p1d_fit, ":")
plt.plot(k1d_Mpc, p1d_sim/p1d_fit, "--")
plt.axhline(1, color="black", ls=":")
plt.xlabel(r'$k$ [Mpc]')
plt.ylabel(r'$P_{\rm 1D}(k)$')
plt.xscale('log')

# %%
# emulator = full_emulator

model3d_coeffs = emulator.predict_Arinyos(emu_params=emu_params)
model3d_coeffs

# %%
sim["Arinyo_min"]

# %% [markdown]
#
