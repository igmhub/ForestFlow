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
transf_data = Transf_data(emu_data, mpg_central_z3)

# %% [markdown]
# ## Train emulator
#
# Datasets:
# - input stand_input_par: cosmology + IGM physics. Tranform, standarize
# - fisher_output_par: Arinyo parameters. Tranform, standarize, Fisher whitening, and global scaling. 
#

# %%
# %load_ext autoreload
# %autoreload 2

import sys
import os
import matplotlib.pyplot as plt
import numpy as np


# %%
# load training data
from forestflow.archive import GadgetArchive3D
Archive3D = GadgetArchive3D(addcentral=True)

# %%
from forestflow.set_training import get_training_data
emu_data = get_training_data(Archive3D.training_data)

# %%
sim_mpg_central = Archive3D.get_testing_data("mpg_central")

ztar = 3.0
for ii, sim in enumerate(sim_mpg_central):
    if sim["z"] == ztar:
        ind_z3 = ii

sim_cen = sim_mpg_central[ind_z3]

# %%
from forestflow.set_training import Transf_data
transf_data = Transf_data(emu_data, sim_cen)

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
input_training = {}
input_training["input_par"] = ts_input
input_training["output_par"] = tswn_output

# %%
import forestflow
from forestflow.P3D_cINN import P3DEmulator

save_path = os.path.join(
    os.path.dirname(forestflow.__path__[0]), "data", "emulator_models", "test"
)

# %%

emulator = P3DEmulator(
    training_data=input_training,
    train=True,
    nLayers_inn=5,
    # nepochs=1250,
    nepochs=2,
    batch_size=16,
    lr=5e-4,
    dims_int=16,
    use_val_set=True,
    # use_val_set=False,
    Nrealizations=10000,
    save_path=save_path,
)

# %% [markdown]
#

# %% [markdown]
# ## Load emulator
#
# Here to directly load the emulator

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
add central to training data?

# %%

# %%

# %%
sim['cosmo_params']

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
