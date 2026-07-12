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
#
# Before training the emulators, we will transform the input parameters using the square root of the Fisher matrix in standarized coordinates. See the explanation below

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
# ## Fisher-weighted input parameterization
#
# The goal is to transform the emulator input parameters so that distances in parameter space reflect their impact on the predicted observable (e.g. $P_{\rm 3D}$ or $P_{\rm 1D}$), rather than their raw numerical values.
#
# 1. **Standardize the input and output parameters**
#
# $$
# \boldsymbol{\theta}' = D^{-1}(\boldsymbol{\theta}-\boldsymbol{\mu}),
# $$
#
# where $D$ contains the parameter standard deviations (or another appropriate scaling).
#

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

# %% [markdown]
# Standarize and modify input data

# %%
from forestflow.set_training import Transf_data
transf_data = Transf_data(emu_data)

# %%
stand_input_par = transf_data.transf_stand(
    emu_data["input_par"], type_stand="input", direct=True
)

fig, ax = plt.subplots(3, 2, figsize=(10, 8), sharex=True, sharey=True)
ax = ax.flatten()
for ii, par in enumerate(stand_input_par):
    ax[ii].hist(stand_input_par[par], bins=20)
    ax[ii].set_title(par)
plt.tight_layout()

# %%
inv_input_par = transf_data.transf_stand(
    stand_input_par, type_stand="input", direct=False
)

fig, ax = plt.subplots(3, 2, figsize=(10, 8), sharex=False, sharey=False)
ax = ax.flatten()
for ii, par in enumerate(stand_input_par):
    ax[ii].hist(emu_data["input_par"][par], bins=20)
    ax[ii].hist(inv_input_par[par], bins=20, alpha=0.5)
    ax[ii].set_title(par)
plt.tight_layout()

# %%
stand_output_par = transf_data.transf_stand(
    emu_data["output_par"], type_stand="output", direct=True
)

fig, ax = plt.subplots(4, 2, figsize=(10, 10), sharex=True, sharey=True)
ax = ax.flatten()
for ii, par in enumerate(stand_output_par):
    ax[ii].hist(stand_output_par[par], bins=20)
    ax[ii].set_title(par)
plt.tight_layout()

# %%
inv_output_par = transf_data.transf_stand(
    stand_output_par, type_stand="output", direct=False
)

fig, ax = plt.subplots(4, 2, figsize=(10, 8), sharex=False, sharey=False)
ax = ax.flatten()
for ii, par in enumerate(stand_output_par):
    ax[ii].hist(emu_data["output_par"][par], bins=20)
    ax[ii].hist(inv_output_par[par], bins=20, alpha=0.5)
    ax[ii].set_title(par)
plt.tight_layout()

# %% [markdown]
#
# 2. **Compute the Fisher matrix**
#
# $$
# F_{ij} =
# \frac{\partial P}{\partial\theta_i}^{\rm T}
# C^{-1}
# \frac{\partial P}{\partial\theta_j},
# $$
#
# where $P$ is the observable and $C$ is its covariance.
#
# We compute the Fisher matrix for the output parameters, and we will evaluate the Fisher matrix for the mpg-central simulation at z=3
#

# %% [markdown]
# #### First, compute covariance for P1D and P3D

# %% [markdown]
# Get output parameters and set Arinyo model

# %%
sim_mpg_central = Archive3D.get_testing_data("mpg_central")

ztar = 3.0
for ii, sim in enumerate(sim_mpg_central):
    if sim["z"] == ztar:
        ind_z3 = ii

sim = sim_mpg_central[ind_z3]

pars_model = {}
pars_model["z"] = sim["z"]
pars_model["Arinyo"] = {}
for par in emu_data["output_par"]:
    pars_model["Arinyo"][par] = sim["Arinyo_min"][par]

# set Arinyo model
cosmo_params_dict = {}
for par in sim["cosmo_params"]:
    if par != "omk":
        cosmo_params_dict[par] = sim["cosmo_params"][par]
    else:
        cosmo_params_dict[par] = 0.0

fid_cosmo = cosmology.Cosmology(cosmo_params_dict=cosmo_params_dict)
model_Arinyo = ArinyoModel(fid_cosmo)

# %% [markdown]
# Compute covariance matrices

# %%
from forestflow.play_with_power import compute_arinyo_power

# it takes 30 s

# Assuming a Gaussian box of twice the size of our simulations, L=67.5 Mpc
# In reality, we have f&p with 3 axes
Lbox_Mpc = 150

noise = {"n_noise": 10000, "keep_all_noise": False, "Lbox_Mpc": Lbox_Mpc}
power = compute_arinyo_power(pars_model, model_Arinyo, noise=noise)

# %%
pars_model["kpar_Mpc"] = power["model_kpar_Mpc"]
pars_model["kper_Mpc"] = power["model_kper_Mpc"]
pars_model["P3D_Mpc"] = power["ari_P3D_Mpc"]
pars_model["std_P3D_Mpc"] = power["ari_std_P3D_Mpc"]

pars_model["k1D_Mpc"] = power["model_k1d_Mpc"]
pars_model["P1D_Mpc"] = power["ari_P1D_Mpc"]
pars_model["std_P1D_Mpc"] = power["ari_std_P1D_Mpc"]

# %% [markdown]
# Plot diag of 3D cov

# %%
from matplotlib.colors import LogNorm

k = np.sqrt(power["model_kpar_Mpc"]**2 + power["model_kper_Mpc"]**2)
mu = power["model_kpar_Mpc"]/k

mu_coord = False

if mu_coord:
    xplot = k
    yplot = mu
else:
    xplot = power["model_kpar_Mpc"]
    yplot = power["model_kper_Mpc"]

plt.pcolormesh(
    xplot,
    yplot,
    power["ari_std_P3D_Mpc"],
    shading="auto",
    norm=LogNorm(),
)
plt.colorbar()

# %% [markdown]
# Plot diag of 1D cov

# %%
plt.loglog(
    power["model_k1d_Mpc"],
    power["ari_std_P1D_Mpc"],
)

# %% [markdown]
# Relative difference of Arinyo to Kaiser

# %%
k = np.sqrt(power["model_kpar_Mpc"]**2 + power["model_kper_Mpc"]**2)
mu = power["model_kpar_Mpc"]/k

mu_coord = False

if mu_coord:
    xplot = k
    yplot = mu
else:
    xplot = power["model_kpar_Mpc"]
    yplot = power["model_kper_Mpc"]

plt.pcolormesh(
    xplot,
    yplot,
    power["ari_P3D_Mpc"]/power["kai_P3D_Mpc"]-1,
    shading="auto",
    # norm=LogNorm(),
)
plt.colorbar()

# %% [markdown]
# Second, compute the derivatives

# %%
pars_model.keys()

# %%
from forestflow.play_with_power import compute_arinyo_derivatives

der_data = compute_arinyo_derivatives(transf_data, pars_model, model_Arinyo)

# %%
pars_model["P3D_der"] = der_data["P3D_der"]
pars_model["P1D_der"] = der_data["P1D_der"]

# %% [markdown]
# Plot 1D derivatives

# %%
fig, ax = plt.subplots(len(pars_model["Arinyo"]), sharex=True, figsize=(8, 20))

for jj, par in enumerate(pars_model["Arinyo"]):
    if par == "beta":
        continue
    ax[jj].plot(pars_model["k1D_Mpc"], der_data["P1D_der"][par], label=par)
    ax[jj].legend()
plt.xscale("log")

# %% [markdown]
# Plot 3D derivatives

# %%
k = np.sqrt(power["model_kpar_Mpc"]**2 + power["model_kper_Mpc"]**2)
mu = power["model_kpar_Mpc"]/k

mu_coord = False

if mu_coord:
    xplot = k
    yplot = mu
else:
    xplot = power["model_kpar_Mpc"]
    yplot = power["model_kper_Mpc"]

plt.pcolormesh(
    xplot,
    yplot,
    power["ari_P3D_Mpc"]/power["kai_P3D_Mpc"]-1,
    shading="auto",
    # norm=LogNorm(),
)
plt.colorbar()

# %%
from matplotlib.colors import LogNorm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import SymLogNorm


mu_coord = False
k = np.sqrt(power["model_kpar_Mpc"]**2 + power["model_kper_Mpc"]**2)
mu = power["model_kpar_Mpc"]/k

if mu_coord:
    xplot = k
    yplot = mu
else:
    xplot = power["model_kpar_Mpc"]
    yplot = power["model_kper_Mpc"]

fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(8, 8))
ax = ax.reshape(-1)

for jj, par in enumerate(pars_model["Arinyo"]):

    if par == "beta":
        continue

    vmax = np.nanmax(np.abs(der_data["P3D_der"][par]))
    # norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    norm = SymLogNorm(
        linthresh=1e-3,  # linear around zero
        linscale=1,
        vmin=-vmax,
        vmax=vmax,
        base=10,
    )

    ax[jj].pcolormesh(
        xplot,
        yplot,
        der_data["P3D_der"][par],
        shading="auto",
        # norm=LogNorm(),
        cmap="RdBu_r",
        norm=norm,
    )
    ax[jj].set_title(par)


# %% [markdown]
# Fisher matrix combining derivatives and covariance

# %%
from forestflow.play_with_power import compute_fisher

fisher = compute_fisher(pars_model)

# %%
arr_fisher = np.zeros((len(fisher), len(fisher)))

for ii, key1 in enumerate(fisher):
    for jj, key2 in enumerate(fisher):
        arr_fisher[ii, jj] = fisher[key1][key2]

# %%
from matplotlib.colors import LogNorm

labs = list(fisher.keys())

fig, ax = plt.subplots()

vmax = np.nanmax(np.abs(arr_fisher))
norm = SymLogNorm(
    linthresh=1e3,   # linear region around zero
    linscale=1,
    vmin=-vmax,
    vmax=vmax,
    base=10,
)

im = ax.imshow(
    arr_fisher,
    origin="lower",
    cmap="RdBu_r",
    norm=norm,
)

ax.set_xticks(np.arange(len(labs)))
ax.set_yticks(np.arange(len(labs)))

ax.set_xticklabels(labs, rotation=45, ha="right")
ax.set_yticklabels(labs)

ax.set_aspect("equal")

plt.colorbar(im)
plt.tight_layout()

# %% [markdown]
#
# 3. **Transform the parameters using the square root of the Fisher matrix**
#
# $$
# \tilde{\boldsymbol{\theta}} = L\,\boldsymbol{\theta}',
# \qquad
# L^{\rm T}L = F',
# $$
#
# where $F' = D^{\rm T}FD$ is the Fisher matrix in the standardized coordinates, which is how we compute it.
#
# The transformed coordinates satisfy
#
# $$
# \|\Delta\tilde{\boldsymbol{\theta}}\|^2
# =
# \Delta\boldsymbol{\theta}^{\rm T}
# F
# \Delta\boldsymbol{\theta},
# $$
#
# so Euclidean distances correspond to differences in the predicted observable. Directions that strongly affect the prediction are stretched, while insensitive directions are compressed.
#
# This approach leaves the normalizing flow unchanged and instead modifies the geometry of the input space. It provides a physically motivated metric for interpolation and may improve emulator performance by making the mapping from inputs to observables more isotropic.

# %% [markdown]
# Set withening

# %%
transf_data.set_whitening(fisher, type_stand="output")

# %% [markdown]
# Check it works both ways

# %%
tfw_params = transf_data.transf_stand_white(
    emu_data["output_par"], direct=True, type_stand="output"
)

inv_tfw_params = transf_data.transf_stand_white(
    tfw_params, type_stand="output", direct=False
)

fig, ax = plt.subplots(4, 2, figsize=(10, 10), sharex=True, sharey=True)
ax = ax.flatten()
for ii, par in enumerate(inv_tfw_params):
    ax[ii].hist(emu_data["output_par"][par], bins=20)
    ax[ii].hist(inv_tfw_params[par], bins=20, alpha=0.5)
    ax[ii].set_title(par)
plt.tight_layout()

# %% [markdown]
# Set global norm

# %%
tfw_params = transf_data.transf_stand_white(
    emu_data["output_par"], direct=True, type_stand="output"
)
transf_data.set_global_norm(tfw_params, type_stand="output")

# %%
tfwn_params = transf_data.transf_stand_white_norm(
    emu_data["output_par"], type_stand="output", direct=True
)

tf_params = transf_data.transf_stand(
    emu_data["output_par"], type_stand="output", direct=True
)

fig, ax = plt.subplots(4, 2, figsize=(10, 10), sharex=True, sharey=True)
ax = ax.flatten()
for ii, par in enumerate(tfwn_params):
    ax[ii].hist(tfwn_params[par], bins=20)
    ax[ii].hist(tf_params[par], bins=20, alpha=0.5)
    ax[ii].set_title(par)
plt.tight_layout()

# %% [markdown]
# Check it works both ways

# %%
tfwn_params = transf_data.transf_stand_white_norm(
    emu_data["output_par"], direct=True, type_stand="output"
)

inv_tfwn_params = transf_data.transf_stand_white_norm(
    tfwn_params, type_stand="output", direct=False
)

fig, ax = plt.subplots(4, 2, figsize=(10, 10), sharex=True, sharey=True)
ax = ax.flatten()
for ii, par in enumerate(inv_tfwn_params):
    ax[ii].hist(emu_data["output_par"][par], bins=20)
    ax[ii].hist(inv_tfwn_params[par], bins=20, alpha=0.5)
    ax[ii].set_title(par)
plt.tight_layout()

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
save_path = os.path.join(
    os.path.dirname(forestflow.__path__[0]), "data", "emulator_models", "test"
)
emulator = P3DEmulator(
    training_data=input_training,
    train=True,
    nLayers_inn=5,
    # nepochs=1250,
    nepochs=100,
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
