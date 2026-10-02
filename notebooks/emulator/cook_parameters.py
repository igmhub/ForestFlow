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
# # Prepare and inspect parameter transformations
#
# This is a development workflow for the preprocessing used by a cINN, not
# an emulator-training tutorial. It inspects forward/inverse transformations,
# then constructs an optional local Fisher metric for the output coefficients.
# It neither trains a network nor saves a new bundle. Replacing the
# transformations of an existing trained model would invalidate its weights.
#
# A local MP-Gadget archive is required. The Fisher section draws 10,000
# Gaussian realizations and may be slow. This study uses the historical
# `Arinyo_min` fits; the corrected training tutorial explicitly selects
# `Cabayol23_fixp3d` and `arinyo_fixp3d` instead.

# %%
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
import numpy as np



from forestflow.model.arinyo import ArinyoModel
from lace.cosmo import cosmology

# %% [markdown]
# ## Standardization and Fisher-weighted output parameterization
#
# Both inputs and outputs are transformed and standardized. The subsequent
# Fisher weighting acts on the **output Arinyo coefficients**, so distances
# reflect their local effect on P3D and P1D rather than their raw values.
# `Transf_data` applies parameter-specific transformations before scaling;
# the schematic equation below refers to those transformed coordinates.
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
from forestflow.archive.gadget_archive import GadgetArchive3D
Archive3D = GadgetArchive3D(addcentral=True)

# %% [markdown]
# #### Get data for training the emulator
#
# - input_par: cosmology and IGM
# - other_par: z, As, ns
# - output_par: Arinyo

# %%
from forestflow.emulator.training import get_training_data
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
#
# `Transf_data` estimates scaling from the supplied training sample. Its
# default `compute_fisher=False` means that whitening is not initialized here;
# the following cells construct it explicitly. `direct=False` reverses the
# corresponding transformation. The round-trip plots should recover the
# original values, for both input and output dictionaries.

# %%
from forestflow.emulator.training import Transf_data
transf_data = Transf_data(emu_data, mpg_central_z3)

# %% [markdown]
# Transformed and standarized

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
# check that inverse is working

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
# check inverse is working

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
from forestflow.statistics.mock_power import make_arinyo_mock_power

# it takes 30 s

# Illustrative Gaussian box, not an exact model of paired/fixed simulations
# or the averaging of their three sightline axes.
Lbox_Mpc = 150.0

noise = {"n_realizations": 10000, "keep_realizations": False, "Lbox_Mpc": Lbox_Mpc}
power = make_arinyo_mock_power(
    pars_model,
    model_Arinyo,
    noise=noise,
    n_3d=20,
    n_1d=20,
    k_min_1d_iMpc=0.1,
    k_max_1d_iMpc=4.0,
    k_min_3d_iMpc=0.1,
    k_max_3d_iMpc=5.0,
)

# %%
pars_model["k_par_iMpc"] = power["model_k_par_iMpc"]
pars_model["k_perp_iMpc"] = power["model_k_perp_iMpc"]
pars_model["P3D_Mpc"] = power["ari_P3D_Mpc"]
pars_model["std_P3D_Mpc"] = power["ari_std_P3D_Mpc"]

pars_model["k_1d_iMpc"] = power["model_k_1d_iMpc"]
pars_model["P1D_Mpc"] = power["ari_P1D_Mpc"]
pars_model["std_P1D_Mpc"] = power["ari_std_P1D_Mpc"]

# %% [markdown]
# Plot diag of 3D cov

# %%
from matplotlib.colors import LogNorm

k = np.sqrt(power["model_k_par_iMpc"]**2 + power["model_k_perp_iMpc"]**2)
mu = power["model_k_par_iMpc"]/k

mu_coord = False

if mu_coord:
    xplot = k
    yplot = mu
else:
    xplot = power["model_k_par_iMpc"]
    yplot = power["model_k_perp_iMpc"]

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
    power["model_k_1d_iMpc"],
    power["ari_std_P1D_Mpc"],
)

# %% [markdown]
# Relative difference of Arinyo to Kaiser

# %%
k = np.sqrt(power["model_k_par_iMpc"]**2 + power["model_k_perp_iMpc"]**2)
mu = power["model_k_par_iMpc"]/k

mu_coord = False

if mu_coord:
    xplot = k
    yplot = mu
else:
    xplot = power["model_k_par_iMpc"]
    yplot = power["model_k_perp_iMpc"]

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
from forestflow.statistics.fisher import compute_arinyo_derivatives

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
    ax[jj].plot(pars_model["k_1d_iMpc"], der_data["P1D_der"][par], label=par)
    ax[jj].legend()
plt.xscale("log")

# %% [markdown]
# Plot 3D derivatives

# %%
k = np.sqrt(power["model_k_par_iMpc"]**2 + power["model_k_perp_iMpc"]**2)
mu = power["model_k_par_iMpc"]/k

mu_coord = False

if mu_coord:
    xplot = k
    yplot = mu
else:
    xplot = power["model_k_par_iMpc"]
    yplot = power["model_k_perp_iMpc"]

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
from matplotlib.colors import SymLogNorm


mu_coord = False
k = np.sqrt(power["model_k_par_iMpc"]**2 + power["model_k_perp_iMpc"]**2)
mu = power["model_k_par_iMpc"]/k

if mu_coord:
    xplot = k
    yplot = mu
else:
    xplot = power["model_k_par_iMpc"]
    yplot = power["model_k_perp_iMpc"]

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
#
# Derivatives are central finite differences in transformed, standardized
# output coordinates. `compute_fisher` combines P3D and P1D contributions
# using inverse **diagonal** variances. It neglects off-diagonal and P3D–P1D
# covariance; this is a preprocessing metric, not a joint likelihood Fisher
# forecast. See `../covariance/` for finite-volume cross-correlation examples.

# %%
from forestflow.statistics.fisher import compute_fisher

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
# Here $F'$ is computed directly in transformed, standardized output
# coordinates. No extra change-of-coordinates factor should be applied.
#
# The transformed coordinates satisfy
#
# $$
# \|\Delta\tilde{\boldsymbol{\theta}}\|^2
# =
# \Delta\boldsymbol{\theta}'^{\rm T}
# F'
# \Delta\boldsymbol{\theta}',
# $$
#
# so Euclidean distances correspond to differences in the predicted observable. Directions that strongly affect the prediction are stretched, while insensitive directions are compressed.
#
# This changes the geometry of the output space before training. The
# implementation uses a Cholesky square root and therefore requires a
# positive-definite Fisher matrix. It is not empirical covariance whitening
# and does not change the emulator architecture.

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
#
# A final global scale follows whitening. Keep standardization, whitening and
# global normalization in this order; inverse calls undo them in reverse.
# Check the final round trips before using these coordinates for new training.

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

# %%

# %%
