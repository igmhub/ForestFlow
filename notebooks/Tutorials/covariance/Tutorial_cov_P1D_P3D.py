# ---
# jupyter:
#   jupytext:
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
# # Estimate covariance between P1D and P3D
#
# Due to cosmic variance

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
# ### Load model for mpg-central at z=3

# %%
# load training data
from forestflow.archive import GadgetArchive3D
Archive3D = GadgetArchive3D(addcentral=True)

# %%
from forestflow.set_training import get_training_data
emu_data = get_training_data(Archive3D.training_data)

# %% [markdown]
# #### Arinyo model

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

# %%
from forestflow.p1d import p1d_from_p3d, get_sigma

nelem_par = 30
nelem_per = 100
# nrand = 1
nrand = 10000
z = 3.0
vol = 200.0**3

kpar_3d = np.linspace(0.1, 5.0, nelem_par)
kper_3d = np.logspace(-3, 2, nelem_per)
# kper_3d = np.linspace(0.1, 1., nelem3d)
kpar2d_3D, kperp2d_3D = np.meshgrid(kpar_3d, kper_3d, indexing="ij")
kk_3d = np.sqrt(kpar2d_3D**2 + kperp2d_3D**2)
mu_3d = kpar2d_3D / kk_3d


res = p1d_from_p3d(
    kpar_3d,
    model_Arinyo.P3D_Mpc_kpar_kperp,
    z,
    pars_model["Arinyo"],
    vol=vol,
    niter=nrand,
    seed=0,
)

# %%
p3d = res["p3d"]
p1d = res["p1d"]
p3d_noise = res["rea_p3d"]
p1d_noise = res["rea_p1d"]

# %%
for ii in range(100):
    plt.plot(kpar_3d, p1d_noise[ii]/p1d-1)

plt.xscale("log")

# %%
for ii in range(10):
    plt.scatter(kperp2d_3D.reshape(-1), p3d_noise[ii, :, :].reshape(-1)/p3d.reshape(-1), alpha=0.5)

plt.xscale("log")

# %%
nmax = nelem_par * nelem_per + nelem_par

both = np.zeros((nrand, nmax))
for ii in range(nrand):
    both[ii, : nelem_par * nelem_per] = p3d_noise[ii].reshape(-1)
    both[ii, nelem_par * nelem_per:] = p1d_noise[ii]


# %%
cov_both = np.cov(both.T)
cov_both.shape

# %%
diag = np.sqrt(np.diag(cov_both))
corr_both = cov_both / np.outer(diag, diag)

# %%
# kpar3D sim kpar1D
# kper/kpar1D = 0.66
# mu = 0.78

# 1/np.sqrt(1 + 0.66**2)

# %%
sc_all = []

for ii in range(kpar_3d.shape[0]):
# for ii in range(2):
    x = kpar2d_3D.reshape(-1) / kpar_3d[ii]
    y = corr_both[nmax - nelem_par + ii, : nmax - nelem_par]
    # _ = (x > 0.9) & (x < 1.1) & (y > 0.05)
    _ = (y > 0.05)
    col = kperp2d_3D.reshape(-1) / kpar_3d[ii]
    # col2 = kk_3d.reshape(-1)/ kpar_3d[ii]
    # col2 = mu_3d.reshape(-1)
    col2 = kpar2d_3D.reshape(-1)
    sc = plt.scatter(col[_], y[_], c=col2[_], alpha=0.6)
    sc_all.append(sc)

vmin = min(sc.get_array().min() for sc in sc_all)
vmax = max(sc.get_array().max() for sc in sc_all)

norm = plt.Normalize(vmin, vmax)

for sc in sc_all:
    sc.set_norm(norm)

plt.axvline(0.66)
plt.xscale("log")
plt.ylim(0.05, 0.4)
plt.colorbar()

# %%
plt.imshow(corr_both)
plt.colorbar()

# %%
# plt.figure(figsize=(8, 8))

x_edges = kpar2d_3D.reshape(-1)

# kpar2d_3D, kperp2d_3D = np.meshgrid(kpar_3d, kper_3d, indexing="ij")
# kk_3d = np.sqrt(kpar2d_3D**2 + kperp2d_3D**2)
# mu_3d = kpar2d_3D / kk_3d

mat = corr_both[nmax - nelem_par :, : nmax - nelem_par]
ind = np.argsort(x_edges)

fig, ax = plt.subplots(figsize=(8, 6))

fontsize = 18
ticksize = 18
sc = ax.pcolormesh(x_edges[ind], kpar_3d, mat[:, ind], shading="auto", rasterized=True)
ax.set_ylabel(r"$k^\mathrm{1D}_\parallel[\mathrm{Mpc}^{-1}]$", fontsize=fontsize)
ax.set_xlabel(r"$k^\mathrm{3D}_\parallel[\mathrm{Mpc}^{-1}]$", fontsize=fontsize)
ax.tick_params(axis="both", labelsize=ticksize)

cbar = fig.colorbar(sc)
cbar.set_label(r"Correlation $P_\mathrm{3D}$ and $P_\mathrm{1D}$", fontsize=fontsize)
cbar.ax.tick_params(labelsize=ticksize)
plt.tight_layout()
plt.savefig("figs/corr_p1d_p3d.pdf")
plt.savefig("figs/corr_p1d_p3d.png")

# %%
# plt.figure(figsize=(8, 8))


mat = corr_both[nmax - nelem_par :, :nmax - nelem_par]

plt.pcolormesh(mat, shading="auto")
plt.colorbar()

# %% [markdown]
#
