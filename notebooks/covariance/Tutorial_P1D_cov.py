# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
# # Compute P1D covariance
#
# Estimate finite-volume Gaussian sample variance for the central MP-Gadget
# snapshot at z=3, using its historical `Arinyo_min` fit. Independent P3D
# fluctuations are integrated into P1D realizations on a common grid.
# This is not observational noise, cINN latent scatter, or the l1O-calibrated
# emulator-error covariance used by inference. No covariance file is saved.
# A local archive and the cosmology dependencies are required.

# %%
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
import numpy as np

from forestflow.archive.gadget_archive import GadgetArchive3D
from forestflow.model.arinyo import ArinyoModel
from forestflow.statistics.mock_power import make_arinyo_mock_power
from lace.cosmo import cosmology

# %% [markdown]
# ## Load data

# %%
Archive3D = GadgetArchive3D(addcentral=True)

# %%

# %%
# get mpg-central at z=3

sim_mpg_central = Archive3D.get_testing_data("mpg_central")

ztar = 3.0
for ii, sim in enumerate(sim_mpg_central):
    if sim["z"] == ztar:
        ind_z3 = ii

# %% [markdown]
# ## Compute P1D with noise
#
# To do so, we add uncorrelated noise at the level of P3D.
#
# The current helper calls `ArinyoModel.P1D_Mpc_Gaussian_noise`. Wavenumbers
# are in inverse Mpc and P1D in Mpc. Retain the 1,000 smaller-box realizations
# for the sample covariance; the larger-box run only needs their scatter.
# The helper's default grids span 0.1–5 inverse Mpc. Both runs use the same
# realization seeds, making their volume-scaling comparison reproducible.

# %%
sim = sim_mpg_central[ind_z3]

pars_model = {"z": sim["z"], "Arinyo": sim["Arinyo_min"]}
fid_cosmo = cosmology.Cosmology(cosmo_params_dict=sim["cosmo_params"])
model_Arinyo = ArinyoModel(fid_cosmo)
noise = {"n_realizations": 1000, "keep_realizations": True, "Lbox_Mpc": 100}
power = make_arinyo_mock_power(pars_model, model_Arinyo, noise=noise)

noise = {"n_realizations": 1000, "keep_realizations": False, "Lbox_Mpc": 1000}
power2 = make_arinyo_mock_power(pars_model, model_Arinyo, noise=noise)
power2.keys()

# %%
k1d = power["model_k_1d_iMpc"]
plt.errorbar(
    k1d,
    k1d * power["ari_P1D_Mpc"],
    k1d * power["ari_std_P1D_Mpc"],
    alpha=0.5
)


plt.errorbar(
    k1d,
    k1d * power2["ari_P1D_Mpc"],
    k1d * power2["ari_std_P1D_Mpc"],
    alpha=0.5
)

# %% [markdown]
# Noise to signal
#
# First inspect the fractional standard deviation, then verify that the
# realization mean approaches the deterministic model. The error bars below
# show realization scatter, not the smaller standard error of the mean.

# %%
plt.plot(
    k1d,
    power["ari_std_P1D_Mpc"]/power["ari_P1D_Mpc"],
)
plt.yscale("log")

# %%

plt.errorbar(
    k1d,
    power["ari_P1D_Mpc_realizations"].mean(axis=0)/power["ari_P1D_Mpc"],
    power["ari_P1D_Mpc_realizations"].std(axis=0)/power["ari_P1D_Mpc"],
    alpha=0.5
)


# %% [markdown]
# ### Scaling with volume
#
# For fixed bins and Gaussian modes, sigma scales as V^(-1/2)=L^(-3/2).
# Multiplying the larger-box scatter by the factor below should reproduce
# the smaller-box result. Survey windows, nonlinear mode coupling and
# paired/fixed initial conditions are not represented by this approximation.

# %%
Lbox_Mpc2 = 1000.
Lbox_Mpc = 100.
fact = (Lbox_Mpc2/Lbox_Mpc)**(3/2)

plt.plot(
    k1d,
    power["ari_std_P1D_Mpc"],
)


plt.plot(
    k1d,
    power2["ari_std_P1D_Mpc"],
)

plt.plot(
    k1d,
    power2["ari_std_P1D_Mpc"]*fact,
    color="orange",
    alpha=0.5,
    ls = "",
    marker="."
)

plt.yscale("log")

# %% [markdown]
# ## Sample covariance and dimensionless correlation
#
# Rows are realizations and columns are k bins, hence `rowvar=False`.
# `np.cov` uses N-1 normalization, whereas the plotted `np.std` uses N;
# their diagonals differ by that small finite-sample correction. Covariance
# has units Mpc squared. Each k_parallel bin integrates a separate set of
# independent P3D cells in this approximation, so the expected P1D covariance
# is diagonal; small off-diagonal entries reflect finite-sample noise.

# %%
cov = np.cov(power["ari_P1D_Mpc_realizations"], rowvar=False)
cov.shape

# %%
# Normalize the covariance to unit diagonal to inspect bin correlations.

corr = np.zeros_like(cov)
for ii in range(cov.shape[0]):
    for jj in range(cov.shape[0]):
        corr[ii, jj] = cov[ii, jj] / np.sqrt(cov[ii, ii] * cov[jj, jj])

# %%
plt.imshow(corr)
plt.colorbar()

# %%
