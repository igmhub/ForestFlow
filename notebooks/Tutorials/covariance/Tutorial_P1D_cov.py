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

# %%
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
import numpy as np

from forestflow.archive import GadgetArchive3D

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
# For more details, see model_p3d_arinyo.P1D_Mpc_Gaussian_noise

# %%
sim = sim_mpg_central[ind_z3]

noise = {"n_noise": 1000, "keep_all_noise": True, "Lbox_Mpc": 100}
power = get_arinyo_power(sim, noise=noise)

noise = {"n_noise": 1000, "keep_all_noise": False, "Lbox_Mpc": 1000}
power2 = get_arinyo_power(sim, noise=noise)
power2.keys()

# %%
k1d = power["model_k1d_Mpc"]
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

# %%
plt.plot(
    k1d,
    power["ari_std_P1D_Mpc"]/power["ari_P1D_Mpc"],
)
plt.yscale("log")

# %%

plt.errorbar(
    k1d,
    power["ari_noise_P1D_Mpc"].mean(axis=0)/power["ari_P1D_Mpc"],
    power["ari_noise_P1D_Mpc"].std(axis=0)/power["ari_P1D_Mpc"],
    alpha=0.5
)


# %% [markdown]
# ### Scaling with volume

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

# %%
cov = np.cov(power["ari_noise_P1D_Mpc"], rowvar=False)
cov.shape

# %%
# uncorrelated noise from cosmic variance

corr = np.zeros_like(cov)
for ii in range(cov.shape[0]):
    for jj in range(cov.shape[0]):
        corr[ii, jj] = cov[ii, jj] / np.sqrt(cov[ii, ii] * cov[jj, jj])

# %%
plt.imshow(corr)
plt.colorbar()

# %%
