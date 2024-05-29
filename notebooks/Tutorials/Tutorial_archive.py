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
# # Tutorial Archive
#
# It shows how to read the data and plot P3D

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import os
import sys
import matplotlib.pyplot as plt

import forestflow
from forestflow.archive import GadgetArchive3D
from forestflow.rebin_p3d import p3d_allkmu, get_p3d_modes, p3d_rebin_mu

# %%
path_program = os.path.dirname(forestflow.__path__[0]) + '/'
path_program

# %% [markdown]
# ## LOAD P3D ARCHIVE

# %%
# %%time
folder_lya_data = path_program + "/data/best_arinyo/"

Archive3D = GadgetArchive3D(
    base_folder=path_program[:-1],
    folder_data=folder_lya_data,
    average="both",
)
print(len(Archive3D.training_data))


# %% [markdown]
# #### Extract training data
#
# 30 fixed-and-paired simulations with 5 mean-flux rescalings per snapshot each of 11 snapshots between 2 < z < 4.5 

# %%
training_data = Archive3D.training_data

# %% [markdown]
# #### Extract testing data
#
# There are 6 test simulations:
# - central simulations (mpg_central): simulations centered at the training Latin hypercube simulations
# - seed simulations (mpg_seed): simulations centered at the training Latin hypercube simulations with different initial conditions
# - growth simulation (mpg_growth): Simulation with a different growth rate than the training simulations
# - neutrinos simulation (mpg_neutrinos): Simulations with massive neutrinos
# - running simulations (mpg_running): Simulation with a different running of the spectral index.
# - reionization simulations (mpg_reio): Simulation with a different HeII reionization history

# %%
sim = Archive3D.get_testing_data("mpg_central")

# %% [markdown]
# ### Plot P3D and P1D
#
# We will rebin P3D to reduce noise

# %%
n_mubins = 4
kmax_3d_plot = 4
kmax_1d_plot = 4

sim = Archive3D.training_data[0]

k3d_Mpc = sim['k3d_Mpc']
mu3d = sim['mu3d']
p3d_Mpc = sim['p3d_Mpc']
# get modes in each k-mu bin
kmu_modes = get_p3d_modes(kmax_3d_plot)

mask_3d = k3d_Mpc[:, 0] <= kmax_3d_plot

mask_1d = (sim['k_Mpc'] <= kmax_1d_plot) & (sim['k_Mpc'] > 0)
k1d_Mpc = sim['k_Mpc'][mask_1d]
p1d_Mpc = sim['p1d_Mpc'][mask_1d]

# %%
# apply rebinning
_ = p3d_rebin_mu(k3d_Mpc[mask_3d], mu3d[mask_3d], p3d_Mpc[mask_3d], kmu_modes, n_mubins=n_mubins)
knew, munew, rebin_p3d_sim, mu_bins = _

# normalize P1D
p1d_sim = k1d_Mpc/np.pi * p1d_Mpc

# %%
for ii in range(n_mubins):
    _ = np.isfinite(rebin_p3d_sim[:, ii])
    plt.plot(knew[_, ii], knew[_, ii]**2*rebin_p3d_sim[_, ii])
plt.xscale('log')

# %%
plt.plot(k1d_Mpc, p1d_sim)
plt.xscale('log')

# %%
