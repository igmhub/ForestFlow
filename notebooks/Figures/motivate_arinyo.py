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
# # MotivateArinyo model

# %% [markdown]
# In this notebook we explain how to compute P3D and P1D from a particular Arinyo model

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"


# %%
from lace.cosmo import camb_cosmo
import forestflow
from forestflow.archive import GadgetArchive3D
from forestflow.model_p3d_arinyo import get_linP_interp
from forestflow.model_p3d_arinyo import ArinyoModel
from forestflow.P3D_cINN import P3DEmulator
from forestflow.rebin_p3d import p3d_allkmu, get_p3d_modes, p3d_rebin_mu

# %% [markdown]
# ## Best-fitting Arinyo model to central
#
# The Arinyo model was optimized to reproduce both the P3D and P1D down to 3 Mpc (both)

# %% [markdown]
# Read sims

# %%
path_forestflow= os.path.dirname(forestflow.__path__[0]) + "/"
Archive3D = GadgetArchive3D(
    base_folder=path_forestflow,
    folder_data=path_forestflow+"/data/best_arinyo/",
)

# %%
test_sim = Archive3D.get_testing_data(
    "mpg_central", force_recompute_plin=False
)
z_grid = [d["z"] for d in test_sim]
zs = 3
test_sim_z = [d for d in test_sim if d["z"] == zs][0]

# %% [markdown]
# ### Data from simulation, rebin to 4 mu bins

# %%
n_mubins = 4
kmax = 6
kmax_fit = 5

k3d_Mpc = test_sim_z['k3d_Mpc']
mu3d = test_sim_z['mu3d']
p3d_Mpc = test_sim_z['p3d_Mpc']

kmu_modes = get_p3d_modes(kmax)

mask_3d = k3d_Mpc[:, 0] <= kmax

_ = p3d_rebin_mu(k3d_Mpc[mask_3d], mu3d[mask_3d], p3d_Mpc[mask_3d], kmu_modes, n_mubins=n_mubins)
knew, munew, rebin_p3d, mu_bins = _

# mask_1d = test_sim_z['k_Mpc'] <= kmax
# k1d_Mpc = test_sim_z['k_Mpc'][mask_1d]
# p1d_Mpc = test_sim_z['p1d_Mpc'][mask_1d]

# %% [markdown]
# ### Model, also rebin

# %%
# arinyo_params = test_sim_z['Arinyo_min_q1'] # best-fitting Arinyo params
# arinyo_params = test_sim_z['Arinyo_min_q1_q2'] # best-fitting Arinyo params
arinyo_params = test_sim_z['Arinyo_min'] # best-fitting Arinyo params
# arinyo_params = test_sim_z['Arinyo_minz'] # best-fitting Arinyo params

kaiser_params = arinyo_params.copy()
kaiser_params["q1"] = 0
kaiser_params["q2"] = 0
kaiser_params["kp"] = 10**5

# old
# model_p3d = test_sim_z['model'].P3D_Mpc(zs, k3d_Mpc, mu3d, arinyo_params)
# plin = test_sim_z['model'].linP_Mpc(zs, k3d_Mpc)

_ = p3d_allkmu(test_sim_z['model'], zs, arinyo_params, kmu_modes, nk=np.sum(mask_3d))
model_p3d, plin = _

_ = p3d_allkmu(test_sim_z['model'], zs, kaiser_params, kmu_modes, nk=np.sum(mask_3d))
kaiser_p3d, plin = _

_ = p3d_rebin_mu(k3d_Mpc[mask_3d], mu3d[mask_3d], model_p3d, kmu_modes, n_mubins=n_mubins,)
knew, munew, rebin_model_p3d, mu_bins = _

_ = p3d_rebin_mu(k3d_Mpc[mask_3d], mu3d[mask_3d], kaiser_p3d, kmu_modes, n_mubins=n_mubins)
knew, munew, rebin_kaiser_p3d, mu_bins = _

_ = p3d_rebin_mu(k3d_Mpc[mask_3d], mu3d[mask_3d], plin, kmu_modes, n_mubins=n_mubins)
knew, munew, rebin_plin, mu_bins = _


# %% [markdown]
# ### Plot P3D from simulation

# %%
from forestflow.plots.motivate_model import plot_motivate_model

# %%
folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"
plot_motivate_model(knew, munew, mu_bins, rebin_p3d, rebin_model_p3d, rebin_kaiser_p3d, rebin_plin, folder=folder, kmax_fit=kmax_fit)

# %%
