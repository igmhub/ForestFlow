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

# %% [markdown]
# ## Arinyo model from default cosmo and params
#
# For more details about the Arinyo model see Eq. 4.5 from Givans+22 (https://arxiv.org/abs/2205.00962)

# %%
zs = np.array([3]) # set target redshift
cosmo = camb_cosmo.get_cosmology() # set default cosmo
camb_results = camb_cosmo.get_camb_results(cosmo, zs=zs, camb_kmax_Mpc=200) # set default cosmo
arinyo = ArinyoModel(cosmo=cosmo, camb_results=camb_results, zs=zs, camb_kmax_Mpc=200) # set model
arinyo.default_params

# %% [markdown]
# ### Compute P3D & P1D

# %%
nn_k = 500 # number of k bins
nn_mu = 11 # number of mu bins
k = np.logspace(-2, 1, nn_k)
mu = np.linspace(0, 1, nn_mu)
k2d = np.tile(k[:, np.newaxis], nn_mu) # k grid for P3D
mu2d = np.tile(mu[:, np.newaxis], nn_k).T # mu grid for P3D

kpar = np.logspace(-1, np.log10(5), nn_k) # kpar for P1D

plin = arinyo.linP_Mpc(zs[0], k) # get linear power spectrum at target z
p3d = arinyo.P3D_Mpc(zs[0], k2d, mu2d, arinyo.default_params) # get P3D at target z
p1d = arinyo.P1D_Mpc(zs[0], kpar, parameters=arinyo.default_params) # get P1D at target z

# %% [markdown]
# #### Plot P3D

# %%
folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"

fig, ax = plt.subplots(1, figsize=(8,6))
ftsize = 20

kdiff = np.zeros((p3d.shape[1], 2))
for ii in range(p3d.shape[1]):
    if(ii < 10):
        col = 'C'+str(ii)
    else:
        col = 'mediumseagreen'
    lab = r'$\mu=$'+str(np.round(mu[ii], 2))
    # if ii % 2 == 0:
    #     lab = r'$\mu=$'+str(np.round(mu[ii], 2))
    # else:
    #     lab = None
    ax.loglog(k, p3d[:, ii]/plin, color=col, label=lab, lw=2)
    ax.plot(k, p3d[0, ii]/plin[0]+k[:]*0, '--', color=col)
    _ = np.argwhere(np.abs(p3d[:, ii]/plin/(p3d[0, ii]/plin[0])-1) > 0.005)[0,0]
    kdiff[ii, 0] = k[_]
    kdiff[ii, 1] = (p3d[:, ii]/plin)[_]
ax.plot(kdiff[:, 0], kdiff[:, 1], "k:", lw=3)
ax.set_xlabel(r'$k$ [Mpc]', fontsize=ftsize)
ax.set_ylabel(r'$P_F(k, \mu)/P_{\rm L}(k)$', fontsize=ftsize)
ax.legend(loc='lower left', ncol=4, fontsize=ftsize-5)
ax.tick_params(axis="both", which="major", labelsize=ftsize)
plt.tight_layout()
plt.savefig(folder+"arinyo.png")
plt.savefig(folder+"arinyo.pdf")

# %% [markdown]
# #### Plot P1D

# %%
plt.plot(kpar, p1d)
plt.xlabel(r'$k$ [Mpc]')
plt.ylabel(r'$P_{\rm 1D}(k)$')
plt.xscale('log')

# %% [markdown]
# ### Best-fitting Arinyo model to LaCE simulation
#
# The Arinyo model was optimized to reproduce both the P3D and P1D down to 4 Mpc (both)

# %% [markdown]
# Read sims

# %%
path_forestflow= os.path.dirname(forestflow.__path__[0]) + "/"
Archive3D = GadgetArchive3D(
    base_folder=path_forestflow,
    folder_data=path_forestflow+"/data/best_arinyo/",
)

# %%
ind_book = 50 # select a random simulation

zs = np.array([Archive3D.training_data[ind_book]['z']]) 

k3d_Mpc = Archive3D.training_data[ind_book]['k3d_Mpc']
mu3d = Archive3D.training_data[ind_book]['mu3d']
p3d_Mpc = Archive3D.training_data[ind_book]['p3d_Mpc']
Plin = Archive3D.training_data[ind_book]['Plin']

k1d_Mpc = Archive3D.training_data[ind_book]['k_Mpc']
p1d_Mpc = Archive3D.training_data[ind_book]['p1d_Mpc']

arinyo_params = Archive3D.training_data[ind_book]['Arinyo_minin'] # best-fitting Arinyo params
print(zs)
arinyo_params

# %% [markdown]
# #### Plot P3D from simulation

# %%
for ii in range(0, p3d_Mpc.shape[1]):
    if ii % 2 == 0:
        lab = r'$<\mu>=$'+str(np.round(mu3d[-1,ii], 2))
    else:
        lab = None
    plt.loglog(k3d_Mpc[:, ii], p3d_Mpc[:, ii]/Plin[:, ii], label=lab)
plt.xlabel(r'$k$ [Mpc]')
plt.ylabel(r'$P/P_{\rm lin}$')
plt.legend(loc='lower left')

# %% [markdown]
# #### Compare P3D from simulation and best-fitting Arinyo

# %%
nn_k = 200 # number of k bins
k = np.logspace(np.log10(0.08), np.log10(8), nn_k)
mu = np.nanmean(mu3d, axis=0)
nn_mu = mu.shape[0] # number of mu bins
k2d = np.tile(k[:, np.newaxis], nn_mu) # k grid for P3D
mu2d = np.tile(mu[:, np.newaxis], nn_k).T # mu grid for P3D

# k2d = k3d_Mpc
# k = np.nanmean(k2d, axis=1)
# mu2d = mu3d

# kpar = np.logspace(-1, np.log10(5), nn_k) # kpar for P1D

# plin = arinyo.linP_Mpc(zs[0], k) # get linear power spectrum at target z
# p3d = arinyo.P3D_Mpc(zs[0], k2d, mu2d, arinyo.default_params) # get P3D at target z
# p1d = arinyo.P1D_Mpc(zs[0], kpar, parameters=arinyo.default_params) # get P1D at target z

# plin = arinyo.linP_Mpc(zs[0], k) # get linear power spectrum at target z
model_p3d = Archive3D.training_data[ind_book]['model'].P3D_Mpc(zs[0], k2d, mu2d, arinyo_params)
plin = Archive3D.training_data[ind_book]['model'].linP_Mpc(zs[0], k)

# %%
folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"

fig, ax = plt.subplots(1, figsize=(8,6))
ftsize = 20

jj = 0
for ii in range(0, p3d_Mpc.shape[1], 3):
    col = 'C'+str(jj)
        
    lab = r'$\mu=$'+str(np.round(mu[ii], 2))
    mask = k3d_Mpc[:, ii] <= 10
    ax.loglog(k3d_Mpc[mask, ii], p3d_Mpc[mask, ii]/Plin[mask, ii], col+'o:', label=lab)
    ax.plot(k, model_p3d[:, ii]/plin, col+'-', lw=2, alpha=0.8)
    kaiser = arinyo_params["bias"]**2*(1+arinyo_params["beta"]*mu[ii]**2)**2
    ax.plot(k, kaiser+k[:]*0, col+'--', lw=1.5, alpha=0.8)
    jj += 1
ax.set_xlabel(r'$k$ [Mpc]', fontsize=ftsize)
ax.set_ylabel(r'$P_F(k, \mu)/P_{\rm L}(k)$', fontsize=ftsize)
ax.legend(loc='lower left', ncol=3, fontsize=ftsize-2)
ax.tick_params(axis="both", which="major", labelsize=ftsize)
plt.tight_layout()
plt.savefig(folder+"arinyo_sim.png")
plt.savefig(folder+"arinyo_sim.pdf")

# %% [markdown]
# #### Ratio

# %%
mask = k3d_Mpc[:,0] < 5
for ii in range(0, p3d_Mpc.shape[1]):
    lab = r'$<\mu>=$'+str(np.round(np.nanmean(mu3d[:,ii]), 2))
    plt.plot(k3d_Mpc[mask, ii], p3d_Mpc[mask, ii]/model_p3d[mask, ii]-1, label=lab)
plt.plot(k3d_Mpc[mask, 0], k3d_Mpc[mask, 0]*0, 'k--')
plt.xscale('log')
plt.xlabel(r'$k$ [Mpc]')
plt.ylabel(r'$P/P_{\rm lin}$')
plt.legend()

# %% [markdown]
# #### Same for P1D

# %%
model_p1d = Archive3D.training_data[ind_book]['model'].P1D_Mpc(zs, k1d_Mpc, parameters=arinyo_params)

# %%
mask = k1d_Mpc < 4
plt.plot(k1d_Mpc[mask], p1d_Mpc[mask]/model_p1d[mask]-1, '-', label='Sim/Model-1')
plt.plot(k1d_Mpc[mask], k1d_Mpc[mask]*0, 'k--')
plt.xscale('log')
plt.xlabel(r'$k$ [Mpc]')
plt.ylabel(r'$P_{\rm 1D}$')
plt.legend()

# %% [markdown]
# ## Arinyo model from emulator

# %% [markdown]
# #### Train emulator (takes 253 seconds in my laptop)

# %%
p3d_emu = P3DEmulator(
    Archive3D.training_data,
    Archive3D.emu_params,
    nepochs=300,
    lr=0.001,  # 0.005
    batch_size=20,
    step_size=200,
    gamma=0.1,
    weight_decay=0,
    adamw=True,
    nLayers_inn=12,  # 15
    Archive=Archive3D,
    use_chains=False,
    chain_samp=100_000,
    model_path="../data/emulator_models/mpg_hypercube.pt",
)

# %%
_p3d_pred, p3d_cov = p3d_emu.predict_P3D_Mpc(
    sim_label=Archive3D.training_data[ind_book]['sim_label'], 
    z=zs, 
    test_sim=[Archive3D.training_data[ind_book]], 
    return_cov=True
)
p3d_pred = np.zeros((p3d_emu.k_mask.shape[0], p3d_emu.k_mask.shape[1]))
p3d_pred[p3d_emu.k_mask] = _p3d_pred

p1d_pred, p1d_cov = p3d_emu.predict_P1D_Mpc(
    sim_label="mpg_central", 
    z=zs, 
    test_sim=[Archive3D.training_data[ind_book]], 
    return_cov=True
)

# %% [markdown]
# #### Ratio with best-fitting model

# %%
mask = k3d_Mpc[:,0] < 4
for ii in range(0, p3d_Mpc.shape[1]):
    lab = r'$<\mu>=$'+str(np.round(np.nanmean(mu3d[:,ii]), 2))
    plt.plot(k3d_Mpc[mask, ii], p3d_pred[mask, ii]/model_p3d[mask, ii]-1, label=lab)
plt.plot(k3d_Mpc[mask, 0], k3d_Mpc[mask, 0]*0, 'k--')
plt.xscale('log')
plt.xlabel(r'$k$ [Mpc]')
plt.ylabel(r'$P/P_{\rm lin}$')
plt.legend()

# %%
# mask = k1d_Mpc < 4
# plt.plot(k1d_Mpc[mask], p1d_pred[mask]/model_p1d[mask]-1, '-', label='Sim/Model-1')
# plt.plot(k1d_Mpc[mask], k1d_Mpc[mask]*0, 'k--')
# plt.xscale('log')
# plt.xlabel(r'$k$ [Mpc]')
# plt.ylabel(r'$P_{\rm 1D}$')
# plt.legend()

# %%
