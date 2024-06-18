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
# # Study precision of model for P3D and P1D

# %%
# %load_ext autoreload
# %autoreload 2

import sys
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

from forestflow.archive import GadgetArchive3D
from forestflow.plots_v0 import plot_test_p3d
from forestflow.P3D_cINN import P3DEmulator
from forestflow.model_p3d_arinyo import ArinyoModel
from forestflow import model_p3d_arinyo
from forestflow.utils import transform_arinyo_params, params_numpy2dict


# %%
def ls_level(folder, nlevels):
    for ii in range(nlevels):
        folder = os.path.dirname(folder)
    folder += "/"
    return folder


path_program = ls_level(os.getcwd(), 2)
print(path_program)
sys.path.append(path_program)

# %% [markdown]
# ## LOAD P3D ARCHIVE

# %%
# %%time
folder_lya_data = path_program + "/data/best_arinyo/"
# folder_interp = path_program+"/data/plin_interp/"

Archive3D = GadgetArchive3D(
    base_folder=path_program[:-1],
    folder_data=folder_lya_data,
    force_recompute_plin=False,
    average="both",
)
print(len(Archive3D.training_data))


# %% [markdown]
# ## LOAD SIMULATIONS

# %%
sim_label = "mpg_central"
central = Archive3D.get_testing_data(
    sim_label, force_recompute_plin=False
)

sim_label = "mpg_seed"
seed = Archive3D.get_testing_data(
    sim_label, force_recompute_plin=False
)

# get average of both
list_merge = []
zlist = []
par_merge = ["mF", "T0", "gamma", "sigT_Mpc", "kF_Mpc"]
for ii in range(len(central)):
    _cen = central[ii]
    _seed = seed[ii]
    zlist.append(_cen["z"])

    tar = _cen.copy()
    for par in par_merge:
        tar[par] = 0.5 * (_cen[par] + _seed[par])
        
    tar["p1d_Mpc"] = (_cen["mF"]**2 * _cen["p1d_Mpc"] + _seed["mF"]**2 * _seed["p1d_Mpc"]) / tar["mF"]**2 / 2
    tar["p3d_Mpc"] = (_cen["mF"]**2 * _cen["p3d_Mpc"] + _seed["mF"]**2 * _seed["p3d_Mpc"]) / tar["mF"]**2 / 2

    # print(cen["mF"], seed["mF"], tar["mF"])
    list_merge.append(tar)

# %%
from forestflow.utils import params_numpy2dict_minimizerz

def paramz_to_paramind(z, paramz):
    paramind = []
    for ii in range(len(z)):
        param = {}
        for key in paramz:
            param[key] = 10 ** np.poly1d(paramz[key])(z[ii])
        paramind.append(param)
    return paramind

file = path_program + "/data/best_arinyo/minimizer/fit_sim_label_combo_kmax3d_5_kmax1d_4.npz"
data = np.load(file, allow_pickle=True)
# best_params = paramz_to_paramind(zlist, data["best_params"].item())

for ii in range(len(list_merge)):
    list_merge[ii]["Arinyo_min"] = data["best_params"][ii]
    # list_merge[ii]["Arinyo_minz"] = params_numpy2dict_minimizerz(best_params[ii])

# %% [markdown]
# ### Error from cosmic variance
#
# Difference across z between best-fitting models to central and seed relative to their average

# %%
from forestflow.rebin_p3d import p3d_allkmu, get_p3d_modes, p3d_rebin_mu
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

n_mubins = 4
kmax_3d_fit = 5
kmax_1d_fit = 4
kmax_3d = kmax_3d_fit + 1
kmax_1d = kmax_1d_fit + 1

k3d_Mpc = central[0]['k3d_Mpc']
mu3d = central[0]['mu3d']
kmu_modes = get_p3d_modes(kmax_3d)
mask_3d = k3d_Mpc[:, 0] <= kmax_3d
nk = np.sum(mask_3d)
mask_1d = central[0]['k_Mpc'] < kmax_1d
k1d_Mpc = central[0]['k_Mpc'][mask_1d]


# %%
list_sims = [central, seed, list_merge]
nsims = len(list_sims)

p3d_measured = np.zeros((nsims, len(central), np.sum(mask_3d), n_mubins))
p3d_model = np.zeros((nsims, len(central), np.sum(mask_3d), n_mubins))
params = np.zeros((nsims, len(central), 2))

p1d_measured = np.zeros((nsims, len(central), np.sum(mask_1d)))
p1d_model = np.zeros((nsims, len(central), np.sum(mask_1d)))

for isnap in range(len(central)):
    z = central[isnap]["z"]
    
    for ii in range(nsims):
        sim = list_sims[ii]
    
        _ = p3d_rebin_mu(k3d_Mpc[mask_3d], mu3d[mask_3d], sim[isnap]['p3d_Mpc'][mask_3d], kmu_modes, n_mubins=n_mubins)
        knew, munew, p3d_measured[ii, isnap, ...], mu_bins = _
        p1d_measured[ii, isnap, :] = sim[isnap]['p1d_Mpc'][mask_1d]
    
        pp = sim[isnap]["Arinyo_min"]
        model_p3d, plin = p3d_allkmu(
            sim[isnap]['model'],
            z,
            pp,
            kmu_modes,
            nk=nk,
            nmu=16,
            compute_plin=True,
        )        
        _ = p3d_rebin_mu(k3d_Mpc[:nk], 
                         mu3d[:nk], 
                         model_p3d[:nk], 
                         kmu_modes, 
                         n_mubins=n_mubins)
        knew, munew, rebin_model_p3d, mu_bins = _
        
        p3d_model[ii, isnap, ...] = rebin_model_p3d
        p1d_model[ii, isnap, :] = sim[isnap]["model"].P1D_Mpc(z, k1d_Mpc, parameters=pp)

        params[ii, isnap, 0] = pp["bias"]
        params[ii, isnap, 1] = pp["beta"]

# %%
for iz in range(len(central)):

    # if(central["z"][iz] == 2) | (central["z"][iz] == 3) | (central["z"][iz] == 4.5):
    if(central[iz]["z"] != 0):
        pass
    else:
        continue
    
    jj = 0
    fig, ax = plt.subplots(2, sharex=True)
    
    for ii in range(n_mubins):
        col = f"C{ii}"
        x = knew[:, ii] 
        _ = np.isfinite(x)        
        y = (p3d_model[0, iz, :, ii] - p3d_model[1, iz, :, ii])/p3d_model[2, iz, :, ii]/np.sqrt(2)
        ax[0].plot(x[_], y[_], col+"-")

    ax[0].axhline(0, linestyle=":", color="k")
    ax[0].axhline(0.1, linestyle=":", color="k")
    ax[0].axhline(-0.1, linestyle=":", color="k")
    ax[0].axvline(kmax_3d_fit, linestyle=":", color="k")
    
    x = k1d_Mpc
    y = (p1d_model[0, iz, :] - p1d_model[1, iz, :])/p1d_model[2, iz, :]/np.sqrt(2)
    ax[1].plot(x, y, "-")
    
    ax[1].axhline(0, linestyle=":", color="k")
    ax[1].axhline(0.01, linestyle=":", color="k")
    ax[1].axhline(-0.01, linestyle=":", color="k")
    ax[1].axvline(kmax_1d_fit, linestyle=":", color="k")
    
    ax[0].set_title("z="+str(central[iz]["z"]))
    ax[0].set_xscale("log")
    ax[0].set_ylim(-0.2, 0.2)
    ax[1].set_ylim(-0.02, 0.02)

# %%
z_grid = np.array([d["z"] for d in central])

for ii in range(2):
    y = (params[0, :, ii] - params[1, :, ii])/params[2, :, ii]/np.sqrt(2)
    print(np.mean(y)*100, np.std(y)*100)
    plt.plot(z_grid, y)

# %%
