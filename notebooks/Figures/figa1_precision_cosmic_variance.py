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
# # Cosmic variance on data
#
# Difference of central and seed divided by their combination

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
combo = []
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

    combo.append(tar)

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
mask_1d = (central[0]['k_Mpc'] < kmax_1d) & (central[0]['k_Mpc'] > 0)
k1d_Mpc = central[0]['k_Mpc'][mask_1d]


# %%
list_sims = [central, seed, combo]
nsims = len(list_sims)
p3d_measured = np.zeros((nsims, len(central), np.sum(mask_3d), n_mubins))
p1d_measured = np.zeros((nsims, len(central), np.sum(mask_1d)))

for isnap in range(len(combo)):
    for ii in range(nsims):
        sim = list_sims[ii]
    
        _ = p3d_rebin_mu(k3d_Mpc[mask_3d], mu3d[mask_3d], sim[isnap]['p3d_Mpc'][mask_3d], kmu_modes, n_mubins=n_mubins)
        knew, munew, p3d_measured[ii, isnap, ...], mu_bins = _
        p1d_measured[ii, isnap, :] = sim[isnap]['p1d_Mpc'][mask_1d]

# %%
_ = np.isfinite(knew) & (knew > 0.5) & (knew < 5)
y = (p3d_measured[0, :, _] - p3d_measured[1, :, _])/p3d_measured[2, :, _]/np.sqrt(2)
res = np.percentile(y, [50, 16, 84])
print(res[0]*100, 0.5*(res[2]-res[1])*100, np.std(y)*100)

y = (p1d_measured[0, :, :] - p1d_measured[1, :, :])/p1d_measured[2, :, :]/np.sqrt(2)
res = np.percentile(y, [50, 16, 84])
print(res[0]*100, 0.5*(res[2]-res[1])*100, np.std(y)*100)

# %%
out = 3
folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"

for iz in range(len(central)):

    # if(central["z"][iz] == 2) | (central["z"][iz] == 3) | (central["z"][iz] == 4.5):
    if(central[iz]["z"] == out):
        pass
    else:
        continue
    
    jj = 0
    ftsize = 20
    fig, ax = plt.subplots(2, figsize=(8, 6), sharex=True)
    
    for ii in range(n_mubins):
        
        if ii == 0:
            lab = str(mu_bins[ii]) + r"$\leq\mu<$" + str(mu_bins[ii + 1])
        else:
            lab = str(mu_bins[ii]) + r"$\leq\mu\leq$" + str(mu_bins[ii + 1])
        col = f"C{ii}"
        x = knew[:, ii] 
        _ = np.isfinite(x)
        y = (p3d_measured[0, iz, :, ii] - p3d_measured[1, iz, :, ii])/p3d_measured[2, iz, :, ii]/np.sqrt(2)
        print(np.nanmax(np.abs(y)))
        ax[0].plot(x[_], y[_], col+"-", lw=3, label=lab)
        
    _ = np.isfinite(knew) & (knew > 0.5) & (knew < 5)
    y = (p3d_measured[0, iz, _] - p3d_measured[1, iz, _])/p3d_measured[2, iz, _]/np.sqrt(2)
    res = np.percentile(y, [50, 16, 84])
    print(res[0]*100, 0.5*(res[2]-res[1])*100, np.std(y)*100)

    x = k1d_Mpc
    y = (p1d_measured[0, iz, :] - p1d_measured[1, iz, :])/p1d_measured[2, iz, :]/np.sqrt(2)
    print(np.nanmax(np.abs(y)))
    ax[1].plot(x, y, "C4-", lw=3)

    res = np.percentile(y, [50, 16, 84])
    print(res[0]*100, 0.5*(res[2]-res[1])*100, np.std(y)*100)

    ax[0].legend(fontsize=16, ncol=2, loc="upper left")
    
    ax[0].axhline(0, linestyle=":", color="k")
    ax[0].axhline(0.1, linestyle="--", color="k")
    ax[0].axhline(-0.1, linestyle="--", color="k")
    ax[0].axvline(kmax_3d_fit, linestyle="--", color="k")
    ax[1].axhline(0, linestyle=":", color="k")
    ax[1].axhline(0.01, linestyle="--", color="k")
    ax[1].axhline(-0.01, linestyle="--", color="k")
    ax[1].axvline(kmax_1d_fit, linestyle="--", color="k")

    ax[0].set_ylabel(r"Residual $P_\mathrm{3D}$", fontsize=ftsize)
    ax[1].set_ylabel(r"Residual $P_\mathrm{1D}$", fontsize=ftsize)
    
    ax[0].set_xlabel(r"$k\, [\mathrm{Mpc}^{-1}]$", fontsize=ftsize)
    ax[1].set_xlabel(r"$k_\parallel\, [\mathrm{Mpc}^{-1}]$", fontsize=ftsize)

    ax[0].tick_params(axis="both", which="major", labelsize=ftsize)
    ax[1].tick_params(axis="both", which="major", labelsize=ftsize)

    if(central[iz]["z"] != out):
        ax[0].set_title("z="+str(central[iz]["z"]))
    ax[0].set_xscale("log")
    ax[0].set_ylim(-0.21, 0.31)
    ax[1].set_ylim(-0.021, 0.021)

    plt.tight_layout()
    plt.savefig(folder + "cvariance_z_"+str(central[iz]["z"])+".png")
    plt.savefig(folder + "cvariance_z_"+str(central[iz]["z"])+".pdf")

# %% [markdown]
# ### Save data for zenodo

# %%
conv = {}
conv["blue"] = 0
conv["orange"] = 1
conv["green"] = 2
conv["red"] = 3
outs = {}

for jj in range(len(central)):
    if(central[jj]["z"] == 3):
        iz = jj

for key in conv.keys():
    ii = conv[key]

    outs["top_" + key + "_x"] = knew[:, ii]
    outs["top_" + key + "_y"] = (p3d_measured[0, iz, :, ii] - p3d_measured[1, iz, :, ii])/p3d_measured[2, iz, :, ii]/np.sqrt(2)

outs["bottom_x"] = k1d_Mpc
outs["bottom_y"] = (p1d_measured[0, iz, :] - p1d_measured[1, iz, :])/p1d_measured[2, iz, :]/np.sqrt(2)


# %%
import forestflow
path_forestflow= os.path.dirname(forestflow.__path__[0]) + "/"
folder = path_forestflow + "data/figures_machine_readable/"
np.save(folder + "figa1", outs)

# %%
res = np.load(folder + "figa1.npy", allow_pickle=True).item()
res.keys()

# %% [markdown]
# ## From different axes

# %%
central_0 = []
central_1 = []
central_2 = []
central_t = []
seed_0 = []
seed_1 = []
seed_2 = []
seed_t = []

for ii in range(len(Archive3D.data)):
    if(Archive3D.data[ii]["sim_label"] == "mpg_central") & (Archive3D.data[ii]["val_scaling"] ==1) & (Archive3D.data[ii]["z"] == 3):
        if(Archive3D.data[ii]["ind_axis"] == 0):
            central_0.append(Archive3D.data[ii])
        if(Archive3D.data[ii]["ind_axis"] == 1):
            central_1.append(Archive3D.data[ii])
        if(Archive3D.data[ii]["ind_axis"] == 2):
            central_2.append(Archive3D.data[ii])
        central_t.append(Archive3D.data[ii])
    if(Archive3D.data[ii]["sim_label"] == "mpg_seed") & (Archive3D.data[ii]["val_scaling"] ==1) & (Archive3D.data[ii]["z"] == 3):
        if(Archive3D.data[ii]["ind_axis"] == 0):
            seed_0.append(Archive3D.data[ii])
        if(Archive3D.data[ii]["ind_axis"] == 1):
            seed_1.append(Archive3D.data[ii])
        if(Archive3D.data[ii]["ind_axis"] == 2):
            seed_2.append(Archive3D.data[ii])
        seed_t.append(Archive3D.data[ii])

# %%
av_p3d_cen = []
av_p3d_seed = []
av_p1d_cen = []
av_p1d_seed = []

for sim in [central_0, central_1, central_2]:
    mF = 0
    for ii in range(2):
        mF += sim[ii]["mF"]/2
    p3d = 0
    p1d = 0
    for ii in range(2):
        p3d += sim[ii]["mF"]**2 * sim[ii]["p3d_Mpc"]
        p1d += sim[ii]["mF"]**2 * sim[ii]["p1d_Mpc"]
    p3d = p3d/mF**2/2
    p1d = p1d/mF**2/2
    
    av_p3d_cen.append(p3d)
    av_p1d_cen.append(p1d)


for sim in [seed_0, seed_1, seed_2]:
    mF = 0
    for ii in range(2):
        mF += sim[ii]["mF"]/2
    p3d = 0
    p1d = 0
    for ii in range(2):
        p3d += sim[ii]["mF"]**2 * sim[ii]["p3d_Mpc"]
        p1d += sim[ii]["mF"]**2 * sim[ii]["p1d_Mpc"]
    p3d = p3d/mF**2/2
    p1d = p1d/mF**2/2
    
    av_p3d_seed.append(p3d)
    av_p1d_seed.append(p1d)

# %%
nsims = len(central_t)

mF = 0
for ii in range(nsims):
    mF += central_t[ii]["mF"]/nsims
av_p3d_cent = 0
av_p1d_cent = 0
for ii in range(nsims):
    av_p3d_cent += central_t[ii]["mF"]**2 * central_t[ii]["p3d_Mpc"]
    av_p1d_cent += central_t[ii]["mF"]**2 * central_t[ii]["p1d_Mpc"]
av_p3d_cent = av_p3d_cent/mF**2/nsims
av_p1d_cent = av_p1d_cent/mF**2/nsims

mF = 0
for ii in range(nsims):
    mF += seed_t[ii]["mF"]/nsims
av_p3d_seedt = 0
av_p1d_seedt = 0
for ii in range(nsims):
    av_p3d_seedt += seed_t[ii]["mF"]**2 * seed_t[ii]["p3d_Mpc"]
    av_p1d_seedt += seed_t[ii]["mF"]**2 * seed_t[ii]["p1d_Mpc"]
av_p3d_seedt = av_p3d_seedt/mF**2/nsims
av_p1d_seedt = av_p1d_seedt/mF**2/nsims

# %%
from forestflow.rebin_p3d import p3d_allkmu, get_p3d_modes, p3d_rebin_mu

isnap = 6
n_mubins = 4
kmax_3d_fit = 5
kmax_1d_fit = 4
kmax_3d = kmax_3d_fit + 1
kmax_1d = kmax_1d_fit + 1

k3d_Mpc = combo[0]['k3d_Mpc']
mu3d = combo[0]['mu3d']
kmu_modes = get_p3d_modes(kmax_3d)
mask_3d = k3d_Mpc[:, 0] <= kmax_3d
mask_1d = combo[0]['k_Mpc'] < kmax_1d
k1d_Mpc = combo[0]['k_Mpc'][mask_1d]

_ = p3d_rebin_mu(k3d_Mpc[mask_3d], mu3d[mask_3d], combo[isnap]['p3d_Mpc'][mask_3d], kmu_modes, n_mubins=n_mubins)
knew, munew, combo_bin, mu_bins = _

_ = p3d_rebin_mu(k3d_Mpc[mask_3d], mu3d[mask_3d], av_p3d_cent[mask_3d], kmu_modes, n_mubins=n_mubins)
knew, munew, av_p3d_cent_bin, mu_bins = _

_ = p3d_rebin_mu(k3d_Mpc[mask_3d], mu3d[mask_3d], av_p3d_seedt[mask_3d], kmu_modes, n_mubins=n_mubins)
knew, munew, av_p3d_seedt_bin, mu_bins = _


# %%

fig, ax = plt.subplots(4, sharex=True, sharey=True, figsize=(8, 6))
for ii in range(3):
    _ = p3d_rebin_mu(k3d_Mpc[mask_3d], mu3d[mask_3d], av_p3d_cen[ii], kmu_modes, n_mubins=n_mubins)
    knew, munew, cen_bin, mu_bins = _
    
    _ = p3d_rebin_mu(k3d_Mpc[mask_3d], mu3d[mask_3d], av_p3d_seed[ii], kmu_modes, n_mubins=n_mubins)
    knew, munew, seed_bin, mu_bins = _

    for jj in range(4):
        _ = np.isfinite(knew[:, jj]) & (knew[:, jj] < 0.5)
        
        y = (cen_bin[:,jj] - seed_bin[:,jj])/combo_bin[:,jj]/np.sqrt(2)
        ax[jj].plot(knew[_, jj], y[_])

        if(ii == 2):
            y = (av_p3d_cent_bin[:,jj] - av_p3d_seedt_bin[:,jj])/combo_bin[:,jj]/np.sqrt(2)
            ax[jj].plot(knew[_, jj], y[_], "--", lw=2)

for jj in range(4):
    ax[jj].axhline(0, linestyle=":", color="k")
    ax[jj].axhline(0.1, linestyle="--", color="k")
    ax[jj].axhline(-0.1, linestyle="--", color="k")
ax[0].set_ylim(-0.21, 0.21)

# plt.xscale('log')
plt.tight_layout()
plt.savefig("cvar_axes.png")

# %%
ls = ["-", ":", "--"]
fig, ax = plt.subplots(16, sharex=True, sharey=True, figsize=(8, 20))
for ii in range(3):

    for jj in range(16):
        _ = np.isfinite(k3d_Mpc[:, jj]) & (k3d_Mpc[:, jj] < 0.5)
        
        y = (av_p3d_cen[ii][:,jj] - av_p3d_seed[ii][:,jj])/combo[isnap]['p3d_Mpc'][:,jj]/np.sqrt(2)
        ax[jj].plot(k3d_Mpc[_, jj], y[_])

        if(ii == 2):
            y = (av_p3d_cent[:,jj] - av_p3d_seedt[:,jj])/combo[isnap]['p3d_Mpc'][:,jj]/np.sqrt(2)
            ax[jj].plot(k3d_Mpc[_, jj], y[_], "--")

for jj in range(16):
    ax[jj].axhline(0, linestyle=":", color="k")
    ax[jj].axhline(0.1, linestyle="--", color="k")
    ax[jj].axhline(-0.1, linestyle="--", color="k")

ax[0].set_ylim(-0.21, 0.21)
# plt.xscale('log')
plt.tight_layout()
plt.savefig("cvar_axes_no_rebinning.png")

# %%
len(av_p3d_seed[ii])

# %%

# %%
