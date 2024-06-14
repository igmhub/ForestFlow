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
# ## LOAD EMULATOR

# %%
# training_type = "Arinyo_min"
# model_path=path_program + "/data/emulator_models/mpg_last.pt"
training_type = "Arinyo_minz"
# model_path=path_program + "/data/emulator_models/mpg_minz.pt"
model_path=path_program + "/data/emulator_models/mpg_jointz.pt"

emulator = P3DEmulator(
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
    # Nrealizations=10000,
    Nrealizations=1000,
    training_type=training_type,
    model_path=model_path,
    # save_path=model_path,
)


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

file = path_program + "/data/best_arinyo/minimizer_z/fit_sim_label_combo_val_scaling_1_kmax3d_3_kmax1d_3.npz"
data = np.load(file, allow_pickle=True)
best_params = paramz_to_paramind(zlist, data["best_params"].item())

for ii in range(len(list_merge)):
    list_merge[ii]["Arinyo_minz"] = params_numpy2dict_minimizerz(best_params[ii])

# %% [markdown]
# ### Error from cosmic variance
#
# Difference across z between best-fitting models to central and seed relative to their average

# %%
from forestflow.rebin_p3d import p3d_allkmu, get_p3d_modes, p3d_rebin_mu
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

n_mubins = 4
kmax = 4
kmax_fit = 3

k3d_Mpc = central[0]['k3d_Mpc']
mu3d = central[0]['mu3d']
kmu_modes = get_p3d_modes(kmax)
mask_3d = k3d_Mpc[:, 0] <= kmax
mask_1d = central[0]['k_Mpc'] < kmax
k1d_Mpc = central[0]['k_Mpc'][mask_1d]

info_power = {
    "sim_label": "mpg_central",
    "k3d_Mpc": k3d_Mpc[mask_3d, :],
    "mu": mu3d[mask_3d, :],
    "kmu_modes": kmu_modes,
    "k1d_Mpc": k1d_Mpc,
    "return_p3d": True,
    "return_p1d": True,
    "return_cov": True,
}

# %%
eva_emu = False

list_sims = [central, seed, list_merge]
nsims = len(list_sims)

p3d_measured = np.zeros((nsims, len(central), np.sum(mask_3d), n_mubins))
p3d_model = np.zeros((nsims, len(central), np.sum(mask_3d), n_mubins))
if eva_emu:
    p3d_emu = np.zeros((len(central), np.sum(mask_3d), n_mubins))

p1d_measured = np.zeros((nsims, len(central), np.sum(mask_1d)))
p1d_model = np.zeros((nsims, len(central), np.sum(mask_1d)))
if eva_emu:
    p1d_emu = np.zeros((len(central), np.sum(mask_1d)))

for isnap in range(len(central)):
    z = central[isnap]["z"]
    if eva_emu:
        info_power["z"] = z    
        out = emulator.evaluate(
            emu_params=central[isnap],
            info_power=info_power,
            Nrealizations=100
        )    
        _ = p3d_rebin_mu(out["k_Mpc"], out["mu"], out["p3d"], kmu_modes, n_mubins=n_mubins)
        knew, munew, p3d_emu[isnap], mu_bins = _
        p1d_emu[isnap] = out["p1d"]
    
    for ii in range(nsims):
        sim = list_sims[ii]
    
        _ = p3d_rebin_mu(k3d_Mpc[mask_3d], mu3d[mask_3d], sim[isnap]['p3d_Mpc'][mask_3d], kmu_modes, n_mubins=n_mubins)
        knew, munew, p3d_measured[ii, isnap, ...], mu_bins = _
        p1d_measured[ii, isnap, :] = sim[isnap]['p1d_Mpc'][mask_1d]
    
        pp = sim[isnap]["Arinyo_minz"]
        p3d_model[ii, isnap, ...] = sim[isnap]["model"].P3D_Mpc(z, knew, munew, pp)
        p1d_model[ii, isnap, :] = sim[isnap]["model"].P1D_Mpc(z, k1d_Mpc, parameters=pp)

# %%
sim[isnap].keys()

# %%
per_use = [50, 16, 84]
per3_data = np.nanpercentile(p3d_measured[:2]/p3d_measured[2] - 1, per_use, axis=(0,1,3))
per3_model = np.nanpercentile(p3d_model[:2]/p3d_model[2] - 1, per_use, axis=(0,1,3))
per3_data_emu = np.nanpercentile(p3d_emu/p3d_measured[0] - 1, per_use, axis=(0,2))
per3_data_model = np.nanpercentile(p3d_model[0]/p3d_measured[0] - 1, per_use, axis=(0, 2))

per1_data = np.nanpercentile(p1d_measured[:2]/p1d_measured[2] - 1, per_use, axis=(0,1))
per1_model = np.nanpercentile(p1d_model[:2]/p1d_model[2] - 1, per_use, axis=(0,1))
per1_data_emu = np.nanpercentile(p1d_emu/p1d_measured[0] - 1, per_use, axis=(0))
per1_data_model = np.nanpercentile(p1d_model[0]/p1d_measured[0] - 1, per_use, axis=(0))

# %%

# %%
knew_av = np.nanmedian(knew, axis=1)

# %%
plt.errorbar(knew_av, per_data[0], np.abs(per_data[1:]-per_data[0]), alpha=0.5)
plt.errorbar(knew_av, per_model[0], np.abs(per_model[1:]-per_model[0]), alpha=0.5)
plt.errorbar(knew_av, per_data_emu[0], np.abs(per_data_emu[1:]-per_data_emu[0]), alpha=0.5)

plt.xscale("log")

# %%
lw = 3

fig, ax = plt.subplots(1)

ax.plot(knew_av, 0.5*(per3_data[2]-per3_data[1])*100, alpha=0.5, lw=lw, 
        label="Data: cosmic variance")
ax.plot(knew_av, 0.5*(per3_model[2]-per3_model[1])*100, alpha=0.5, lw=lw, 
        label="Fit: cosmic variance")
ax.plot(knew_av, 0.5*(per3_data_model[2]-per3_data_model[1])*100, alpha=0.5, lw=lw, 
        label="Fit vs data")
ax.plot(knew_av, 0.5*(per3_data_emu[2]-per3_data_emu[1])*100, alpha=0.5, lw=lw, 
        label="ForestFlow vs data")

ax.set_ylabel("Error P3D [%]")
ax.set_xlabel("k")
ax.legend()

ax.set_xscale("log")
plt.savefig("pre_p3d.png")

# %%
lw = 3

fig, ax = plt.subplots(1)

ax.plot(k1d_Mpc, 0.5*(per1_data[2]-per1_data[1])*100, alpha=0.5, lw=lw, 
        label="Data: cosmic variance")
ax.plot(k1d_Mpc, 0.5*(per1_model[2]-per1_model[1])*100, alpha=0.5, lw=lw, 
        label="Fit: cosmic variance")
ax.plot(k1d_Mpc, 0.5*(per1_data_model[2]-per1_data_model[1])*100, alpha=0.5, lw=lw, 
        label="Fit vs data")
ax.plot(k1d_Mpc, 0.5*(per1_data_emu[2]-per1_data_emu[1])*100, alpha=0.5, lw=lw, 
        label="ForestFlow vs data")

ax.set_ylabel("Error P1D [%]")
ax.set_xlabel("kpar")
ax.legend()

ax.set_xscale("log")
plt.savefig("pre_p1d.png")

# %%
# per3_data = np.nanpercentile(p3d_measured[:2]/p3d_measured[2] - 1, per_use, axis=(0,1,3))
# per3_model = np.nanpercentile(p3d_model[:2]/p3d_model[2] - 1, per_use, axis=(0,1,3))
per3_data_emu = np.nanpercentile(p3d_emu/p3d_measured[0] - 1, per_use, axis=(2))
per3_data_model = np.nanpercentile(p3d_model[0]/p3d_measured[0] - 1, per_use, axis=(2))

# per1_data = np.nanpercentile(p1d_measured[:2]/p1d_measured[2] - 1, per_use, axis=(0,1))
# per1_model = np.nanpercentile(p1d_model[:2]/p1d_model[2] - 1, per_use, axis=(0,1))
per1_data_emu = np.nanpercentile(p1d_emu/p1d_measured[0] - 1, per_use)
per1_data_model = np.nanpercentile(p1d_model[0]/p1d_measured[0] - 1, per_use)

# %%
for ii in range(per3_data_emu.shape[1]):
    col = "C"+str(ii%10)
    plt.plot(knew_av, per3_data_model[0, ii], col, label="z="+str(zlist[ii]))
    plt.plot(knew_av, per3_data_emu[0, ii], col+'--')
plt.xscale('log')

plt.ylabel("P3D/P3Ddata-1")
plt.xlabel("k")
plt.legend()
plt.savefig("p3d_z.png")

# %%
per3_data_emu.shape

# %%

# %%
p3d_measured.shape

# %%
per_data[1]

# %% [markdown]
# ### Continue

# %%
Arinyo_coeffs_central = np.array(
    [list(central[i][training_type].values()) for i in range(len(central))]
)

Arinyo_coeffs_seed = np.array(
    [list(seed[i][training_type].values()) for i in range(len(seed))]
)

Arinyo_coeffs_merge = np.array(
    [list(list_merge[i][training_type].values()) for i in range(len(list_merge))]
)


# %%
Arinyo_central = []
Arinyo_seed = []
Arinyo_merge = []
for ii in range(Arinyo_coeffs_central.shape[0]):
    dict_params = params_numpy2dict(Arinyo_coeffs_central[ii])
    new_params = transform_arinyo_params(dict_params, central[ii]["f_p"])
    Arinyo_central.append(new_params)
    
    dict_params = params_numpy2dict(Arinyo_coeffs_seed[ii])
    new_params = transform_arinyo_params(dict_params, seed[ii]["f_p"])
    Arinyo_seed.append(new_params)

    dict_params = params_numpy2dict(Arinyo_coeffs_merge[ii])
    new_params = transform_arinyo_params(dict_params, list_merge[ii]["f_p"])
    Arinyo_merge.append(new_params)

# %% [markdown]
# ## LOOP OVER REDSHIFTS PREDICTING THE ARINYO PARAMETERS

# %%
z_central = [d["z"] for d in central]
z_central

# %%
Arinyo_emu = []
Arinyo_emu_std = []

for iz, z in enumerate(z_central):
    test_sim_z = [d for d in central if d["z"] == z]
    out = p3d_emu.evaluate(
        emu_params=test_sim_z[0],
        natural_params=True,
        Nrealizations=10000,
    )
    Arinyo_emu.append(out["coeffs_Arinyo"])
    Arinyo_emu_std.append(out["coeffs_Arinyo_std"])


# %% [markdown]
# ## PLOT

# %%
from forestflow.plots.params_z import plot_arinyo_z, plot_forestflow_z

# %%
folder_fig = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"

# %%
# for ii in range(len(Arinyo_emu)):
#     print(ii, Arinyo_emu[ii]["kv"])
#     Arinyo_emu[ii]["kv"] = Arinyo_emu[ii]["kv"]**Arinyo_emu[ii]["av"]
#     print(Arinyo_emu[ii]["kv"])

# %%
plot_arinyo_z(z_central, Arinyo_central, Arinyo_seed, Arinyo_merge, folder_fig=folder_fig, ftsize=20)

# %%
plot_forestflow_z(z_central, Arinyo_central, Arinyo_seed, Arinyo_emu, Arinyo_emu_std, folder_fig=folder_fig, ftsize=20)

# %%
Arinyo_central[0].keys()

# %%
kaiser_cen = np.zeros((len(Arinyo_central), 2))
kaiser_seed = np.zeros((len(Arinyo_central), 2))
kaiser_merge = np.zeros((len(Arinyo_central), 2))
for iz in range(len(Arinyo_central)):
    kaiser_cen[iz, 0] = (Arinyo_central[iz]["bias"])**2
    kaiser_cen[iz, 1] = (Arinyo_central[iz]["bias"] + Arinyo_central[iz]["bias_eta"])**2
    kaiser_seed[iz, 0] = (Arinyo_seed[iz]["bias"])**2
    kaiser_seed[iz, 1] = (Arinyo_seed[iz]["bias"] + Arinyo_seed[iz]["bias_eta"])**2
    kaiser_merge[iz, 0] = (Arinyo_merge[iz]["bias"])**2
    kaiser_merge[iz, 1] = (Arinyo_merge[iz]["bias"] + Arinyo_merge[iz]["bias_eta"])**2

# %%
for ii in range(2):
    y = np.concatenate([kaiser_cen[:,ii]/kaiser_merge[:,ii] - 1, kaiser_seed[:,ii]/kaiser_merge[:,ii] - 1])
    s_pred = np.percentile(y, [16, 50, 84])
    std = 0.5 * (s_pred[2] - s_pred[0]) * 100
    print(ii, s_pred[1]* 100, std)


# %%
