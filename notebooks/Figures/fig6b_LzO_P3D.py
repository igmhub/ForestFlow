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
# # Leave-redshift-out

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from forestflow.model_p3d_arinyo import ArinyoModel
from forestflow.archive import GadgetArchive3D
from forestflow.P3D_cINN import P3DEmulator
from forestflow.plots.l1O_p3d import plot_p3d_L1O
from forestflow.plots.l1O_p1d import plot_p1d_L1O

from forestflow.rebin_p3d import get_p3d_modes, p3d_allkmu, p3d_rebin_mu

from matplotlib import rcParams

from forestflow.utils import (
    params_numpy2dict,
    transform_arinyo_params,
)

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

# %% [markdown]
#
# ## DEFINE FUNCTIONS


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
# # LOAD DATA

# %%
# %%time
folder_interp = path_program + "/data/plin_interp/"
folder_lya_data = path_program + "/data/best_arinyo/"

Archive3D = GadgetArchive3D(
    base_folder=path_program,
    folder_data=folder_lya_data,
    force_recompute_plin=False,
    average="both",
)
print(len(Archive3D.training_data))


# %% [markdown]
# ### Train L1Os

# %%
z_test = [3.5, 2.5]

model_path = path_program+"/data/emulator_models/"
training_type = "Arinyo_min"
for iz, zdrop in enumerate(z_test):
    print(f"Dropping redshift {zdrop}")

    training_data = [d for d in Archive3D.training_data if d["z"] != zdrop]

    p3d_emu = P3DEmulator(
        training_data,
        Archive3D.emu_params,
        nepochs=300,
        lr=0.001,  # 0.005
        batch_size=20,
        step_size=200,
        gamma=0.1,
        weight_decay=0,
        adamw=True,
        nLayers_inn=12,  # 15
        Nrealizations=200,
        Archive=Archive3D,
        training_type=training_type,
        save_path=model_path + f"mpg_dropz{zdrop}.pt",
    )

# %% [markdown]
# ### Evaluate L1Os

# %%
training_type = "Arinyo_min"
model_path = path_program+"/data/emulator_models/"

Nsim = 30
zs = np.flip(np.arange(2, 4.6, 0.25))
Nz = zs.shape[0]

n_mubins = 4
kmax_3d_fit = 5
kmax_1d_fit = 4
kmax_3d_plot = kmax_3d_fit + 1
kmax_1d_plot = kmax_1d_fit + 1

sim = Archive3D.training_data[0]

k3d_Mpc = sim['k3d_Mpc']
mu3d = sim['mu3d']
p3d_Mpc = sim['p3d_Mpc']
kmu_modes = get_p3d_modes(kmax_3d_plot)

mask_3d = k3d_Mpc[:, 0] <= kmax_3d_plot

mask_1d = (sim['k_Mpc'] <= kmax_1d_plot) & (sim['k_Mpc'] > 0)
k1d_Mpc = sim['k_Mpc'][mask_1d]
p1d_Mpc = sim['p1d_Mpc'][mask_1d]

# %%

# %%

z_test = [3.5, 2.5]
Nz = len(z_test)

arr_p3d_sim = np.zeros((Nsim, Nz, np.sum(mask_3d), n_mubins))
arr_p3d_emu = np.zeros((Nsim, Nz, np.sum(mask_3d), n_mubins))
arr_p1d_sim = np.zeros((Nsim, Nz, np.sum(mask_1d)))
arr_p1d_emu = np.zeros((Nsim, Nz, np.sum(mask_1d)))
params_sim = np.zeros((Nsim, Nz, 2))
params_emu = np.zeros((Nsim, Nz, 2))

for iz, zdrop in enumerate(z_test):
    print(f"Dropping redshift {zdrop}")

    training_data = [d for d in Archive3D.training_data if d["z"] != zdrop]

    p3d_emu = P3DEmulator(
        training_data,
        Archive3D.emu_params,
        nepochs=300,
        lr=0.001,  # 0.005
        batch_size=20,
        step_size=200,
        gamma=0.1,
        weight_decay=0,
        adamw=True,
        nLayers_inn=12,  # 15
        Nrealizations=200,
        Archive=Archive3D,
        training_type=training_type,
        model_path=model_path + f"mpg_dropz{zdrop}.pt",
    )

    for isim in range(Nsim):
        print(isim)

        # define test sim
        sim_label = f"mpg_{isim}"
        dict_sim = [
            d
            for d in Archive3D.training_data
            if d["z"] == zdrop
            and d["sim_label"] == sim_label
            and d["val_scaling"] == 1
        ]

        info_power = {
            "sim_label": sim_label,
            "k3d_Mpc": k3d_Mpc[mask_3d, :],
            "mu": mu3d[mask_3d, :],
            "kmu_modes": kmu_modes,
            "k1d_Mpc": k1d_Mpc,
            "return_p3d": True,
            "return_p1d": True,
            "z": zdrop,
        }
        
        out = p3d_emu.evaluate(
            emu_params=dict_sim[0],
            info_power=info_power,
            natural_params=True,
            Nrealizations=100
        )
        
        # p1d and p3d from sim
        _ = p3d_rebin_mu(out["k_Mpc"], out["mu"], dict_sim[0]["p3d_Mpc"][mask_3d], kmu_modes, n_mubins=n_mubins)
        knew, munew, arr_p3d_sim[isim, iz], mu_bins = _
        
        _ = p3d_rebin_mu(out["k_Mpc"], out["mu"], out["p3d"], kmu_modes, n_mubins=n_mubins)
        knew, munew, arr_p3d_emu[isim, iz], mu_bins = _

        arr_p1d_emu[isim, iz] = out["p1d"]
        arr_p1d_sim[isim, iz] = dict_sim[0]["p1d_Mpc"][mask_1d]

        params_emu[isim, iz, 0] = out['coeffs_Arinyo']["bias"]
        params_emu[isim, iz, 1] = out['coeffs_Arinyo']["bias_eta"]

        _ = transform_arinyo_params(dict_sim[0]["Arinyo_minz"], dict_sim[0]["f_p"])        
        params_sim[isim, iz, 0] = _["bias"]
        params_sim[isim, iz, 1] = _["bias_eta"]
        
        break
    break

# %%
folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"
np.savez(
    folder + "temporal_l1Oz", 
    arr_p3d_sim=arr_p3d_sim, 
    arr_p3d_emu=arr_p3d_emu, 
    arr_p1d_sim=arr_p1d_sim, 
    arr_p1d_emu=arr_p1d_emu,
    params_sim=params_sim,
    params_emu=params_emu
)

# %%
folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"
fil = np.load(folder + "temporal_l1Oz.npz")
params_emu = fil["params_emu"]
params_sim = fil["params_sim"]
arr_p3d_sim = fil["arr_p3d_sim"]
arr_p3d_emu = fil["arr_p3d_emu"]
arr_p1d_sim = fil["arr_p1d_sim"]
arr_p1d_emu = fil["arr_p1d_emu"]

# %%
for ii in range(2):
    print(ii)
    print(np.mean(params_emu[...,ii]/params_sim[...,ii]-1))
    print(np.std(params_emu[...,ii]/params_sim[...,ii]-1))
    rat = 0.5*(np.percentile(params_emu[...,ii]/params_sim[...,ii]-1, 68) - np.percentile(params_emu[...,ii]/params_sim[...,ii]-1, 16))
    print(rat * 100)
    # rat = 0.5*(np.percentile(param_pred[...,ii]/param_sims[...,ii]-1, 68, axis=0) - np.percentile(param_pred[...,ii]/param_sims[...,ii]-1, 16, axis=0))
    # print(rat)

# %% [markdown]
# ## PLOTTING

# %%
folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"
z_use = np.array(z_test)
z_use

# %% [markdown]
# #### P3D

# %%
residual3d = (arr_p3d_emu / arr_p3d_sim -1)

# %%
savename=folder+"l1O/l1O_z_P3D.png"
plot_p3d_L1O(z_use, knew, munew, residual3d, mu_bins, 
             kmax_3d_fit=kmax_3d_fit, legend=False, savename=savename)
savename=folder+"l1O/l1O_z_P3D.pdf"
plot_p3d_L1O(z_use, knew, munew, residual3d, mu_bins, 
             kmax_3d_fit=kmax_3d_fit, legend=False, savename=savename)

# %% [markdown]
# #### P1D

# %%
residual1d = (arr_p1d_emu / arr_p1d_sim -1)
residual1d.shape

# %%
savename=folder+"l1O/l1O_z_P1D.png"
plot_p1d_L1O(z_use, k1d_Mpc, residual1d, kmax_1d_fit=kmax_1d_fit, savename=savename)
savename=folder+"l1O/l1O_z_P1D.pdf"
plot_p1d_L1O(z_use, k1d_Mpc, residual1d, kmax_1d_fit=kmax_1d_fit, savename=savename)

# %% [markdown]
# ### Save data for zenodo

# %%
conv = {}
conv["blue"] = 0
conv["orange"] = 1
conv["green"] = 2
conv["red"] = 3
outs = {}

med_rat_p3d = np.median(residual3d, axis=0)
med_rat_p1d = np.median(residual1d, axis=0)

for jj in range(med_rat_p3d.shape[0]):
    for key in conv.keys():
        ii = conv[key]
        
        outs["p3d_panel" + str(jj) + "_" + key + "_x"] = knew[:, ii]
        outs["p3d_panel" + str(jj) + "_" + key + "_y"] = med_rat_p3d[jj, :, ii]
    
    outs["p1d_panel" + str(jj) + "_x"] = k1d_Mpc
    outs["p1d_panel" + str(jj) + "_y"] = med_rat_p1d[jj]


# %%
import forestflow
path_forestflow= os.path.dirname(forestflow.__path__[0]) + "/"
folder = path_forestflow + "data/figures_machine_readable/"
np.save(folder + "fig5b", outs)

# %%
res = np.load(folder + "fig6b.npy", allow_pickle=True).item()
res.keys()

# %%
