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
# # Leave-simulation-out

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
from forestflow.archive import GadgetArchive3D
Archive3D = GadgetArchive3D()


# %% [markdown]
# ### Train L1Os
#
# Trained in Tutorial l1O_cov

# %%
do_training = False

if do_training:
    model_path = path_program+"/data/emulator_models/"
    training_type = "Arinyo_min"
    for s in range(1, 30):
        print(f"Starting simulation {s}")
        print()
    
        training_data = [
            d for d in Archive3D.training_data if d["sim_label"] != f"mpg_{s}"
        ]
    
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
            save_path=model_path + "mpg_drop"+str(s)+".pt",
        )

# %% [markdown]
# ### Evaluate L1Os

# %%
training_type = "Arinyo_min"
model_path = path_program + "/data/emulator_models/"

Nsim = 30
zs = np.flip(np.arange(2, 4.6, 0.25))
zs = zs[1:]
Nz = zs.shape[0]

n_mubins = 4
kmax_3d_fit = 5
kmax_1d_fit = 4
kmax_3d_plot = kmax_3d_fit + 1
kmax_1d_plot = kmax_1d_fit + 1

sim = Archive3D.training_data[0]

k3d_Mpc = sim["k3d_Mpc"]
mu3d = sim["mu3d"]
p3d_Mpc = sim["p3d_Mpc"]
kmu_modes = get_p3d_modes(kmax_3d_plot)

mask_3d = k3d_Mpc[:, 0] <= kmax_3d_plot

mask_1d = (sim["k_Mpc"] <= kmax_1d_plot) & (sim["k_Mpc"] > 0)
k1d_Mpc = sim["k_Mpc"][mask_1d]
p1d_Mpc = sim["p1d_Mpc"][mask_1d]

sim = Archive3D.training_data[0]
_ = p3d_rebin_mu(
    k3d_Mpc[mask_3d],
    mu3d[mask_3d],
    sim["p3d_Mpc"][mask_3d],
    kmu_modes,
    n_mubins=n_mubins,
)
knew, munew, p3d_measured, mu_bins = _

# %%
from lace.cosmo import cosmology

# %%
arr_p3d_sim = np.zeros((Nsim, Nz, np.sum(mask_3d), n_mubins))
arr_p3d_emu = np.zeros((Nsim, Nz, np.sum(mask_3d), n_mubins))
arr_p1d_sim = np.zeros((Nsim, Nz, np.sum(mask_1d)))
arr_p1d_emu = np.zeros((Nsim, Nz, np.sum(mask_1d)))
params_sim = np.zeros((Nsim, Nz, 3))
params_emu = np.zeros((Nsim, Nz, 3))

for isim in range(Nsim):
    sim_label = f"mpg_{isim}"
    print(f"Starting simulation {isim}")
    print()

    name_emu = "l1O/forest_mpg_l1O_" + str(isim)
    emulator = P3DEmulator(key=name_emu)

    for iz, z in enumerate(zs):
        print(z)
        # define test sim
        dict_sim = [
            d
            for d in Archive3D.training_data
            if d["z"] == z and d["sim_label"] == sim_label and d["val_scaling"] == 1
        ]

        info_power = {
            "sim_label": sim_label,
            "z": z,
            "k3d_Mpc": k3d_Mpc[mask_3d, :],
            "mu": mu3d[mask_3d, :],
            "kmu_modes": kmu_modes,
            "k1d_Mpc": k1d_Mpc,
        }

        input_emu = {}
        for par in emulator.input_labels:
            input_emu[par] = dict_sim[0][par]
        out = emulator.evaluate(emu_params=input_emu)

        params_emu[isim, iz, 0] = out["bias"]
        params_emu[isim, iz, 1] = out["bias_eta"]
        params_emu[isim, iz, 2] = 0

        params_sim[isim, iz, 0] = dict_sim[0][training_type]["bias"]
        params_sim[isim, iz, 1] = dict_sim[0][training_type]["bias_eta"]
        params_sim[isim, iz, 2] = dict_sim[0][training_type]["beta"]

        if iz == 0:
            cosmo_params_dict = dict_sim[0]["cosmo_params"]
            fid_cosmo = cosmology.Cosmology(cosmo_params_dict=cosmo_params_dict)
            model_Arinyo = ArinyoModel(fid_cosmo)

        p3d_emu = model_Arinyo.P3D_Mpc_k_mu(
            info_power["z"], info_power["k3d_Mpc"], info_power["mu"], out
        )
        p1d_emu = model_Arinyo.P1D_Mpc(info_power["z"], info_power["k1d_Mpc"], out)

        _ = p3d_rebin_mu(
            info_power["k3d_Mpc"],
            info_power["mu"],
            dict_sim[0]["p3d_Mpc"][mask_3d],
            kmu_modes,
            n_mubins=n_mubins,
        )
        knew, munew, arr_p3d_sim[isim, iz], mu_bins = _
        _ = p3d_rebin_mu(
            info_power["k3d_Mpc"],
            info_power["mu"],
            p3d_emu,
            kmu_modes,
            n_mubins=n_mubins,
        )
        knew, munew, arr_p3d_emu[isim, iz], mu_bins = _

        arr_p1d_emu[isim, iz] = p1d_emu
        arr_p1d_sim[isim, iz] = dict_sim[0]["p1d_Mpc"][mask_1d]

    p3d_emu = 0
    # break


# %%
# folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"
folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures_new/"
np.savez(
    folder + "temporal_l1O", 
    arr_p3d_sim=arr_p3d_sim, 
    arr_p3d_emu=arr_p3d_emu, 
    arr_p1d_sim=arr_p1d_sim, 
    arr_p1d_emu=arr_p1d_emu,
    params_sim=params_sim,
    params_emu=params_emu
)

# %%
# folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"
folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures_new/"
fil = np.load(folder + "temporal_l1O.npz")
params_emu = fil["params_emu"]
params_sim = fil["params_sim"]
arr_p3d_sim = fil["arr_p3d_sim"]
arr_p3d_emu = fil["arr_p3d_emu"]
arr_p1d_sim = fil["arr_p1d_sim"]
arr_p1d_emu = fil["arr_p1d_emu"]

# %%
for ii in range(2):
    y = np.percentile(params_emu[..., ii] / params_sim[..., ii] - 1, [50, 16, 84])
    print(y[0]*100)
    print(0.5*(y[2] - y[1])*100)

# %%
kaiser_emu = np.zeros((params_emu.shape[0], params_emu.shape[1], 2))
kaiser_sim = np.zeros((params_emu.shape[0], params_emu.shape[1], 2))
kaiser_emu[:, :, 0] = params_emu[:, :, 0]**2
kaiser_emu[:, :, 1] = params_emu[:, :, 0]**2 * (1+params_emu[:, :, 2])**2
kaiser_sim[:, :, 0] = params_sim[:, :, 0]**2
kaiser_sim[:, :, 1] = params_sim[:, :, 0]**2 * (1+params_sim[:, :, 2])**2

for ii in range(2):
    y = np.percentile(kaiser_emu[:, :, ii] / kaiser_sim[:, :, ii] - 1, [50, 16, 84])
    print(y[0]*100)
    print(0.5*(y[2] - y[1])*100)

# %%
_ = np.isfinite(knew) & (knew > 0.3) & (knew < 5)
y = np.percentile(arr_p3d_emu[:, :, _]/arr_p3d_sim[:, :, _], [50, 16, 84]) - 1
print(y[0]*100, 0.5*(y[2]-y[1])*100)

# %%
_ = np.isfinite(k1d_Mpc) & (k1d_Mpc < 4)
y = np.percentile(arr_p1d_emu[:, :, _]/arr_p1d_sim[:, :, _], [50, 16, 84]) - 1
print(y[0]*100, 0.5*(y[2]-y[1])*100)

# %% [markdown]
# ### L1O of each sim

# %% [markdown]
# ## PLOTTING

# %%
folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures_new/"
z_use = np.arange(2, 4.25, 0.5)[::-1]

mask_z = np.zeros(len(z_use), dtype=int)
for ii in range(len(z_use)):
    mask_z[ii] = np.argwhere(z_use[ii] == zs)[0,0]
mask_z

# %% [markdown]
# #### P3D

# %%
residual3d = (arr_p3d_emu / arr_p3d_sim -1)

# %%
# savename = folder+"l1O/l1O_P3D.png"
# plot_p3d_L1O(z_use, knew, munew, residual[:, mask_z, :, :], mu_bins, kmax_3d_fit=kmax_fit, savename=savename)
savename = folder+"l1O/l1O_P3D.pdf"
plot_p3d_L1O(z_use, knew, munew, residual3d[:, mask_z, :, :], mu_bins, kmax_3d_fit=kmax_3d_fit, savename=savename, legend=True)


# %%

# %% [markdown]
# #### P1D

# %%
residual1d = (arr_p1d_emu / arr_p1d_sim -1)

# %%
# savename=folder+"l1O/l1O_P1D.png"
# plot_p1d_L1O(z_use, k1d_Mpc, residual[:, mask_z, :], kmax_1d_fit=kmax_fit, savename=savename)
savename=folder+"l1O/l1O_P1D.pdf"
plot_p1d_L1O(z_use, k1d_Mpc, residual1d[:, mask_z, :], kmax_1d_fit=kmax_1d_fit, savename=savename)

# %% [markdown]
# ### Save data for zenodo

# %%
conv = {}
conv["blue"] = 0
conv["orange"] = 1
conv["green"] = 2
conv["red"] = 3
outs = {}

med_rat_p3d = np.median(residual3d[:, mask_z, :, :], axis=0)
med_rat_p1d = np.median(residual1d[:, mask_z, :], axis=0)

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
np.save(folder + "fig5a", outs)

# %%
res = np.load(folder + "fig6a.npy", allow_pickle=True).item()
res.keys()

# %%
