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
# # Precision of forestflow accross parameters space
#
# check precision old, 0, 2, 3

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from forestflow.model_p3d_arinyo import ArinyoModel
from forestflow.archive import get_camb_interp
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


# %%
p3d_emu = P3DEmulator(
    # model_path=path_program+"/data/emulator_models/new_emu",
    # model_path=path_program+"/data/emulator_models/new_emu2",
    model_path=path_program+"/data/emulator_models/new_emu3",
)

# %% [markdown]
# ### Evaluate across parameter space

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

sim = Archive3D.training_data[0]
_ = p3d_rebin_mu(k3d_Mpc[mask_3d], mu3d[mask_3d], sim['p3d_Mpc'][mask_3d], kmu_modes, n_mubins=n_mubins)
knew, munew, p3d_measured, mu_bins = _

# %%
# %%time
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
    
    for iz, z in enumerate(zs):
        print(z)
        # define test sim
        dict_sim = [
            d
            for d in Archive3D.training_data
            if d["z"] == z
            and d["sim_label"] == sim_label
            and d["val_scaling"] == 1
        ]

        if iz == 0:
            cosmo = dict_sim[0]["cosmo_params"]
            pk_interp = get_camb_interp({"cosmo_params": cosmo})
            model_Arinyo = ArinyoModel(camb_pk_interp=pk_interp)

        info_power = {
            # "sim_label": sim_label,
            "k3d_Mpc": k3d_Mpc[mask_3d, :],
            "mu": mu3d[mask_3d, :],
            "kmu_modes": kmu_modes,
            "k1d_Mpc": k1d_Mpc,
            "return_p3d": True,
            "return_p1d": True,
            "z": z,
        }

        input_pars = {}
        for par in p3d_emu.emu_input_names:
            input_pars[par] = dict_sim[0][par]

        out = p3d_emu.evaluate_arinyo(
            input_pars,
            model_Arinyo,
            info_power=info_power,
        )
        
        # # p1d and p3d from sim
        _ = p3d_rebin_mu(out["k_Mpc"], out["mu"], dict_sim[0]["p3d_Mpc"][mask_3d], kmu_modes, n_mubins=n_mubins)
        knew, munew, arr_p3d_sim[isim, iz], mu_bins = _
        
        _ = p3d_rebin_mu(out["k_Mpc"], out["mu"], out["p3d"], kmu_modes, n_mubins=n_mubins)
        knew, munew, arr_p3d_emu[isim, iz], mu_bins = _

        arr_p1d_emu[isim, iz] = out["p1d"]
        arr_p1d_sim[isim, iz] = dict_sim[0]["p1d_Mpc"][mask_1d]

        params_emu[isim, iz, 0] = out['coeffs_Arinyo']["bias"]
        params_emu[isim, iz, 2] = out['coeffs_Arinyo']["beta"]
        _ = transform_arinyo_params(out['coeffs_Arinyo'], dict_sim[0]["f_p"])
        params_emu[isim, iz, 1] = _["bias_eta"]

        params_sim[isim, iz, 0] = dict_sim[0][training_type]["bias"]
        params_sim[isim, iz, 2] = dict_sim[0][training_type]["beta"]
        _ = transform_arinyo_params(dict_sim[0][training_type], dict_sim[0]["f_p"])
        params_sim[isim, iz, 1] = _["bias_eta"]

        # break
        
    # break


# %% [markdown]
# #### first sim

# %%
y = params_emu[0]/params_sim[0]-1
print(np.mean(y, axis=0), np.std(y, axis=0))

# %%
[0.00087759 0.01030584 0.00949894] [0.00473999 0.03392438 0.03584392]

# %%
_ = np.isfinite(knew) & (knew > 0.3) & (knew < 5)
y = arr_p3d_emu[0, :, _]/arr_p3d_sim[0, :, _]-1
print(np.mean(y), np.std(y))

# %%
0.012157139933189776 0.028941480521219257

# %%
_ = np.isfinite(k1d_Mpc) & (k1d_Mpc < 4)
y = arr_p1d_emu[0, :, _]/arr_p1d_sim[0, :, _]-1
print(np.mean(y), np.std(y))

# %%
0.010318689755883593 0.008766082002409253

# %% [markdown]
# #### All sims

# %%
y = params_emu/params_sim-1
print(np.mean(y, axis=(0,1)), np.std(y, axis=(0,1)))

# %%
[0.0023533  0.00685083 0.00457457] [0.00750521 0.03878708 0.0405999 ]

# %%
_ = np.isfinite(knew) & (knew > 0.3) & (knew < 5)
y = arr_p3d_emu[:, :, _]/arr_p3d_sim[:, :, _]-1
print(np.mean(y), np.std(y))

# %%
0.01118326057585371 0.029269027387910655

# %%
_ = np.isfinite(k1d_Mpc) & (k1d_Mpc < 4)
y = arr_p1d_emu[:, :, _]/arr_p1d_sim[:, :, _]-1
print(np.mean(y), np.std(y))

# %%
0.00838753010162733 0.010505141480596628

# %%

# %%

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
folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"
z_use = np.arange(2, 4.5, 0.5)[::-1]

mask_z = np.zeros(len(z_use), dtype=int)
for ii in range(len(z_use)):
    mask_z[ii] = np.argwhere(z_use[ii] == zs)[0,0]
mask_z

# %% [markdown]
# #### P3D

# %%
residual3d = (arr_p3d_emu / arr_p3d_sim -1)
print(np.median(residual3d), np.std(residual3d))

# %%
# savename = folder+"l1O/l1O_P3D.png"
# plot_p3d_L1O(z_use, knew, munew, residual[:, mask_z, :, :], mu_bins, kmax_3d_fit=kmax_fit, savename=savename)
# savename = folder+"l1O/l1O_P3D.pdf"
savename = None
plot_p3d_L1O(z_use, knew, munew, residual3d[:, mask_z, :, :], mu_bins, kmax_3d_fit=kmax_3d_fit, savename=savename, legend=True)


# %%

# %% [markdown]
# #### P1D

# %%
residual1d = (arr_p1d_emu / arr_p1d_sim -1)
print(np.median(residual1d), np.std(residual1d))

# %%
# savename=folder+"l1O/l1O_P1D.png"
# plot_p1d_L1O(z_use, k1d_Mpc, residual[:, mask_z, :], kmax_1d_fit=kmax_fit, savename=savename)
# savename=folder+"l1O/l1O_P1D.pdf"
savename = None
plot_p1d_L1O(z_use, k1d_Mpc, residual1d[:, mask_z, :], kmax_1d_fit=kmax_1d_fit, savename=savename)

# %%
