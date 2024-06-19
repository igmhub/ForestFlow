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
# # Central simulation at z=3, l10 test simulations

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
from forestflow.plots.test_sims import (
    plot_p1d_test_sims, 
    plot_p3d_test_sims, 
    plot_p1d_snap,
    plot_p3d_snap
)
from forestflow.utils import params_numpy2dict
from forestflow.rebin_p3d import p3d_allkmu, get_p3d_modes, p3d_rebin_mu

from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"


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
folder_interp = path_program + "/data/plin_interp/"

Archive3D = GadgetArchive3D(
    base_folder=path_program[:-1],
    folder_data=folder_lya_data,
    force_recompute_plin=False,
    average="both",
)
print(len(Archive3D.training_data))


# %% [markdown]
# ## Load emulator

# %%
# training_type = "Arinyo_min_q1"
# training_type = "Arinyo_min_q1_q2"
training_type = "Arinyo_min"
# training_type = "Arinyo_minz"

if (training_type == "Arinyo_min_q1"):
    nparams = 7
    model_path = path_program+"/data/emulator_models/mpg_q1/mpg_hypercube.pt"
elif(training_type == "Arinyo_min"):
    nparams = 8
    # model_path = path_program+"/data/emulator_models/mpg_q1_q2/mpg_hypercube.pt"
    model_path=path_program+"/data/emulator_models/mpg_joint.pt"
elif(training_type == "Arinyo_minz"):
    nparams = 8
    # model_path = path_program+"/data/emulator_models/mpg_q1_q2/mpg_hypercube.pt"
    model_path=path_program+"/data/emulator_models/mpg_jointz.pt"

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
    Nrealizations=10000,
    training_type=training_type,
    model_path=model_path,
    # save_path=model_path,
)

# %% [markdown]
# #### General stuff

# %%
Nsim = 30
zs = np.flip(np.arange(2, 4.6, 0.25))
Nz = zs.shape[0]

n_mubins = 4
kmax_3d_plot = 4
kmax_1d_plot = 4
kmax_fit = 3

sim = Archive3D.training_data[0]

k3d_Mpc = sim['k3d_Mpc']
mu3d = sim['mu3d']
p3d_Mpc = sim['p3d_Mpc']
kmu_modes = get_p3d_modes(kmax_3d_plot)

mask_3d = k3d_Mpc[:, 0] <= kmax_3d_plot

mask_1d = (sim['k_Mpc'] <= kmax_1d_plot) & (sim['k_Mpc'] > 0)
k1d_Mpc = sim['k_Mpc'][mask_1d]
p1d_Mpc = sim['p1d_Mpc'][mask_1d]

# %% [markdown]
# ### Central simulation

# %%
zcen = 3

info_power = {
    "sim_label": "mpg_central",
    "k3d_Mpc": k3d_Mpc[mask_3d, :],
    "mu": mu3d[mask_3d, :],
    "kmu_modes": kmu_modes,
    "k1d_Mpc": k1d_Mpc,
    "return_p3d": True,
    "return_p1d": True,
    "return_cov": True,
    "z": zcen,
}

sim_label = info_power["sim_label"]
test_sim = Archive3D.get_testing_data(
    sim_label, force_recompute_plin=False
)
test_sim_z = [d for d in test_sim if d["z"] == info_power["z"]]
emu_params = test_sim_z[0]

out = emulator.evaluate(
    emu_params=emu_params,
    info_power=info_power,
    natural_params=True,
    Nrealizations=100
)

# %% [markdown]
# #### Rebin data

# %%
_ = p3d_rebin_mu(out["k_Mpc"], out["mu"], test_sim_z[0]["p3d_Mpc"][mask_3d], kmu_modes, n_mubins=n_mubins)
knew, munew, rebin_p3d_sim, mu_bins = _

_ = p3d_rebin_mu(out["k_Mpc"], out["mu"], out["p3d"], kmu_modes, n_mubins=n_mubins)
knew, munew, rebin_p3d_emu, mu_bins = _

_ = p3d_rebin_mu(out["k_Mpc"], out["mu"], out["p3d_std"], kmu_modes, n_mubins=n_mubins)
knew, munew, rebin_p3d_std_emu, mu_bins = _

_ = p3d_rebin_mu(out["k_Mpc"], out["mu"], out["Plin"], kmu_modes, n_mubins=n_mubins)
knew, munew, rebin_plin, mu_bins = _

# %%
norm_p1d = out["k1d_Mpc"]/np.pi
p1d_emu = norm_p1d * out["p1d"]
p1d_std_emu = norm_p1d * out["p1d_std"]
p1d_sim = norm_p1d * test_sim_z[0]["p1d_Mpc"][mask_1d]

# %%
folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"
plot_p3d_snap(
    folder, 
    knew, 
    munew,
    rebin_p3d_sim/rebin_plin,
    rebin_p3d_emu/rebin_plin,
    rebin_p3d_std_emu/rebin_plin,
    mu_bins,
)

# %%
plot_p1d_snap(
    folder, 
    out["k1d_Mpc"], 
    p1d_sim,
    p1d_emu,
    p1d_std_emu,
)

# %% [markdown]
# ## TEST SIMULATIONS

# %%
sim_labels = [
    "mpg_central",    
    "mpg_seed",
    "mpg_growth",
    "mpg_neutrinos",
    "mpg_curved",
    # "mpg_running",
    "mpg_reio",
]

# %%
arr_p3d_sim = np.zeros((len(sim_labels), Nz, np.sum(mask_3d), n_mubins))
arr_p3d_emu = np.zeros((len(sim_labels), Nz, np.sum(mask_3d), n_mubins))
arr_p1d_sim = np.zeros((len(sim_labels), Nz, np.sum(mask_1d)))
arr_p1d_emu = np.zeros((len(sim_labels), Nz, np.sum(mask_1d)))

for isim, sim_label in enumerate(sim_labels):    
    test_sim = Archive3D.get_testing_data(
        sim_label, force_recompute_plin=False
    )

    z_grid = [d["z"] for d in test_sim]
    for iz, z in enumerate(z_grid):
        print(sim_label, z)
        test_sim_z = [d for d in test_sim if d["z"] == z]

        info_power = {
            "sim_label": sim_label,
            "k3d_Mpc": k3d_Mpc[mask_3d, :],
            "mu": mu3d[mask_3d, :],
            "kmu_modes": kmu_modes,
            "k1d_Mpc": k1d_Mpc,
            "return_p3d": True,
            "return_p1d": True,
            # "return_cov": True,
            "z": z,
        }

        emu_params = test_sim_z[0]
        
        out = emulator.evaluate(
            emu_params=emu_params,
            info_power=info_power,
            # natural_params=True,
            Nrealizations=100
        )
        
        _ = p3d_rebin_mu(out["k_Mpc"], out["mu"], test_sim_z[0]["p3d_Mpc"][mask_3d], kmu_modes, n_mubins=n_mubins)
        knew, munew, arr_p3d_sim[isim, iz], mu_bins = _
        
        _ = p3d_rebin_mu(out["k_Mpc"], out["mu"], out["p3d"], kmu_modes, n_mubins=n_mubins)
        knew, munew, arr_p3d_emu[isim, iz], mu_bins = _
        
        arr_p1d_emu[isim, iz] = out["p1d"]
        arr_p1d_sim[isim, iz] = test_sim_z[0]["p1d_Mpc"][mask_1d]

# %%
rat_p3d = arr_p3d_emu/arr_p3d_sim - 1
rat_p1d = arr_p1d_emu/arr_p1d_sim - 1

# %%
folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"

# %%
savename = folder + "test_cosmo/test_cosmo_P3D"
for ext in [".png", ".pdf"]:
    plot_p3d_test_sims(
        sim_labels,
        knew,
        munew,
        rat_p3d,
        mu_bins=mu_bins,
        savename=savename+ext,
        fontsize=20,
    )

# %%
rat_p1d.shape

# %%
savename = folder + "test_cosmo/test_cosmo_P1D"
for ext in [".png", ".pdf"]:
    plot_p1d_test_sims(
        sim_labels,
        out["k1d_Mpc"],
        rat_p1d,
        savename=savename+ext,
        fontsize=20,
    );

# %%

# %%
