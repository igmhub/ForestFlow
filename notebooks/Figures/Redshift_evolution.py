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
# # Redshift evolution of Arinyo parameters

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
# training_type = "Arinyo_min_q1"
# model_path = path_program + "/data/emulator_models/mpg_q1/mpg_hypercube.pt"

training_type = "Arinyo_min_q1_q2"
model_path=path_program + "/data/emulator_models/mpg_q1_q2/mpg_hypercube.pt"
# model_path=path_program + "/data/emulator_models/mpg_hypercube.pt"

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
    # Nrealizations=10000,
    Nrealizations=100,
    training_type=training_type,
    model_path=model_path,
)


# %% [markdown]
# ## LOAD CENTRAL SIMULATION

# %%
sim_label = "mpg_central"
central = Archive3D.get_testing_data(
    sim_label, force_recompute_plin=False
)

sim_label = "mpg_seed"
seed = Archive3D.get_testing_data(
    sim_label, force_recompute_plin=False
)

# %%
Arinyo_coeffs_central = np.array(
    [list(central[i][training_type].values()) for i in range(len(central))]
)

Arinyo_coeffs_seed = np.array(
    [list(seed[i][training_type].values()) for i in range(len(seed))]
)


# %%
Arinyo_central = []
Arinyo_seed = []
for ii in range(Arinyo_coeffs_central.shape[0]):
    dict_params = params_numpy2dict(Arinyo_coeffs_central[ii])
    new_params = transform_arinyo_params(dict_params, central[ii]["f_p"])
    Arinyo_central.append(new_params)
    
    dict_params = params_numpy2dict(Arinyo_coeffs_seed[ii])
    new_params = transform_arinyo_params(dict_params, seed[ii]["f_p"])
    Arinyo_seed.append(new_params)

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
    out = p3d_emu.predict_P3D_Mpc(
        sim_label="mpg_central", 
        z=z,
        emu_params=test_sim_z[0],
        natural_params=True
    )
    Arinyo_emu.append(out["coeffs_Arinyo"])
    Arinyo_emu_std.append(out["coeffs_Arinyo_std"])


# %% [markdown]
# ## PLOT

# %%
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# %%
folder_fig = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"

# %%
ftsize = 20

# Create a 2x1 grid for plotting
fig, ax = plt.subplots(4, 1, figsize=(8, 12), sharex=True, height_ratios=[3, 1, 3, 1])
name_params = ['bias', 'bias_eta', 'q1', 'q2', 'kv', 'av', 'bv', 'kp']
# name_params = list(Arinyo_emu[0].keys())

name2label = {
    'bias':r"$-b_\delta$", 
    'bias_eta':r"$-b_\eta$", 
    'q1':r"$0.5(q_1+q_2)$", 
    'q2':r"$0.5(q_1-q_2)$",
    'kv':r"$k_\mathrm{v}$", 
    'av':r"$a_\mathrm{v}$", 
    'bv':r"$b_\mathrm{v}$", 
    'kp':r"$k_\mathrm{p}$", 
}

# Plot the original and emulator data in the upper panel
for i in range(len(name_params)):
    if(i < 4):
        ax1 = ax[0]
        ax2 = ax[1]
    else:
        ax1 = ax[2]
        ax2 = ax[3]
    col = "C"+str(i)
    if(name_params[i] == "q1"):
        ari_emu1 = np.array([d["q1"] for d in Arinyo_emu])       
        ari_emu2 = np.array([d["q2"] for d in Arinyo_emu])
        ari_emu = 0.5*(ari_emu1 + ari_emu2)
        
        ari_emu_std1 = np.array([d["q1"] for d in Arinyo_emu_std])
        ari_emu_std2 = np.array([d["q2"] for d in Arinyo_emu_std])
        ari_emu_std = 0.5*np.sqrt(ari_emu_std1**2 + ari_emu_std2**2)
        
        ari_cen1 = np.array([d["q1"] for d in Arinyo_central])
        ari_cen2 = np.array([d["q2"] for d in Arinyo_central])
        ari_cen = 0.5*(ari_cen1 + ari_cen2)
        
        ari_cen1 = np.array([d["q1"] for d in Arinyo_seed])
        ari_cen2 = np.array([d["q2"] for d in Arinyo_seed])
        ari_seed = 0.5*(ari_cen1 + ari_cen2)
    elif(name_params[i] == "q2"):
        ari_emu1 = np.array([d["q1"] for d in Arinyo_emu])       
        ari_emu2 = np.array([d["q2"] for d in Arinyo_emu])
        ari_emu = 0.5*(ari_emu1 - ari_emu2)
        
        ari_emu_std1 = np.array([d["q1"] for d in Arinyo_emu_std])
        ari_emu_std2 = np.array([d["q2"] for d in Arinyo_emu_std])
        ari_emu_std = 0.5*np.sqrt(ari_emu_std1**2 + ari_emu_std2**2)
        
        ari_cen1 = np.array([d["q1"] for d in Arinyo_central])
        ari_cen2 = np.array([d["q2"] for d in Arinyo_central])
        ari_cen = 0.5*(ari_cen1 - ari_cen2)
        
        ari_cen1 = np.array([d["q1"] for d in Arinyo_seed])
        ari_cen2 = np.array([d["q2"] for d in Arinyo_seed])
        ari_seed = 0.5*(ari_cen1 - ari_cen2)
    else:
        ari_emu = np.array([d[name_params[i]] for d in Arinyo_emu])
        ari_emu_std = np.array([d[name_params[i]] for d in Arinyo_emu_std])
        ari_cen = np.array([d[name_params[i]] for d in Arinyo_central])
        ari_seed = np.array([d[name_params[i]] for d in Arinyo_seed])

    print(name_params[i])
    # print(np.mean(np.abs(ari_emu)/np.abs(ari_cen)-1))
    # print(np.std(np.abs(ari_emu)/np.abs(ari_cen)-1))
    # if i != 6:
    ax1.plot(
        z_central,
        np.abs(ari_cen),
        "--",
        color=col,
        lw=2
    )
    ax1.plot(
        z_central,
        np.abs(ari_seed),
        "-.",
        color=col,
        lw=2
    )
    
    ax1.plot(
        z_central,
        np.abs(ari_emu),
        color=col,
        ls="-",
    )    
    ax1.fill_between(
        z_central, 
        np.abs(ari_emu)-0.5*ari_emu_std, 
        np.abs(ari_emu)+0.5*ari_emu_std,
        color=col,
        alpha=0.2
    )

    ax2.plot(z_central,
        np.abs(ari_cen)/np.abs(ari_emu)-1,
        "--",
        color=col,
        lw=2,
        alpha=0.8
    )
    ax2.fill_between(
        z_central, 
        np.abs(ari_cen)/(np.abs(ari_emu)-0.5*ari_emu_std)-1, 
        np.abs(ari_cen)/(np.abs(ari_emu)+0.5*ari_emu_std)-1,
        color=col,
        alpha=0.2
    )
    
    ax2.plot(z_central,
        np.abs(ari_seed)/np.abs(ari_emu)-1,
        "-.",
        color=col,
        lw=2,
        alpha=0.8
    )
    ax2.fill_between(
        z_central, 
        np.abs(ari_seed)/(np.abs(ari_emu)-0.5*ari_emu_std)-1, 
        np.abs(ari_seed)/(np.abs(ari_emu)+0.5*ari_emu_std)-1,
        color=col,
        alpha=0.2
    )
    
    
    # ax2.plot(z_central, np.abs(ari_cen)
    # / np.abs(ari_emu)
    # - 1, color=colors[i], ls="-")

for ii in range(0,4,2):
    ax[ii].set_ylabel("Parameter", fontsize=ftsize)
    ax[ii].set_yscale("log")
    ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)


for ii in range(1,5,2):
    ax[ii].set_ylabel("Residual", fontsize=ftsize)
    ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)
    ax[ii].set_ylim(-0.5, 0.5)

ax[0].set_ylim(8e-2, 2.5)
ax[2].set_ylim(0.02, 25)
    
ax[-1].set_xlabel("$z$", fontsize=ftsize)


hand = []
for i in range(4):
    col = "C"+str(i)
    hand.append(mpatches.Patch(color=col, label=name2label[name_params[i]]))
legend1 = ax[0].legend(fontsize=ftsize-2, loc="lower right", handles=hand, ncols=2)

line1 = Line2D([0], [0], label='Best fit to data', color='k', ls="", marker="o")
line2 = Line2D([0], [0], label='ForestFlow', color='k', ls="-")
hand = [line1, line2]
ax[0].legend(fontsize=ftsize-2, loc="upper left", handles=hand, ncols=2)
ax[0].add_artist(legend1)

hand = []
for i in range(4, len(name_params)):
    col = "C"+str(i)
    hand.append(mpatches.Patch(color=col, label=name2label[name_params[i]]))
legend1 = ax[2].legend(fontsize=ftsize-2, loc="lower right", handles=hand, ncols=4)

# plt.gca().add_artist(legend1)
# Adjust layout
plt.tight_layout()

# plt.savefig(folder_fig+"arinyo_z.png")
# plt.savefig(folder_fig+"arinyo_z.pdf")


# %%
