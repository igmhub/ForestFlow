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
    Nrealizations=1000,
    model_path=path_program+"/data/emulator_models/mpg_hypercube.pt",
)

# %% [markdown]
# ## LOAD CENTRAL SIMULATION

# %%
sim_label = "mpg_central"

# %%
test_sim = central = Archive3D.get_testing_data(
    sim_label, force_recompute_plin=False
)


# %%
Arinyo_coeffs_central = np.array(
    [list(test_sim[i]["Arinyo"].values()) for i in range(len(test_sim))]
)


# %%
Arinyo_sim = []
for ii in range(Arinyo_coeffs_central.shape[0]):
    dict_params = params_numpy2dict(Arinyo_coeffs_central[ii])
    new_params = transform_arinyo_params(dict_params, test_sim[ii]["f_p"])
    Arinyo_sim.append(new_params)

# %% [markdown]
# ## LOOP OVER REDSHIFTS PREDICTING THE ARINYO PARAMETERS

# %%
z_central = [d["z"] for d in test_sim]

# %%
Arinyo_emu = []
Arinyo_emu_std = []
for iz, z in enumerate(z_central):
    test_sim_z = [d for d in test_sim if d["z"] == z]

    # #testing_condition = p3d_emu._get_test_condition(test_sim_z)
    # Arinyo_mean = p3d_emu.predict_Arinyos(
    #     test_sim_z,
    #     true_coeffs=None,
    #     plot=False,
    #     return_all_realizations=False,
    # )
    out = p3d_emu.predict_P3D_Mpc(
        sim_label=sim_label, 
        z=z, 
        emu_params=test_sim_z[0],
        natural_params=True
    )

    Arinyo_emu.append(out["coeffs_Arinyo"])
    Arinyo_emu_std.append(out["coeffs_Arinyo_std"])


# %%
Arinyo_emu_std

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
fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
name_params = ['bias', 'bias_eta', 'q1', 'q2', 'kv', 'av', 'bv', 'kp']
# name_params = list(Arinyo_emu[0].keys())

name2label = {
    'bias':r"$-b_\delta$", 
    'bias_eta':r"$-b_\eta$", 
    'q1':r"$q_1$", 
    'q2':r"$q_2$",
    'kv':r"$k_\mathrm{v}$", 
    'av':r"$a_\mathrm{v}$", 
    'bv':r"$b_\mathrm{v}$", 
    'kp':r"$k_\mathrm{p}$", 
}

# Plot the original and emulator data in the upper panel
for i in range(len(name_params)):
    if(i < 4):
        ax1 = ax[0]
    else:
        ax1 = ax[1]
    col = "C"+str(i)
    ari_emu = np.array([d[name_params[i]] for d in Arinyo_emu])
    ari_emu_std = np.array([d[name_params[i]] for d in Arinyo_emu_std])
    ari_cen = np.array([d[name_params[i]] for d in Arinyo_sim])

    print(name_params[i])
    print(np.mean(np.abs(ari_emu)/np.abs(ari_cen)-1))
    print(np.std(np.abs(ari_emu)/np.abs(ari_cen)-1))
    # if i != 6:
    ax1.plot(
        z_central,
        np.abs(ari_cen),
        "o:",
        color=col,
        lw=2
        # label=name2label[name_params[i]],
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
    # ax2.plot(z_central, np.abs(ari_cen)
    # / np.abs(ari_emu)
    # - 1, color=colors[i], ls="-")

for ii in range(2):
    ax[ii].set_ylabel("Parameter", fontsize=ftsize)
    ax[ii].set_yscale("log")
    ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)

ax[0].set_ylim(0.02, 2)
ax[1].set_ylim(0.02, 25)
    
ax[-1].set_xlabel("$z$", fontsize=ftsize)


hand = []
for i in range(4):
    col = "C"+str(i)
    hand.append(mpatches.Patch(color=col, label=name2label[name_params[i]]))
legend1 = ax[0].legend(fontsize=ftsize-2, loc="lower left", handles=hand, ncols=4)

line1 = Line2D([0], [0], label='MLE fit', color='k', ls=":", marker="o")
line2 = Line2D([0], [0], label='ForestFlow', color='k', ls="-")
hand = [line1, line2]
ax[0].legend(fontsize=ftsize-2, loc="upper left", handles=hand, ncols=2)
ax[0].add_artist(legend1)

hand = []
for i in range(4, len(name_params)):
    col = "C"+str(i)
    hand.append(mpatches.Patch(color=col, label=name2label[name_params[i]]))
legend1 = ax[1].legend(fontsize=ftsize-2, loc="lower left", handles=hand, ncols=4)

# plt.gca().add_artist(legend1)
# Adjust layout
plt.tight_layout()

plt.savefig(folder_fig+"arinyo_z.png")
plt.savefig(folder_fig+"arinyo_z.pdf")

# Show the plot
# plt.show()

# %%
bias
0.0025521241775302472
0.010315501514549085
bias_eta
0.0007729678046709139
0.03842253599787945

# %%
