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
# # NOTEBOOK PRODUCING FIGURE X, Y P3D PAPER
#

# %%
# %load_ext autoreload
# %autoreload 2


import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from corner import corner

import matplotlib

plt.rc("text", usetex=False)
plt.rcParams["font.family"] = "serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"

from forestflow.archive import GadgetArchive3D
from forestflow.plots_v0 import plot_err_uncertainty
from forestflow.P3D_cINN import P3DEmulator
from forestflow.utils import load_Arinyo_chains

# from forestflow.model_p3d_arinyo import ArinyoModel
# from forestflow import model_p3d_arinyo
# from forestflow.likelihood import Likelihood

# %%
def ls_level(folder, nlevels):
    for ii in range(nlevels):
        folder = os.path.dirname(folder)
    folder += "/"
    return folder


path_program = ls_level(os.getcwd(), 2)
print(path_program)
sys.path.append(path_program)


# %%
def sort_dict(dct, keys):
    """
    Sort a list of dictionaries based on specified keys.

    Args:
        dct (list): List of dictionaries to be sorted.
        keys (list): List of keys to sort the dictionaries by.

    Returns:
        list: The sorted list of dictionaries.
    """
    for d in dct:
        sorted_d = {
            k: d[k] for k in keys
        }  # create a new dictionary with only the specified keys
        d.clear()  # remove all items from the original dictionary
        d.update(
            sorted_d
        )  # update the original dictionary with the sorted dictionary
    return dct


# %% [markdown]
# ## LOAD P3D ARCHIVE

# %%
# %%time
folder_lya_data = path_program + "/data/best_arinyo/"
# folder_interp = path_program+"/data/plin_interp/"
# folder_chains = "/pscratch/sd/l/lcabayol/P3D/p3d_fits_new/"

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
training_type = "Arinyo_min_q1"
model_path = path_program + "/data/emulator_models/mpg_q1/mpg_hypercube.pt"

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
    Nrealizations=50000,
    training_type=training_type,
    model_path=model_path,
)

# %% [markdown]
# ## PLOT TEST SIMULATION AT z=3

# %%
z_use = 3.0
central = Archive3D.get_testing_data(sim_label="mpg_central")
central_z3 = [d for d in central if d["z"] == z_use]

# %%
cosmo_central = [
    {
        key: value
        for key, value in central_z3[i].items()
        if key in Archive3D.emu_params
    }
    for i in range(len(central_z3))
]

# %%
condition_central = sort_dict(cosmo_central, Archive3D.emu_params)

# %%
Arinyo_coeffs_central = central_z3[0]["Arinyo"]

# %%
Arinyo_preds, Arinyo_preds_mean = p3d_emu.predict_Arinyos(
    central_z3[0], return_all_realizations=True
)
print(Arinyo_preds.shape)


# %%

from forestflow.utils import load_Arinyo_chains

# %%
folder_chains = path_program + "/data/mcmc/"
mcmc_chains = load_Arinyo_chains(Archive3D, folder_chains, sim_label="mpg_central", z=z_use, chain_samp=240000, training_type=training_type)

# %%
mcmc_chains.shape

# %% [markdown]
# #### transform params

# %%
from forestflow.utils import (
    # get_covariance,
    # sort_dict,
    params_numpy2dict,
    transform_arinyo_params,
)

# %%
if(training_type == "Arinyo_min_q1"):
    param_order = np.array([0, 1, 2, 3, 4, 5, 6], dtype=int)
else:
    param_order = np.array([0, 1, 2, 7, 3, 4, 5, 6], dtype=int)

arinyo_emu_natural = np.zeros_like(Arinyo_preds)
arinyo_sim_natural = np.zeros_like(mcmc_chains)

for ii in range(Arinyo_preds.shape[0]):
    _par = transform_arinyo_params(
        params_numpy2dict(Arinyo_preds[ii]), central_z3[0]["f_p"]
    )
    arinyo_emu_natural[ii] = np.array(list(_par.values()))


    _par = transform_arinyo_params(
        params_numpy2dict(mcmc_chains[ii]), central_z3[0]["f_p"]
    )
    arinyo_sim_natural[ii] = np.array(list(_par.values()))

_par = transform_arinyo_params(
    Arinyo_coeffs_central, central_z3[0]["f_p"]
)
arinyo_mle_natural = np.array(list(_par.values()))

arinyo_emu_natural[:,0] = -np.abs(arinyo_emu_natural[:,0])
arinyo_emu_natural[:,1] = -np.abs(arinyo_emu_natural[:,1])
arinyo_sim_natural[:,0] = -np.abs(arinyo_sim_natural[:,0])
arinyo_sim_natural[:,1] = -np.abs(arinyo_sim_natural[:,1])

arinyo_emu_natural = arinyo_emu_natural[:, param_order]
arinyo_sim_natural = arinyo_sim_natural[:, param_order]
arinyo_mle_natural = arinyo_mle_natural[param_order]

_ = np.argwhere((np.isfinite(arinyo_sim_natural[:,4]) == True) & (arinyo_sim_natural[:,4] > 1e-2)  & (arinyo_sim_natural[:,4] < 10))[:,0]
arinyo_sim_natural = arinyo_sim_natural[_]

# %%
range_fig = np.percentile(arinyo_sim_natural, [0.01, 99.99], axis=0)
range_use = []
for ii in range(len(param_order)):
    range_use.append(tuple(range_fig.T[ii]))
range_use

# %%
std = np.std(arinyo_sim_natural, axis=0)
std = 0.5*(np.percentile(arinyo_sim_natural, 68, axis=0) - np.percentile(arinyo_sim_natural, 16, axis=0))

arinyo_mle_natural/std

# %%
5.2 and 4.4

# %%
folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"
if(training_type == "Arinyo_min_q1"):
    labs = [
        r"$b_\delta$",
        r"$b_\eta$",
        "$q_1$",
        "$k_\mathrm{v}$",
        "$a_\mathrm{v}$",
        "$b_\mathrm{v}$",
        "$k_\mathrm{p}$",
    ]
else:
    labs = [
        r"$b_\delta$",
        r"$b_\eta$",
        "$q_1$",
        "$q_2$",
        "$k_\mathrm{v}$",
        "$a_\mathrm{v}$",
        "$b_\mathrm{v}$",
        "$k_\mathrm{p}$",
    ]

corner_plot = corner(
    arinyo_sim_natural,
    labels=labs,
    truths=list(arinyo_mle_natural),
    truth_color="C2",
    color="C1",
    range=range_use,
    plot_density=False,
    hist_bin_factor=1,
    levels=(0.68, 0.95),
    hist_kwargs=dict(density=True, linewidth=2, log=True),
    labelpad=0.25,
    contour_kwargs=dict(linewidths=4)
)

corner(
    arinyo_emu_natural, 
    fig=corner_plot, 
    color="C0", 
    smooth=True,
    range=range_use,
    plot_density=False,
    hist_bin_factor=4,
    levels=(0.68, 0.95),
    hist_kwargs=dict(density=True, linewidth=2, alpha=0.75, log=True),
    contour_kwargs=dict(linewidths=4)
)

# corner_plot.suptitle(f"Contours for central simulation at $z$=3", fontsize=25)
# Increase the label font size for this plot

ftsize1 = 45
ftsize2 = 30
axes = corner_plot.get_axes()
for ax in axes:
    ax.xaxis.label.set_fontsize(ftsize1)
    ax.yaxis.label.set_fontsize(ftsize1)
    ax.xaxis.set_tick_params(labelsize=ftsize2)
    ax.yaxis.set_tick_params(labelsize=ftsize2)

black_line = Line2D([0], [0], color="C2", lw=10, label="Best fit")
blue_patch = mpatches.Patch(color="C1", label="Posterior")
red_patch = mpatches.Patch(color="C0", label="ForestFlow")

axes[7].legend(
    handles=[red_patch, blue_patch, black_line],
    bbox_to_anchor=(1, 0.7),
    fontsize=ftsize1,
)
axes[0].set_ylabel("$\log N$", fontsize=30)
# plt.savefig(folder+"contours_central_z3_q1.pdf", bbox_inches="tight")
plt.savefig(folder+"contours_central_z3_q1_q2.pdf", bbox_inches="tight")

# %%

# %%

# %%
