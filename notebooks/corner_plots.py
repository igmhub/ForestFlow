# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: emulators2
#     language: python
#     name: emulators2
# ---

# %% [markdown]
# # NOTEBOOK PRODUCING FIGURE X, Y P3D PAPER
#

# %%
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# %%
import matplotlib 

plt.rc('text', usetex=False)
plt.rcParams["font.family"] = "serif"
matplotlib.rcParams['mathtext.fontset'] = 'cm'

# %% jupyter={"outputs_hidden": true}
from forestflow.archive import GadgetArchive3D
from forestflow.plots_v0 import plot_err_uncertainty
from forestflow.P3D_cINN import P3DEmulator
from forestflow.utils import load_Arinyo_chains
#from forestflow.model_p3d_arinyo import ArinyoModel
#from forestflow import model_p3d_arinyo
#from forestflow.likelihood import Likelihood

# %%
def ls_level(folder, nlevels):
    for ii in range(nlevels):
        folder = os.path.dirname(folder)
    folder += "/"
    return folder

path_program = ls_level(os.getcwd(), 1)
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
folder_lya_data = path_program +  "/data/best_arinyo/"
#folder_interp = path_program+"/data/plin_interp/"
folder_chains='/pscratch/sd/l/lcabayol/P3D/p3d_fits_new/'

Archive3D = GadgetArchive3D(
    base_folder=path_program[:-1], 
    folder_data=folder_lya_data, 
    force_recompute_plin=False,
    average='both'
)
print(len(Archive3D.training_data))


# %% [markdown]
# ## TRAIN EMULATOR

# %%
p3d_emu = P3DEmulator(
    Archive3D.training_data,
    Archive3D.emu_params,
    nepochs=300,
    lr=0.001,#0.005
    batch_size=20,
    step_size=200,
    gamma=0.1,
    weight_decay=0,
    adamw=True,
    nLayers_inn=12,#15
    Archive=Archive3D,
    Nrealizations=10_000,
    model_path='../data/emulator_models/mpg_hypercube.pt'
)

# %% [markdown]
# ## PLOT TEST SIMULATION AT z=3

# %%
central = Archive3D.get_testing_data(sim_label='mpg_central')
central_z3 = [d for d in central if d['z']==3]

# %%
cosmo_central = [{key: value
        for key, value in central_z3[i].items()
        if key in Archive3D.emu_params}
    for i in range(len(central_z3))]

# %%
condition_central = sort_dict(
            cosmo_central, Archive3D.emu_params) 

# %%
Arinyo_coeffs_central = central_z3[0]["Arinyo"].values() 

# %%
Arinyo_preds, Arinyo_preds_mean = p3d_emu.predict_Arinyos(central_z3[0],
                                                         return_all_realizations=True)



# %%
corner_plot = corner.corner(Arinyo_preds, 
              labels=[r'$b$', r'$\beta$', '$q_1$', '$k_{vav}$','$a_v$','$b_v$','$k_p$','$q_2$'], 
              truths=list(Arinyo_coeffs_central),
              truth_color='crimson')


corner_plot.suptitle(f"Predicted Arinyo coefficients for central simulation at $z$=3", fontsize=18)
# Increase the label font size for this plot
for ax in corner_plot.get_axes():
    ax.xaxis.label.set_fontsize(18)
    ax.yaxis.label.set_fontsize(18)
    ax.xaxis.set_tick_params(labelsize=12)  
    ax.yaxis.set_tick_params(labelsize=12)
plt.show()

# %% [markdown]
# ## LOAD MCMC CHAINS

# %%
mcmc_chains = load_Arinyo_chains(Archive3D, sim_label='mpg_central', z=3.0)

# %%
corner_plot = corner.corner(mcmc_chains, 
              labels=[r'$b$', r'$\beta$', '$q_1$', '$k_{vav}$','$a_v$','$b_v$','$k_p$','$q_2$'], 
              truths=list(Arinyo_coeffs_central),
              truth_color='crimson')


corner_plot.suptitle(f"Predicted Arinyo coefficients for central simulation at $z$=3", fontsize=18)
# Increase the label font size for this plot
for ax in corner_plot.get_axes():
    ax.xaxis.label.set_fontsize(18)
    ax.yaxis.label.set_fontsize(18)
    ax.xaxis.set_tick_params(labelsize=12)  
    ax.yaxis.set_tick_params(labelsize=12)
plt.show()

# %%


axes[0, 0].legend(handles=[red_patch, blue_patch, green_patch], bbox_to_anchor=(2,0.5))

# %%
axes = corner_plot.get_axes()

# %%

# %%
fig = corner_plot = corner.corner(mcmc_chains, 
              labels=[r'$b$', r'$\beta$', '$q_1$', '$k_{vav}$','$a_v$','$b_v$','$k_p$','$q_2$'], 
              truths=list(Arinyo_coeffs_central),
              truth_color='black',
              color='navy',
              smooth=True)

corner.corner(Arinyo_preds, fig=fig, color='crimson', smooth=True)

corner_plot.suptitle(f"Contours for central simulation at $z$=3", fontsize=25)
# Increase the label font size for this plot

axes = corner_plot.get_axes()
for ax in axes:
    ax.xaxis.label.set_fontsize(28)
    ax.yaxis.label.set_fontsize(28)
    ax.xaxis.set_tick_params(labelsize=15)  
    ax.yaxis.set_tick_params(labelsize=15)
    
red_patch = mpatches.Patch(color='crimson', label='Forest Flow')
blue_patch = mpatches.Patch(color='navy', label='MCMC')
black_line = Line2D([0], [0], color='black', lw=3, label='Best fit MCMC')

axes[0].legend(handles=[red_patch, blue_patch, black_line], bbox_to_anchor=(1,0.7), fontsize=18)
plt.savefig('contours_central_z3.pdf', bbox_inches='tight')
plt.show()

# %%
