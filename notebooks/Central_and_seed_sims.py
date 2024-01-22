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
#     display_name: ForestFlow
#     language: python
#     name: forestflow
# ---

# %% [markdown]
# # NOTEBOOK PRODUCING FIGURE X, Y P3D PAPER
#

# %%
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# %%
from forestflow.archive import GadgetArchive3D
from forestflow.plots_v0 import plot_err_uncertainty
from forestflow.P3D_cINN import P3DEmulator
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
    Nrealizations=1000,
    model_path='../data/emulator_models/mpg_hypercube.pt'
)

# %% [markdown]
# ## PLOT TEST SIMULATION AT z=z_test

# %%
sim_label = 'mpg_central'
z_test = 3.0
val_scaling=1.0

# %%
plot_err_uncertainty(archive = Archive3D,
                     emulator=p3d_emu, 
                    sim_labels=['mpg_central','mpg_seed'],
                     mu_lims_p3d = [0.31,0.38],
                     z=z_test, 
                     val_scaling=1.0, 
                     colors = ['deepskyblue', 'goldenrod'])


# %%
