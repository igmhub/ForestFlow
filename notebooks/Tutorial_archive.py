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
#     display_name: forestflow
#     language: python
#     name: forestflow
# ---

# %% [markdown]
# # TUTORIAL FOR THE P3D EMULATOR (forestflow)

# %%
import sys
import os
import matplotlib.pyplot as plt

# %%
from forestflow.archive import GadgetArchive3D


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
folder_lya_data = path_program + "/data/best_arinyo/"

Archive3D = GadgetArchive3D(
    base_folder=path_program[:-1],
    folder_data=folder_lya_data,
    force_recompute_plin=True,
    average="both",
)
print(len(Archive3D.training_data))


# %% [markdown]
# ## GET TRAINING DATA

# %% [markdown]
# #### This consists of 30 simulations, with 11 snapshots per simulation within 2 < z < 4.5 and 5 mean-flux rescalings per snapshot

# %%
training_data = Archive3D.get_training_data(Archive3D.emu_params)

# %% [markdown]
# ## GET TESTING DATA

# %% [markdown]
# #### There are 6 test simulations:
# #### - central simulations (mpg_central): simulations centered at the training Latin hypercube simulations
# #### - seed simulations (mpg_seed): simulations centered at the training Latin hypercube simulations with different initial conditions
# #### - growth simulation (mpg_growth): Simulation with a different growth rate than the training simulations
# #### - neutrinos simulation (mpg_neutrinos): Simulations with massive neutrinos
# #### - running simulations (mpg_running): Simulation with a different running of the spectral index.
# #### - reionization simulations (mpg_reio): Simulation with a different HeII reionization history

# %%
central = Archive3D.get_testing_data("mpg_central", force_recompute_plin=True)

# %%
neutrinos = Archive3D.get_testing_data(
    "mpg_neutrinos", force_recompute_plin=True
)

# %%

# %%
