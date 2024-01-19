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
# # TUTORIAL FOR THE P3D EMULATOR (forestflow)

# %%
import sys
import os
import matplotlib.pyplot as plt

# %%
from forestflow.archive import GadgetArchive3D
from forestflow.plots_v0 import plot_test_p3d
from forestflow.P3D_cINN import P3DEmulator


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
    force_recompute_plin=False,
    average="both",
)
print(len(Archive3D.training_data))


# %% [markdown]
# ## TRAIN EMULATOR

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
    use_chains=False,
    chain_samp=100_000,
    folder_chains="/data/desi/scratch/jchavesm/p3d_fits_new/",
)

# %% [markdown]
# ## TEST EMULATOR

# %%
sim_label = "mpg_central"
ind_book = 6
plot_test_p3d(ind_book, Archive3D, p3d_emu, sim_label)

# %% [markdown]
# ## LOAD TRAINED EMULATOR

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# To load a trained model, one needs to specify the path to the model in the argument 'model_path'.

# %% [markdown]
# The folder '/data/emulator_models/' contains the models trained with all the Latinhypercube simulations: 'mpg_hypercube.pt'

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
    use_chains=False,
    chain_samp=100_000,
    folder_chains="/data/desi/scratch/jchavesm/p3d_fits_new/",
    model_path="../data/emulator_models/mpg_hypercube.pt",
)

# %%
sim_label = "mpg_central"
ind_book = 6
plot_test_p3d(ind_book, Archive3D, p3d_emu, sim_label)

# %% [markdown]
# ## PREDICT P1D AND P3D FOR A TEST SIMULATION

# %% [markdown]
# The available test simulations are:
# sim_labels = ['mpg_central', 'mpg_seed', 'mpg_growth', 'mpg_neutrinos', 'mpg_curved','mpg_running', 'mpg_reio']

# %%
sim_label = "mpg_central"
z_test = 3

# %%
test_sim = central = Archive3D.get_testing_data(
    "mpg_central", force_recompute_plin=True
)
dict_sim = [d for d in test_sim if d["z"] == z_test and d["val_scaling"] == 1]

# %%
p3d_pred, p3d_cov = p3d_emu.predict_P3D_Mpc(
    sim_label="mpg_central", z=z_test, test_sim=dict_sim, return_cov=True
)

# %%
p1d_pred, p1d_cov = p3d_emu.predict_P1D_Mpc(
    sim_label="mpg_central", z=z_test, test_sim=dict_sim, return_cov=True
)

# %%
