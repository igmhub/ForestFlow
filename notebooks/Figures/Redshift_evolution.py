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
training_type = "Arinyo_min"
model_path=path_program + "/data/emulator_models/mpg_last.pt"
training_type = "Arinyo_minz"
model_path=path_program + "/data/emulator_models/mpg_minz.pt"

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
    Nrealizations=1000,
    training_type=training_type,
    # model_path=model_path,
    save_path=model_path,
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
from forestflow.plots.params_z import plot_arinyo_z

# %%
folder_fig = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"

# %%
# for ii in range(len(Arinyo_emu)):
#     print(ii, Arinyo_emu[ii]["kv"])
#     Arinyo_emu[ii]["kv"] = Arinyo_emu[ii]["kv"]**Arinyo_emu[ii]["av"]
#     print(Arinyo_emu[ii]["kv"])

# %%
plot_arinyo_z(z_central, Arinyo_central, Arinyo_seed, Arinyo_emu, Arinyo_emu_std, folder_fig=folder_fig, ftsize=20)

# %%
z_central

# %%


# %%
