# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: lace
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Compare paper emu

# %%
# %load_ext autoreload
# %autoreload 2

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

import forestflow
from forestflow.archive import GadgetArchive3D
from forestflow.P3D_cINN import P3DEmulator
from forestflow.old_emu.paper_P3D_cINN import P3DEmulator as old_P3DEmulator

path_program = os.path.dirname(forestflow.__path__[0]) + '/'
path_program

# %% [markdown]
# #### Load new emu

# %%
p3d_emu = P3DEmulator(
    model_path=os.path.join(
        os.path.dirname(forestflow.__path__[0]),
        "data",
        "emulator_models",
        "forest_mpg",
    )
)

# %% [markdown]
# #### Load old emu

# %%
# %%time
Archive3D = GadgetArchive3D()
print(len(Archive3D.training_data))

# %%
old_p3d_emu = old_P3DEmulator(
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
    training_type='Arinyo_min',
    model_path=path_program+"/data/emulator_models/mpg_hypercube.pt",
)

# %%
old_p3d_emu = old_P3DEmulator(
    Archive3D.training_data,
    Archive3D.emu_params,
    nLayers_inn=12,
    Archive=Archive3D,
    Nrealizations=2000,
    training_type='Arinyo_min',
    model_path=os.path.join(
        os.path.dirname(forestflow.__path__[0]),
        "data",
        "emulator_models",
        "mpg_hypercube.pt",
    )
)

# %%
old_p3d_emu.Arinyo_params

# %%
list_input_params = [
    {'Delta2_p': 0.18489945277410613,
      'n_p': -2.331713201486465,
      'mF': 0.23475637218289533,
      'sigT_Mpc': 0.10040737452608385,
      'gamma': 1.2115605945334802,
      'kF_Mpc': 14.191866950067904},
     {'Delta2_p': 0.20276666703485943,
      'n_p': -2.3317132064538915,
      'mF': 0.310236058401032,
      'sigT_Mpc': 0.10751395885731446,
      'gamma': 1.2059890102644482,
      'kF_Mpc': 13.177851268715806},
]

# %%
# %%time
coeffs_all, coeffs_mean = p3d_emu.predict_Arinyos(
    emu_params=list_input_params,
    return_all_realizations=True,
    Nrealizations=2000,
    seed=0,
)
coeffs_mean

# %%
# %%time

for ii in range(2):
    coeffs_all, coeffs_mean = old_p3d_emu.predict_Arinyos(
        emu_params=list_input_params[ii],
        return_all_realizations=True,
        Nrealizations=2000,
        seed=0
    )
    print(coeffs_mean)
# %%
