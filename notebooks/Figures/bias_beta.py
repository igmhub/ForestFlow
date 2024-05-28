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
# # Compare bias-beta with observations

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
    get_modes, 
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
# ### DESI predictions

# %%
from lace.cosmo.thermal_broadening import thermal_broadening_kms
from lace.cosmo import camb_cosmo, fit_linP

# %%
# target 
# DESI KP6 Table 5
bias = -0.1078
err_bias = 0.0036
beta = 1.745
err_beta = 0.5*(0.076 + 0.088)

# input emu
# DESI KP6 
z = 2.33
omnuh2 = 0.0006
mnu = omnuh2 * 93.14
cosmo = {
    'H0': 67.36,
    'omch2': 0.12,
    'ombh2': 0.02237,
    'mnu': mnu,
    'omk': 0,
    'As': 2.1e-09,
    'ns': 0.9649,
    'nrun': 0.0,
    'w': -1.0
}
sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(cosmo)
# compute linear power parameters at each z (in Mpc units)
linP_zs = fit_linP.get_linP_Mpc_zs(
    sim_cosmo, [z], 0.7
)
print(linP_zs[0])
dkms_dMpc_zs = camb_cosmo.dkms_dMpc(sim_cosmo, z=np.array([z]))

# weird values for gamma
# # Fig 9 of Palanque-Delabrouille et al. (2020)
# T0 = 20725
# sigma_T_kms = thermal_broadening_kms(T0)
# sigT_Mpc = sigma_T_kms / dkms_dMpc_zs[0]
# gamma = 0.5442
# mF = 0.7935
# # Fig. 4 of https://arxiv.org/pdf/1704.08366
# lambdap = 95 # [kpc]
# kF_Mpc = 1/(lambdap/1000)


# Table 3 https://arxiv.org/pdf/1808.04367
T0 = 0.5*(0.789+0.831)*1e4
sigma_T_kms = thermal_broadening_kms(T0)
sigT_Mpc = sigma_T_kms / dkms_dMpc_zs[0]
gamma = 0.5*(2.13 + 2.07)
mF = 0.5*(0.796+0.772)
lambdap = 0.5*(91.0+87.2) # [kpc]
kF_Mpc = 1/(lambdap/1000)

# Table 4 https://arxiv.org/pdf/1808.04367
# T0 = 0.5*(1.014+1.165)*1e4
# sigma_T_kms = thermal_broadening_kms(T0)
# sigT_Mpc = sigma_T_kms / dkms_dMpc_zs[0]
# gamma = 0.5*(1.74 + 1.63)
# mF = 0.5*(0.825+0.799)
# lambdap = 0.5*(79.4+81.1) # [kpc]
# kF_Mpc = 1/(lambdap/1000)

emu_params = {
    "mF": mF,
    "gamma": gamma,
    "sigT_Mpc":sigT_Mpc,
    "kF_Mpc":kF_Mpc,
}

print(emu_params)

# %%
dkms_dMpc_zs

# %%
Archive3D.data[0].keys()

# %%
test = Archive3D.get_testing_data("mpg_central")
print(test[-2]["Delta2_p"], test[-2]["n_p"])
for key in emu_params:
    print(key, test[-2][key])

# %% [markdown]
# Comprare predictions of emu and values from observations

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
# training_type = "Arinyo_min"
training_type = "Arinyo_minz"

if (training_type == "Arinyo_min_q1"):
    nparams = 7
    model_path = path_program+"/data/emulator_models/mpg_q1/mpg_hypercube.pt"
elif(training_type == "Arinyo_min"):
    nparams = 8
    # model_path = path_program+"/data/emulator_models/mpg_q1_q2/mpg_hypercube.pt"
    model_path=path_program+"/data/emulator_models/mpg_last.pt"
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
)

# %%
info_power = {
    "cosmo": cosmo,
    "z": z,
}

out = emulator.evaluate(
    emu_params=emu_params,
    info_power=info_power,
    Nrealizations=10000
)

# %%
emu_bias = -out['coeffs_Arinyo']["bias"]
emu_err_bias = out['coeffs_Arinyo_std']["bias"]
emu_beta = out['coeffs_Arinyo']["beta"]
emu_err_beta = out['coeffs_Arinyo_std']["beta"]

# %%
print(bias, err_bias)
print(emu_bias, emu_err_bias)

diff = np.abs(bias-emu_bias)/np.sqrt(err_bias**2+emu_err_bias**2)
print("sigma diff", diff)

print(beta, err_beta)
print(emu_beta, emu_err_beta)

diff = np.abs(beta-emu_beta)/np.sqrt(err_beta**2+emu_err_beta**2)
print("sigma diff", diff)

# %% [markdown]
# - Table 3
#
# -0.1078 0.0036
#
# -0.11649664863944054 0.008865836427629606
#
# sigma diff 0.9088491085393751
#
# 1.745 0.08199999999999999
#
# 1.940719187259674 0.22034360778028178
#
# sigma diff 0.8324685360543379
#
# - Table 4
#   
# -0.1078 0.0036
#
# -0.10775738209486008 0.006852913429005032
#
# sigma diff 0.005505508244426056
#
# 1.745 0.08199999999999999
#
# 2.04947566986084 0.16760074650807494
#
# sigma diff 1.631832462094521

# %%
