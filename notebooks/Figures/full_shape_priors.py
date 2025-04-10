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

# %%
from lace.cosmo.thermal_broadening import thermal_broadening_kms
from lace.cosmo import camb_cosmo, fit_linP

# %% [markdown]
# #### Given:
# - a range of possible values for the cosmology parameters (or even at fixed Planck cosmo)
# - a range of possible values for IGM parameters based on the literature (mean flux, T0, gamma, maybe also kF)
# #### One can compute:
# - what range of bias/beta parameters are predicted by ForestFlow (this could be a useful sanity check, but I wouldn’t use any such prior!)
# - what range of values are predicted for the other D_NL parameters (q1, bv, av, etc.)
# - We could then inflate these a bit, if needed, and use these as priors

# %%
# %%time

# which DESI constraints to use
use_w0wa = False

# number of samples
nn = 20
emu_params = {
    "Delta2_p": np.zeros(nn),
    "n_p": np.zeros(nn),
    "mF": np.zeros(nn),
    "gamma": np.zeros(nn),
    "sigT_Mpc": np.zeros(nn),
    "kF_Mpc": np.zeros(nn),
}

kp_Mpc = 0.7

## Redshift
# DESI-DR1 KP6 
z = 2.33

## IGM
# Table 4 https://arxiv.org/pdf/1808.04367
T0 = 0.5 * (1.014 + 1.165) * 1e4
err_T0 = 0.25 * (0.25 + 0.15 + 0.29 + 0.19) * 1e4

gamma = 0.5 * (1.74 + 1.63)
err_gamma = 0.25 * (0.15 + 0.21 + 0.16 + 0.19)

mF = 0.5 * (0.825 + 0.799)
err_mF = 0.25 * (0.009 + 0.008 + 0.008 + 0.008)

lambdap = 0.5 * (79.4 + 81.1) # [kpc]
err_lambdap = 0.25 * (5.1 + 5.0 + 4.6 + 4.7)

err_T0_use = np.random.normal(size=nn) * err_T0
err_gamma_use = np.random.normal(size=nn) * err_gamma
err_mF_use = np.random.normal(size=nn) * err_mF
err_lambdap_use = np.random.normal(size=nn) * err_lambdap

## COSMO
if use_w0wa == False:
    # TABLE V
    # DESI DR2 + CMB LCDM
    Om = 0.3027
    Om_err = 0.0036
    H0 = 68.17
    H0_err = 0.28
    w0 = -1
    w0_err = 0
    wa = 0
    wa_err = 0
else:
    # DESI+CMB+DESY5 w0waCDM
    Om=0.3191
    Om_err=0.0056
    H0 = 66.74
    H0_err = 0.56
    w0 = -0.752
    w0_err = 0.057
    wa = -0.86
    wa_err = 0.22

omnuh2 = 0.0006 # fixed
mnu = omnuh2 * 93.14

# Planck2018 Table 1
ombh2 = 0.02233
ombh2_err = 0.00015
# omch2 = 0.1198
# omch2_err = 0.0012
ln_As1010 = 3.043
ln_As1010_err = 0.014
ns = 0.9652
ns_err = 0.0042

err_ln_As1010_use = np.random.normal(size=nn) * ln_As1010_err
err_ns_use = np.random.normal(size=nn) * ns_err
err_Om_use = np.random.normal(size=nn) * Om_err
err_ombh2_use = np.random.normal(size=nn) * ombh2_err
err_H0_use = np.random.normal(size=nn) * H0_err
err_w0_use = np.random.normal(size=nn) * w0_err
err_wa_use = np.random.normal(size=nn) * wa_err

for ii in range(nn):
    _H0 = H0 + err_H0_use[ii]
    _Om = Om + err_Om_use[ii]
    _ombh2 = ombh2 + err_ombh2_use[ii]
    _omch2 = _Om * (_H0/100)**2 - _ombh2
    _ln_As1010 = np.exp(ln_As1010 + err_ln_As1010_use[ii]) * 1e-10
    _ns = ns + err_ns_use[ii]
    _w0 = w0 + err_w0_use[ii]
    _wa = wa + err_wa_use[ii]
    
    cosmo = {
        'H0': _H0,
        'omch2': _omch2,
        'ombh2': _ombh2,
        'mnu': mnu,
        'omk': 0,
        'As': _ln_As1010,
        'ns': _ns,
        'nrun': 0.0,
        'w': _w0,
        "wa": _wa
    }
    
    sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(cosmo)
    # compute linear power parameters at each z (in Mpc units)
    linP_zs = fit_linP.get_linP_Mpc_zs(sim_cosmo, [z], kp_Mpc)
    # print(linP_zs[0])
    dkms_dMpc_zs = camb_cosmo.dkms_dMpc(sim_cosmo, z=np.array([z]))

    emu_params["Delta2_p"][ii] = linP_zs[0]["Delta2_p"]
    emu_params["n_p"][ii] = linP_zs[0]["n_p"]
    
    emu_params["mF"][ii] = mF + err_mF_use[ii]
    emu_params["gamma"][ii] = gamma + err_gamma_use[ii]
    
    sigma_T_kms = thermal_broadening_kms(T0 + err_T0_use[ii])
    sigT_Mpc = sigma_T_kms / dkms_dMpc_zs[0]    
    emu_params["sigT_Mpc"][ii] = sigT_Mpc
    
    kF_Mpc = 1/((lambdap + err_lambdap_use[ii])/1000)
    emu_params["kF_Mpc"][ii] = kF_Mpc
    

# %%
# plt.scatter(emu_params["Delta2_p"], emu_params["n_p"])
# plt.scatter(emu_params["mF"], emu_params["gamma"])
100000/100 * 18/3600

# %%
folder_data = "/home/jchaves/Proyectos/projects/lya/ForestFlow/scripts/out/"
size = 25
nn = 100000

ind = np.arange(nn)
ind_use = np.array_split(ind, size)
for rank in range(size):
    if rank == 0:
        data = np.load(folder_data + "input_priors"+str(rank)+".npy",allow_pickle=True).item()
    else:
        _data = np.load(folder_data + "input_priors"+str(rank)+".npy",allow_pickle=True).item()
        for key in data.keys():
            data[key][ind_use[rank]] = _data[key][ind_use[rank]]


# %%
ind = data["Delta2_p"] != 0

for key in data.keys():
    data[key] = data[key][ind] 

# %%
data

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

# %%
training_type = "Arinyo_min"
model_path=path_program+"/data/emulator_models/mpg_hypercube.pt"

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
nelem = data["n_p"].shape[0]

out_arinyo = {}
for key in emulator.Arinyo_params:
    out_arinyo[key] = np.zeros(nelem)


for ii in range(nelem):

    emu_params = {}
    for key in data:
        emu_params[key] = data[key][ii]
    
    out = emulator.predict_Arinyos(
        emu_params,
        Nrealizations=10000
    )

    for jj, key in enumerate(emulator.Arinyo_params):
        if key == "bias":
            out_arinyo[key][ii] = -out[jj]
        else:
            out_arinyo[key][ii] = out[jj]


# %%
out_arinyo

# %%
arr_all = np.zeros((out_arinyo["bias"].shape[0], len(out_arinyo.keys())))
for jj, key in enumerate(out_arinyo):
    arr_all[:, jj] = out_arinyo[key]

# %%
from corner import corner

# %%
corner(arr_all);

# %%

# %%
