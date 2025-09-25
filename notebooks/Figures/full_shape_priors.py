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
# # Priors on Arinyo parameters
#
# In this notebook, we compute priors for the Arinyo parameters using two methods:
#
# - We generate priors using Forestflow. To do so, we use as input the DESI DR2 + CMB LCDM cosmology and IGM parameters from Table 4 of Walther+18
#
# - From the best-fitting of the Arinyo model to the LaCE suite of simulations (Pedersen+21)

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

from corner import corner


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
# ## Using forestflow

# %%
from lace.cosmo.thermal_broadening import thermal_broadening_kms
from lace.cosmo import camb_cosmo, fit_linP

# %% [markdown]
# #### Compute input parameters of forestflow
#
# Example here, computed using script

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


# %% [markdown]
# ### Load result from previous computation

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
# corner(np.array([data['Delta2_p'], data['n_p']]).T);

# %% [markdown]
# #### Load emulator

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

# %% [markdown]
# #### Evaluate emulator

# %%
# %%time
nelem = data["n_p"].shape[0]

out_arinyo = {}
for key in emulator.Arinyo_params:
    out_arinyo[key] = np.zeros(nelem)

for ii in range(nelem):
    if ii % 1000 == 0:
        print(ii, nelem, ii/nelem)

    emu_params = {}
    for key in data:
        emu_params[key] = data[key][ii]
    
    out = emulator.predict_Arinyos(
        emu_params,
        Nrealizations=500
    )

    for jj, key in enumerate(emulator.Arinyo_params):
        if key == "bias":
            out_arinyo[key][ii] = -out[jj]
        else:
            out_arinyo[key][ii] = out[jj]


# %%
folder_data = "/home/jchaves/Proyectos/projects/lya/ForestFlow/scripts/out/"
np.save("forestflow_pred.npy", out_arinyo)

# %% [markdown]
# #### Load results from emulator

# %%
folder_data = "/home/jchaves/Proyectos/projects/lya/ForestFlow/scripts/out/"
file_name = "forestflow_pred.npy"

file = "/pscratch/sd/j/jjchaves/forestflow_pred.npy" # in nersc
out_arinyo = np.load(file_name, allow_pickle=True).item()
out_arinyo.keys()

# %%
arr_all = np.zeros((out_arinyo["bias"].shape[0], len(out_arinyo.keys())))
for jj, key in enumerate(out_arinyo):
    arr_all[:, jj] = out_arinyo[key]

# %%
figure = corner(
    arr_all, 
    show_titles=True, 
    labels=emulator.Arinyo_params, 
    label_kwargs={"fontsize":20}, 
    title_kwargs={"fontsize":20},
    title_fmt='.3f'
)
for ax in figure.get_axes():
    ax.tick_params(axis='both', labelsize=16)
plt.savefig("arinyo_from_DR2cosmo_waltherIGM_z233.pdf")

# %% [markdown]
# ## From the best-fit of the Arinyo model

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
Archive3D.training_data[0].keys()

# %%
dict_arinyo = Archive3D.training_data[0]["Arinyo_min"].copy()
nsims = len(Archive3D.training_data)
zz = np.zeros((nsims))
for key in dict_arinyo:
    dict_arinyo[key] = np.zeros((nsims))
    for ii in range(nsims):
        dict_arinyo[key][ii] = Archive3D.training_data[ii]["Arinyo_min"][key]
for ii in range(nsims):
    zz[ii] = Archive3D.training_data[ii]["z"]

# %%
ind = np.argwhere(zz < 3)[:,0]

# %%
dict_arinyo["bias"] = -dict_arinyo["bias"]

# %%
arr_arinyo = np.zeros((len(ind), len(dict_arinyo)))
for ii, key in enumerate(dict_arinyo):
    arr_arinyo[:, ii] = dict_arinyo[key][ind]

# %%
figure = corner(
    arr_arinyo, 
    show_titles=True, 
    labels=emulator.Arinyo_params, 
    label_kwargs={"fontsize":20}, 
    title_kwargs={"fontsize":20},
    title_fmt='.3f'
)
for ax in figure.get_axes():
    ax.tick_params(axis='both', labelsize=16)
plt.savefig("arinyo_from_fits_LaCE_z200_275.pdf")

# %%
