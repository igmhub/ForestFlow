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
import forestflow

# path of the repo
path_repo = os.path.dirname(forestflow.__path__[0])

# %% [markdown]
# ### DESI predictions

# %%
from lace.cosmo.thermal_broadening import thermal_broadening_kms
from lace.cosmo import camb_cosmo, fit_linP

# %% [markdown]
# #### Using other people results, old(KP6 + Walther)

# %%
# target 
# DESI KP6 Table 5
bias = -0.1078
err_bias = 0.5*(0.0045+0.0054)
beta = 1.743
err_beta = 0.5*(0.074 + 0.1)

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
# T0 = 0.5*(0.789+0.831)*1e4
# sigma_T_kms = thermal_broadening_kms(T0)
# sigT_Mpc = sigma_T_kms / dkms_dMpc_zs[0]
# gamma = 0.5*(2.13 + 2.07)
# mF = 0.5*(0.796+0.772)
# lambdap = 0.5*(91.0+87.2) # [kpc]
# kF_Mpc = 1/(lambdap/1000)

# Table 4 https://arxiv.org/pdf/1808.04367
T0 = 0.5*(1.014+1.165)*1e4
sigma_T_kms = thermal_broadening_kms(T0)
sigT_Mpc = sigma_T_kms / dkms_dMpc_zs[0]
gamma = 0.5*(1.74 + 1.63)
mF = 0.5*(0.825+0.799)
lambdap = 0.5*(79.4+81.1) # [kpc]
kF_Mpc = 1/(lambdap/1000)

emu_params = {
    "mF": mF,
    "gamma": gamma,
    "sigT_Mpc":sigT_Mpc,
    "kF_Mpc":kF_Mpc,
}

print(emu_params)

# %% [markdown]
# #### Using our results

# %% [markdown]
# - Load chain with As, ns, and IGM params
# - Get ForestFlow input
# - Evaluate ForestFlow to get bias and beta
# - Get other Arinyo, evaluate P1D, and compare with P1D from lace-mpg?

# %%

from cup1d.likelihood.input_pipeline import Args
from cup1d.likelihood.pipeline import Pipeline

# %% [markdown]
# #### Set cup1d

# %%

data_label = "DESIY1_QMLE3"
emulator_label="CH24_mpgcen_gpr"
name_variation = None

args = Args(data_label=data_label, emulator_label=emulator_label)
args.set_baseline(
    fit_type="global_opt",
    fix_cosmo=False,
    P1D_type=data_label,
    name_variation=name_variation,
)
pip = Pipeline(args, out_folder=args.out_folder)

# %% [markdown]
# #### Load chain

# %%
# local
<<<<<<< HEAD
base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1"
folder = os.path.join(base, "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7/")
# nersc
# base = "/global/cfs/cdirs/desi/users/jjchaves/p1d"

fname = os.path.join(folder, "chain.npy")
chain = np.array(np.load(fname))
=======
base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"
# nersc
# base = "/pscratch/sd/j/jjchaves/data/out_DESI_DR1/"
folder = base + "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7/"
chain = np.array(np.load(folder + "chain.npy"))
>>>>>>> 732e4ca (updating notebook)
chain = chain.reshape(-1, 53)
chain.shape

# %% [markdown]
# #### Initiate input from chain
#
# We select 500 points from the chains randomly

# %%

zs = np.linspace(2.2, 3.8, 8)
nn = 500

ind = np.random.permutation(np.arange(2739200))[:nn]
pars_chain = {}
pars_chain["z"] = zs
pars_chain["Delta2_p"] = np.zeros((ind.shape[0], zs.shape[0]))
pars_chain["n_p"] = np.zeros((ind.shape[0], zs.shape[0]))
pars_chain["mF"] = np.zeros((ind.shape[0], zs.shape[0]))
pars_chain["gamma"] = np.zeros((ind.shape[0], zs.shape[0]))
pars_chain["sigT_Mpc"] = np.zeros((ind.shape[0], zs.shape[0]))
pars_chain["kF_Mpc"] = np.zeros((ind.shape[0], zs.shape[0]))

# %% [markdown]
# #### From each point, we get the input parameters to ForestFlow

# %%
for ii in range(ind.shape[0]):
# for ii in range(1):

    chain_params = pip.fitter.like.parameters_from_sampling_point(chain[ind[ii], :])

    # Planck cosmo
    
    cosmo = {
        'H0': 67.66,
        'mnu': 0,
        'omch2': 0.119,
        'ombh2': 0.0224,
        'omk': 0,
        # 'As': 2.105e-09,
        'As': chain_params[0].value_from_cube(chain[ind[ii],0]),
        # 'ns': 0.9665,
        'ns': chain_params[1].value_from_cube(chain[ind[ii],1]),
        'nrun': 0.0,
        'pivot_scalar':0.05,
        'w': -1.0
    }
    sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(cosmo)
    # compute linear power parameters at each z (in Mpc units)
    kp_Mpc = 0.7
    linP_zs = fit_linP.get_linP_Mpc_zs(
        sim_cosmo, zs, kp_Mpc
    )
    dkms_dMpc_zs = camb_cosmo.dkms_dMpc(sim_cosmo, z=zs)

    for jj in range(len(linP_zs)):
        pars_chain["Delta2_p"][ii, jj] = linP_zs[jj]["Delta2_p"]
        pars_chain["n_p"][ii, jj] = linP_zs[jj]["n_p"]

    pars_chain["mF"][ii] = pip.fitter.like.theory.model_igm.models[
        "F_model"
    ].get_mean_flux(zs, like_params=chain_params)
    pars_chain["gamma"][ii] = pip.fitter.like.theory.model_igm.models[
        "T_model"
    ].get_gamma(zs, like_params=chain_params)

    
    sigT_kms = pip.fitter.like.theory.model_igm.models[
        "T_model"
    ].get_sigT_kms(zs, like_params=chain_params)
    
    pars_chain["sigT_Mpc"][ii] = sigT_kms / dkms_dMpc_zs

    kF_kms = pip.fitter.like.theory.model_igm.models[
        "P_model"
    ].get_kF_kms(zs, like_params=chain_params)
    pars_chain["kF_Mpc"][ii] = kF_kms * dkms_dMpc_zs

# %%
# pars_chain

# %% [markdown]
# load model to evaluate IGM parameters

# %%


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
# T0 = 0.5*(0.789+0.831)*1e4
# sigma_T_kms = thermal_broadening_kms(T0)
# sigT_Mpc = sigma_T_kms / dkms_dMpc_zs[0]
# gamma = 0.5*(2.13 + 2.07)
# mF = 0.5*(0.796+0.772)
# lambdap = 0.5*(91.0+87.2) # [kpc]
# kF_Mpc = 1/(lambdap/1000)

# Table 4 https://arxiv.org/pdf/1808.04367
# T0 = 0.5*(1.014+1.165)*1e4
# sigma_T_kms = thermal_broadening_kms(T0)
# sigT_Mpc = sigma_T_kms / dkms_dMpc_zs[0]
# gamma = 0.5*(1.74 + 1.63)
# mF = 0.5*(0.825+0.799)
# lambdap = 0.5*(79.4+81.1) # [kpc]
# kF_Mpc = 1/(lambdap/1000)

# emu_params = {
#     "mF": mF,
#     "gamma": gamma,
#     "sigT_Mpc":sigT_Mpc,
#     "kF_Mpc":kF_Mpc,
# }

# print(emu_params)

# %% [markdown]
# ## LOAD P3D ARCHIVE (needed for forestflow)

# %%
# %%time
folder_lya_data = path_repo + "/data/best_arinyo/"
folder_interp = path_repo + "/data/plin_interp/"

Archive3D = GadgetArchive3D(
    base_folder=path_repo,
    folder_data=folder_lya_data,
    force_recompute_plin=False,
    average="both",
)
print(len(Archive3D.training_data))


# %% [markdown]
# ## Load ForestFlow

# %%
training_type = "Arinyo_min"
model_path=path_repo + "/data/emulator_models/mpg_hypercube.pt"

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
# #### Initiate output to store Arinyo parameters

# %%
pars = pars_chain.keys()
pars_ari = ['bias', 'beta', 'q1', 'kvav', 'av', 'bv', 'kp', 'q2']

out_ari = {}
for par in pars_ari:
    out_ari[par] = np.zeros_like(pars_chain['mF'])

# %% [markdown]
# #### Call forestflow for each of the 500 points from the chain

# %%
# info_power = {
#     "cosmo": cosmo,
#     "z": z,
# }
# out

for ii in range(pars_chain['mF'].shape[0]):
    for jj in range(pars_chain['mF'].shape[1]):
        emu_params = {}
        for par in pars:
            if par == "z":
                continue
            emu_params[par] = pars_chain[par][ii, jj]

        # print(emu_params)
        out = emulator.evaluate(
            emu_params=emu_params,
            # info_power=info_power,
            Nrealizations=10000
        )

        for par in pars_ari:
            out_ari[par][ii, jj] = out['coeffs_Arinyo'][par]

# %% [markdown]
# #### Needed because the bias is possitive internally in forestflow

# %%
out_ari['bias'] = -out_ari['bias']

# %% [markdown]
# #### Store output for future use

# %%
<<<<<<< HEAD
save = True
=======
save = False
>>>>>>> 732e4ca (updating notebook)
if save:
    dict_out_all = {}
    dict_out_all["emu_params"] = pars_chain
    dict_out_all["ari_params"] = out_ari
    dict_out_all["zs"] = zs
    np.save("arinyo_from_p1d.npy", dict_out_all)

# %% [markdown]
# #### Load output

# %%
dict_out_all = np.load("arinyo_from_p1d.npy", allow_pickle=True).item()
zs = dict_out_all["zs"]
out_ari = dict_out_all["ari_params"]

# %%
dict_out_all.keys()

# %%
out_ari['bias'].shape

# %%
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8,10))
ftsize = 18

# target 
# DESI DR1 Table 5
z = 2.33
bias = -0.1078
err_bias = 0.5*(0.0045+0.0054)
beta = 1.743
err_beta = 0.5*(0.074 + 0.1)

# DESI DR2 (combined)
z = 2.33
bias2 = -0.1352
err_bias2 = 0.0073
beta2 = 1.445
err_beta2 = 0.064

# Hiram
hi_z = np.array([2.13, 2.40, 2.81])
hi_bias = np.array([-0.0703, -0.1428, -0.2286])
hi_bias_err = np.array([[0.0093, 0.012], [0.015, 0.0093], [0.011, 0.0052]])
hi_beta = np.array([2.25, 1.469, 1.208])
hi_beta_err = np.array([[0.35, 0.26], [0.14, 0.071], [0.069, 0.047]])


# err
mean = out_ari['bias'].mean(axis=0)
percen = np.percentile(out_ari['bias'], [16, 84], axis=0)
low = mean - percen[0]
high = percen[1] - mean
err = np.zeros((2, mean.shape[0]))
err[0] = low
err[1] = high

ax[0].errorbar(zs, mean, err, label="DR1 P1D")
ax[0].errorbar([z,z], np.zeros(2)+bias, np.zeros(2)+err_bias, label="DR1 BAO") 
ax[0].errorbar([z,z], np.zeros(2)+bias2, np.zeros(2)+err_bias2, label="DR2 BAO")

col = "C3"
for jj in range(len(hi_z)):
    if jj == 0:
        label = "Hiram DR2 BAO"
    else:
        label = None
    ax[0].errorbar([hi_z[jj], hi_z[jj]], np.zeros(2)+hi_bias[jj], hi_bias_err[jj:jj+1, :].T, label=label, color=col, fmt=".")

# err
mean = out_ari['beta'].mean(axis=0)
percen = np.percentile(out_ari['beta'], [16, 84], axis=0)
low = mean - percen[0]
high = percen[1] - mean
err = np.zeros((2, mean.shape[0]))
err[0] = low
err[1] = high

ax[1].errorbar(zs, mean, err)
ax[1].errorbar([z,z], np.zeros(2)+beta, np.zeros(2)+err_beta) 
ax[1].errorbar([z,z], np.zeros(2)+beta2, np.zeros(2)+err_beta2) 


for jj in range(len(hi_z)):
    ax[1].errorbar([hi_z[jj], hi_z[jj]], np.zeros(2)+hi_beta[jj], hi_beta_err[jj:jj+1, :].T, label=label, color=col, fmt=".")

ax[0].legend(fontsize=ftsize)

ax[0].set_ylabel(r"$b$", fontsize=ftsize)
ax[1].set_ylabel(r"$\beta$", fontsize=ftsize)
ax[1].set_xlabel(r"$z$", fontsize=ftsize)


plt.tight_layout()
plt.savefig("bias_beta_BAOvsP1D.png")
plt.savefig("bias_beta_BAOvsP1D.pdf")

# %%
