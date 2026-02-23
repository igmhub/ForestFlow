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
# # Test ForestFlow with ACCEL2
#
# - Get chain data from ACCEL2 Dfit
# - Read cup1d chain and evaluate ForestFlow to get Arinyo, P1D, and P3D
# - Compare P1D and P3D from ForestFlow with ACCEL2 predictions
# - Get priors on Arinyo from fit
# - Compare ACCEL2 P1D with best-fitting P1D to DESIY1
# - 

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

from cup1d.likelihood.pipeline import Pipeline
from lace.cosmo import camb_cosmo, fit_linP
import forestflow
from forestflow.P3D_cINN import P3DEmulator
# path of the repo
path_repo = os.path.dirname(forestflow.__path__[0])

np.__version__

# %% [markdown]
# # Read chain using cup1d (DESI DR1), jump to load chain below after running once

# %% [markdown]
# #### Set cup1d

# %%

from cup1d.likelihood.input_pipeline import Args

# %%
emu = "mpg"
fit_type = "global_opt"
# mcmc_conf = "explore"
mcmc_conf = "test"
path_data = "jjchaves"
# path_data = "nersc"
cov_label = "DESIY1_QMLE3"
data_label = "accel2"
name_variation = "sim_" + data_label

zmin = 2.2
zmax = 4.2

args = Args(
    data_label=data_label,
    cov_label=cov_label,
    emulator_label="CH24_" + emu + "cen_gpr",
    true_cosmo_label=data_label,
    apply_smoothing=True,
    add_noise=False,
    seed_noise=0,
    emu_cov_type="full",
)

args.set_baseline(
    fit_type=fit_type,
    fix_cosmo=False,
    fid_cosmo_label=data_label,
    P1D_type=cov_label,
    name_variation=name_variation,
    z_min=zmin,
    z_max=zmax,
    mcmc_conf=mcmc_conf,
)

if data_label == "accel2":
    if path_data == "jjchaves":
        args.path_data = (
            "/home/jchaves/Proyectos/projects/lya/data/accel2/frontier_grid"
        )
    elif path_data == "nersc":
        args.path_data = "/global/cfs/cdirs/desi/users/ravouxco/accel2/shared_files/frontier_grid"
    else:
        raise ValueError(
            "path_data not defined for data_label = " + data_label
        )

pip = Pipeline(args)

# %% [markdown]
# #### Load chain

# %%
# local
folder = "/home/jchaves/Proyectos/projects/lya/data/accel2/chains/chain_1/"
# nersc
# folder = "/global/cfs/cdirs/desi/users/jjchaves/p1d"

fname = os.path.join(folder, "chain.npy")
chain = np.array(np.load(fname))
chain = chain.reshape(-1, 53)
chain.shape

# %% [markdown]
# #### Initiate input from chain
#
# We select 500 points from the chains randomly

# %%

zs_dat = pip.fitter.like.data.z.copy()
# you change zs, take a look at the end of the cell
zs = np.array([2.2, 2.33, 2.6, 3. , 3.6, 4.]) # adding 2.33 for the priors
nn = 1000

ind = np.random.permutation(np.arange(chain.shape[0]))[:nn]
pars_chain = {}

pars_chain["z"] = zs

# cosmo (for Arinyo model)
pars_chain["As"] = np.zeros((ind.shape[0]))
pars_chain["ns"] = np.zeros((ind.shape[0]))

# input ForestFlow
pars_chain["Delta2_p"] = np.zeros((ind.shape[0], zs.shape[0]))
pars_chain["n_p"] = np.zeros((ind.shape[0], zs.shape[0]))
pars_chain["mF"] = np.zeros((ind.shape[0], zs.shape[0]))
pars_chain["gamma"] = np.zeros((ind.shape[0], zs.shape[0]))
pars_chain["sigT_Mpc"] = np.zeros((ind.shape[0], zs.shape[0]))
pars_chain["kF_Mpc"] = np.zeros((ind.shape[0], zs.shape[0]))

# store p1d w/ and w/o contaminants
ii = 0
p1d = pip.fitter.like.get_p1d_kms(
    pip.fitter.like.data.z, pip.fitter.like.data.k_kms, chain[ind[ii], :], no_contaminants=True
)
pars_chain["k_kms"] = np.zeros((zs.shape[0], len(p1d[0][-1])))
pars_chain["p1d"] = np.zeros((ind.shape[0], zs.shape[0], len(p1d[0][-1])))
pars_chain["p1d_nocont"] = np.zeros((ind.shape[0], zs.shape[0], len(p1d[0][-1])))

for jj in range(zs.shape[0]):
    _ind = np.argmin(np.abs(zs[jj] - zs_dat))
    nelem = len(p1d[0][_ind])
    pars_chain["k_kms"][jj, :nelem] = pip.fitter.like.data.k_kms[_ind]

# %%
chi2_ev = np.zeros((100))
for ii in range(100):
    chi2_ev[ii] = pip.fitter.like.get_chi2(chain[ind[ii], :])
plt.hist(chi2_ev, bins=20);

# %%
ii = 88
pip.fitter.like.plot_p1d(chain[ind[ii], :], print_chi2=False)

# pip.fitter.like.plot_p1d(p0, print_chi2=False)

# %% [markdown]
# #### From each point, we get the input parameters to ForestFlow
#
# Most of the time goes into calling CAMB

# %%
# we used the Planck cosmo as fiducial in the DR1 fit

Omegam = 0.31
Omegab = 0.0487
h = 0.675
omch2 = (Omegam - Omegab) * h**2
ombh2 = Omegab * h**2
# accel2
fid_cosmo = {
    "H0": h * 100,
    "mnu": 0,
    "omch2": omch2,
    "ombh2": ombh2,
    "omk": 0,
    # 'As': 2.105e-09,
    "As": np.exp(3.094) / 1e10,
    # 'ns': 0.9665,
    "ns": 0.96,
    "nrun": 0.0,
    "pivot_scalar": 0.05,
    "w": -1.0,
}

# only need to call it once since we are only chaing As and ns during inference
sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(fid_cosmo)

# compute linear power parameters at each z (in Mpc units)
kp_Mpc = 0.7
linP_zs = fit_linP.get_linP_Mpc_zs(sim_cosmo, zs, kp_Mpc)

# compute scaling of kms to Mpc
dkms_dMpc_zs = camb_cosmo.dkms_dMpc(sim_cosmo, z=zs)


# %%
def rescale_linP(fid_cosmo, tar_cosmo, linP_zs, kp_Mpc=0.7, ks_Mpc=0.05):

    ratio_As = 1.0
    delta_ns = 0.0
    delta_nrun = 0.0

    for par in tar_cosmo:
        ratio_As = tar_cosmo["As"] / fid_cosmo["As"]
        delta_ns = tar_cosmo["ns"] - fid_cosmo["ns"]
        delta_nrun = tar_cosmo["nrun"] - fid_cosmo["nrun"]
    
        ln_kp_ks = np.log(kp_Mpc / ks_Mpc)

        # compute scalings
        delta_alpha_p = delta_nrun
        delta_n_p = delta_ns + delta_nrun * ln_kp_ks
        ln_ratio_A_p = (
            np.log(ratio_As)
            + (delta_ns + 0.5 * delta_nrun * ln_kp_ks) * ln_kp_ks
        )

        # update values of linP_params at emulator pivot point, at each z
        linP_Mpc_params = []
        for zlinP in linP_zs:
            linP_Mpc_params.append(
                {
                    "Delta2_p": zlinP["Delta2_p"] * np.exp(ln_ratio_A_p),
                    "n_p": zlinP["n_p"] + delta_n_p,
                    "alpha_p": zlinP["alpha_p"] + delta_alpha_p,
                }
            )
        return linP_Mpc_params


# %%

# %%
for ii in range(ind.shape[0]):
    if ii % 100 == 0:
        print(ii)

    chain_params = pip.fitter.like.parameters_from_sampling_point(
        chain[ind[ii], :]
    )

    pars_chain["mF"][ii] = pip.fitter.like.theory.model_igm.models[
        "F_model"
    ].get_mean_flux(zs, like_params=chain_params)

    pars_chain["gamma"][ii] = pip.fitter.like.theory.model_igm.models[
        "T_model"
    ].get_gamma(zs, like_params=chain_params)

    sigT_kms = pip.fitter.like.theory.model_igm.models["T_model"].get_sigT_kms(
        zs, like_params=chain_params
    )
    pars_chain["sigT_Mpc"][ii] = sigT_kms / dkms_dMpc_zs

    kF_kms = pip.fitter.like.theory.model_igm.models["P_model"].get_kF_kms(
        zs, like_params=chain_params
    )
    pars_chain["kF_Mpc"][ii] = kF_kms * dkms_dMpc_zs

    p1d = pip.fitter.like.get_p1d_kms(
        pip.fitter.like.data.z, pip.fitter.like.data.k_kms, chain[ind[ii], :]
    )
    p1d_no = pip.fitter.like.get_p1d_kms(
        pip.fitter.like.data.z, pip.fitter.like.data.k_kms, chain[ind[ii], :], no_contaminants=True
    )    

    
    for jj in range(zs.shape[0]):
        _ind = np.argmin(np.abs(zs[jj] - zs_dat))
        nelem = len(p1d[0][_ind])
        pars_chain["p1d"][ii, jj, :nelem] = p1d[0][_ind]
        pars_chain["p1d_nocont"][ii, jj, :nelem] = p1d_no[0][_ind]
    
    tar_cosmo = {
        "As": chain_params[0].value_from_cube(chain[ind[ii], 0]),
        "ns": chain_params[1].value_from_cube(chain[ind[ii], 1]),
        "nrun":0
    }

    pars_chain["As"][ii] = tar_cosmo["As"]
    pars_chain["ns"][ii] = tar_cosmo["ns"]
    res_linP_zs = rescale_linP(fid_cosmo, tar_cosmo, linP_zs)    
    for jj in range(zs.shape[0]):        
        pars_chain["Delta2_p"][ii, jj] = res_linP_zs[jj]["Delta2_p"]
        pars_chain["n_p"][ii, jj] = res_linP_zs[jj]["n_p"]


# %%
np.save("inter_chain_accel2.npy", pars_chain)

# %%

# %% [markdown]
# # Load results from chain

# %%
pars_chain = np.load("inter_chain_accel2.npy", allow_pickle=True).item()

# %% [markdown]
# #### Best-fitting uncontaminated P1D measurements

# %%
for ii in range(pars_chain["z"].shape[0]):
    _ = pars_chain["k_kms"][ii] != 0
    kk = pars_chain["k_kms"][ii, _]
    plt.errorbar(
        kk, 
        kk/np.pi * np.mean(pars_chain["p1d_nocont"][:, ii, _], axis=0), 
        kk/np.pi * np.std(pars_chain["p1d_nocont"][:, ii, _], axis=0)
    )
    # plt.plot(
    #     kk, 
    #     kk/np.pi * np.mean(pars_chain["p1d"][:, ii, _], axis=0), 
    # )
plt.xscale("log")

# %%

# %%

# %%

# %% [markdown]
# ## Load ForestFlow

# %%
emulator = P3DEmulator(
    model_path = path_repo + "/data/emulator_models/forest_mpg",
)

# %% [markdown]
# #### Initiate output to store Arinyo parameters

# %%
pars = pars_chain.keys()

out_ari = {}
for par in emulator.Arinyo_params:
    out_ari[par] = np.zeros_like(pars_chain["mF"])

out_ari["p1d"] = np.zeros_like(pars_chain["p1d_nocont"])

# %% [markdown]
# #### Read P3D data cup1d

# %%
from cup1d.p1ds.data_accel2 import load_data
folder = "/home/jchaves/Proyectos/projects/lya/data/accel2/frontier_grid/"
data_accel2 = load_data(folder)
out_ari["plin"] = np.zeros((out_ari["p1d"].shape[0], out_ari["p1d"].shape[1], data_accel2["k3d_Mpc"].shape[0]))

# %%
nmu = 25
knew3d = np.zeros((data_accel2["k3d_Mpc"].shape[0], nmu))
munew3d = np.zeros((data_accel2["k3d_Mpc"].shape[0], nmu))
mnu_uni = np.linspace(0, 1, nmu)
for ii in range(nmu):
    knew3d[:, ii] = data_accel2["k3d_Mpc"][:, 0]
    munew3d[:, ii] = mnu_uni[ii]
    
out_ari["p3d"] = np.zeros((out_ari["p1d"].shape[0], out_ari["p1d"].shape[1], knew3d.shape[0], knew3d.shape[1]))

# %%
data_accel2.keys()

# %%
from forestflow.model_p3d_arinyo import ArinyoModel

# %%
#initiate Arinyo model, needed to compute P1D

# accel2 cosmo
Omegam = 0.31
Omegab = 0.0487
h = 0.675
omch2 = (Omegam - Omegab) * h**2
ombh2 = Omegab * h**2
# accel2
fid_cosmo = {
    "H0": h * 100,
    "mnu": 0,
    "omch2": omch2,
    "ombh2": ombh2,
    "omk": 0,
    # 'As': 2.105e-09,
    "As": np.exp(3.094) / 1e10,
    # 'ns': 0.9665,
    "ns": 0.96,
    "nrun": 0.0,
    "pivot_scalar": 0.05,
    "w": -1.0,
}

model_Arinyo = ArinyoModel(fid_cosmo)

# %%
# %%time
# chain step
for ii in range(pars_chain["mF"].shape[0]):
    if ii % 100 == 0:
        print(ii)

    new_cosmo = {
        "H0": h * 100,
        "mnu": 0,
        "omch2": omch2,
        "ombh2": ombh2,
        "omk": 0,
        # 'As': 2.105e-09,
        "As": pars_chain["As"][ii],
        # 'ns': 0.9665,
        "ns": pars_chain["ns"][ii],
        "nrun": 0.0,
        "pivot_scalar": 0.05,
        "w": -1.0,
    }
    # redshift
    for jj in range(pars_chain["mF"].shape[1]):
        input_emu = {}
        for par in pars:
            if par not in ['Delta2_p', 'n_p', 'mF', 'gamma', 'sigT_Mpc', 'kF_Mpc']:
                continue
            input_emu[par] = pars_chain[par][ii, jj]

        par_ari = emulator.predict_Arinyos(emu_params=input_emu)
        for par in par_ari:
            out_ari[par][ii, jj] = par_ari[par]

        # set k_Mpc
        _ = pars_chain["k_kms"][jj] != 0
        # we can use this one because we only chage the priordial power
        k_Mpc = pars_chain["k_kms"][jj, _] * dkms_dMpc_zs[jj]
        P1D_Mpc = model_Arinyo.P1D_Mpc(
            pars_chain["z"][jj],
            k_Mpc,
            par_ari,
            cosmo_new=new_cosmo,
        )
        out_ari["p1d"][ii, jj, :np.sum(_)]  = P1D_Mpc * dkms_dMpc_zs[jj]
        out_ari["p3d"][ii, jj, ...] = model_Arinyo.P3D_Mpc(
            pars_chain["z"][jj],
            knew3d,
            munew3d,
            par_ari,
            cosmo_new=new_cosmo,
        )
        out_ari["plin"][ii, jj, :] = model_Arinyo.linP_Mpc(
            pars_chain["z"][jj],
            data_accel2["k3d_Mpc"][:, 0],
            cosmo_new=new_cosmo,
        )



# %%
save = True
if save:
    dict_out_all = {}
    dict_out_all["emu_params"] = pars_chain
    dict_out_all["forest_out"] = out_ari
    dict_out_all["zs"] = zs
    np.save("arinyo_from_p1d_new_accel2.npy", dict_out_all)

# %%

# %%
for jj in range(4):
    ind = slice(6*jj,6*(jj+1)+1)
    print(mnu_uni[ind])

# %%
diffk = 0.5 * (data_accel2["k3d_Mpc"][1,0] - data_accel2["k3d_Mpc"][0,0])
diffk = 0

# %%

# %%

fig, ax = plt.subplots(3,1, figsize=(10, 14), sharex=True)
ax = ax.reshape(-1)

_ = knew3d[:,0] < 4

for ii in range(2, pars_chain["z"].shape[0]-1):
    for jj in range(4):
        if (ii == 2) and (jj == 0):
            label = "ForestFlow"
        else:
            label = None

        # average of many mu bins, same range as accel2 data
        ind = slice(6*jj,6*(jj+1)+1)
        
        yy = np.mean(out_ari["p3d"][:, ii, :, ind]/out_ari["plin"][:, ii-1, :, np.newaxis], axis=(0,2))
        yerr = np.std(out_ari["p3d"][:, ii, :, ind]/out_ari["plin"][:, ii-1, :, np.newaxis], axis=(0,2))
        ax[ii-2].fill_between(
            knew3d[_, jj], 
            (yy - 0.5 * yerr)[_], 
            (yy + 0.5 * yerr)[_], 
            alpha=0.3,
            color="C"+str(jj),
            label=label
        )

        
        if (ii == 2) and (jj == 0):
            label = "ACCEL2"
        else:
            label = None

        ax[ii-2].plot(
            knew3d[_, jj] - diffk,
            (data_accel2["p3d_Mpc"][ii-1, :, jj]/np.mean(out_ari["plin"][:, ii-1, :], axis=0))[_],
            "C"+str(jj)+"-",
            label=label
        )

ftsize = 20
ax[0].legend(fontsize=ftsize)
ax[0].set_ylabel("$P(k)/P_\mathrm{lin}(k)$ at z=2.6", fontsize=ftsize)
ax[1].set_ylabel("$P(k)/P_\mathrm{lin}(k)$ at z=3.0", fontsize=ftsize)
ax[2].set_ylabel("$P(k)/P_\mathrm{lin}(k)$ at z=3.6", fontsize=ftsize)
# ax[3].set_ylabel("P(k)/Plin(k) at z=4.0")

# ax[2].set_xlabel("k")
ax[-1].set_xlabel("$k_\parallel [1/Mpc]$", fontsize=ftsize)

plt.xscale("log")
plt.tight_layout()
plt.savefig("forestflow_p3d_accel2.png")

# %%
# for ii in range(2):
for ii in range(pars_chain["z"].shape[0]):
    _ = pars_chain["k_kms"][ii] != 0
    kk = pars_chain["k_kms"][ii, _]
    plt.errorbar(
        kk, 
        kk/np.pi * np.mean(pars_chain["p1d_nocont"][:, ii, _], axis=0), 
        kk/np.pi * np.std(pars_chain["p1d_nocont"][:, ii, _], axis=0),
        alpha=0.5,
        color="C"+str(ii)
    )
    # plt.errorbar(
    #     kk, 
    #     kk/np.pi * np.mean(out_ari["p1d"][:, ii, _], axis=0), 
    #     kk/np.pi * np.std(out_ari["p1d"][:, ii, _], axis=0)
    # )
    
    plt.plot(
        kk, 
        kk/np.pi * np.mean(out_ari["p1d"][:, ii, _], axis=0), 
        # kk/np.pi * np.std(out_ari["p1d"][:, ii, _], axis=0)
        color="C"+str(ii)
    )
plt.xscale("log")

# %% [markdown]
# #### Load output

# %%
dict_out_all = np.load("arinyo_from_p1d_new_accel2.npy", allow_pickle=True).item()
zs = dict_out_all["zs"]
out_ari = dict_out_all["forest_out"]

# %%
params_accel2 = {
    "z": np.array([4, 3.6, 3., 2.6, 2.]),
    "bias": {
        "value": np.array([-0.404, -0.315, -0.209, -0.15, -0.0805]),
        "error": np.array([0.00676, 0.00701, 0.00453, 0.00334, 0.002]),
    },
    "beta": {
        "value": np.array([0.874, 1.07, 1.42, 1.61, 1.75]),
        "error": np.array([0.0717, 0.0752, 0.0774, 0.0819, 0.0918]),
    },
    "q1": {
        "value": np.array([2.55, 1.92, 1.08, 0.779, 0.562]),
        "error": np.array([0.121, 0.239, 0.171, 0.147, 0.125]),
    },
    "q2": {
        "value": np.array([-1.0, -0.684, -0.156, 0.0235, 0.16]),
        "error": np.array([0.349, 0.33, 0.192, 0.137, 0.0835]),
    },
    "k_v": {
        "value": np.array([2.67, 1.78, 1.01, 0.61, 0.152]),
        "error": np.array([0.856, 0.516, 0.324, 0.266, 0.15]),
    },
    "a_v": {
        "value": np.array([0.25, 0.353, 0.445, 0.394, 0.225]),
        "error": np.array([0.0781, 0.093, 0.1, 0.0916, 0.068]),
    },
    "b_v": {
        "value": np.array([1.65, 1.69, 1.74, 1.7, 1.54]),
        "error": np.array([0.131, 0.126, 0.109, 0.0907, 0.0613]),
    },
    "k_p": {
        "value": np.array([16.0, 18.4, 21.0, 21.3, 15.8]),
        "error": np.array([1.9, 4.38, 6.89, 7.61, 3.63]),
    },
}

# %%

# %%
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 10))
ftsize = 20

# target
# DESI DR1 Table 5
z = 2.33
bias = -0.1078
err_bias = 0.5 * (0.0045 + 0.0054)
beta = 1.743
err_beta = 0.5 * (0.074 + 0.1)

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

zmax = 5

# err
mean = -out_ari["bias"].mean(axis=0)
percen = np.percentile(-out_ari["bias"], [16, 84], axis=0)

_ = zs < zmax
ax[0].fill_between(zs[_], percen[0][_], percen[1][_], alpha=0.5, label="ACCEL2 ForestFlow")
# low = mean - percen[0]
# high = percen[1] - mean
# err = np.zeros((2, mean.shape[0]))
# err[0] = low
# err[1] = high

# ax[0].errorbar(zs, mean, err, label="DR1 P1D")
ax[0].errorbar(
    [z, z],
    np.zeros(2) + bias,
    np.zeros(2) + err_bias,
    label="BAO DR1",
    color="C1",
    fmt=".",
)
ax[0].errorbar(
    [z, z],
    np.zeros(2) + bias2,
    np.zeros(2) + err_bias2,
    label="BAO DR2",
    color="C2",
    fmt=".",
)

col = "C3"
for jj in range(len(hi_z)):
    if jj == 0:
        label = "Hiram BAO DR2"
    else:
        label = None
    ax[0].errorbar(
        [hi_z[jj], hi_z[jj]],
        np.zeros(2) + hi_bias[jj],
        hi_bias_err[jj : jj + 1, :].T,
        label=label,
        color=col,
        fmt=".",
    )

ax[0].errorbar(
    params_accel2["z"],
    params_accel2["bias"]["value"],
    params_accel2["bias"]["error"],
    color="C4",
    label="ACCEL2 Arinyo fit"
)

# err
mean = out_ari["beta"].mean(axis=0)
percen = np.percentile(out_ari["beta"], [16, 84], axis=0)
# low = mean - percen[0]
# high = percen[1] - mean
# err = np.zeros((2, mean.shape[0]))
# err[0] = low
# err[1] = high

_ = zs < zmax
ax[1].fill_between(zs[_], percen[0][_], percen[1][_], alpha=0.5)
ax[1].errorbar(
    [z, z], np.zeros(2) + beta, np.zeros(2) + err_beta, fmt=".", color="C1"
)
ax[1].errorbar(
    [z, z], np.zeros(2) + beta2, np.zeros(2) + err_beta2, fmt=".", color="C2"
)


for jj in range(len(hi_z)):
    ax[1].errorbar(
        [hi_z[jj], hi_z[jj]],
        np.zeros(2) + hi_beta[jj],
        hi_beta_err[jj : jj + 1, :].T,
        label=label,
        color=col,
        fmt=".",
    )

ax[1].errorbar(
    params_accel2["z"],
    params_accel2["beta"]["value"],
    params_accel2["beta"]["error"],
    color="C4",
    label="ACCEL2"
)

ax[0].legend(fontsize=ftsize)

ax[0].set_ylabel(r"$b$", fontsize=ftsize)
ax[1].set_ylabel(r"$\beta$", fontsize=ftsize)
ax[1].set_xlabel(r"$z$", fontsize=ftsize)

for ii in range(2):
    ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)

plt.tight_layout()
plt.savefig("bias_beta_BAOvsaccel2.png")
plt.savefig("bias_beta_BAOvsaccel2.pdf")

# %% [markdown]
# ### Get priors

# %%
out_ari["bias"].shape

# %%
fig, ax = plt.subplots(len(emulator.Arinyo_params), 1, sharex=True, figsize=(8, 16))

print("par", "mean", "std", "min", "max")

for ii, par in enumerate(emulator.Arinyo_params):
    if par == "bias":
        sing = -1
    else:
        sing = 1
    percen = np.percentile(sing * out_ari[par], [16, 84], axis=0)
    ax[ii].fill_between(zs, percen[0], percen[1], label="ACCEL2 DR1-like mock")
    percen = np.percentile(sing * out_ari[par], [5, 95], axis=0)
    cen = np.mean(sing * out_ari[par][:, 1])
    std = np.std(sing * out_ari[par][:, 1])

    # ax[ii].errorbar(
    #     params_accel2["z"],
    #     params_accel2[par]["value"],
    #     params_accel2[par]["error"],
    #     color="C1",
    #     label="ACCEL2"
    # )

    print(
        par,
        np.round(cen, 3),
        np.round(std, 3),
        np.round(np.min(percen[0, 1]), 3),
        np.round(np.max(percen[1, 1]), 3),
    )
    ax[ii].set_ylabel(par)
    # print(par, np.mean(out_ari[par])
ax[-1].set_xlabel(r"$z$")
plt.tight_layout()
plt.savefig("Arinyo_with_z_accel2.pdf")
plt.savefig("Arinyo_with_z_accel2.png")

# %% [markdown]
# desi

# %%
par mean std min max
bias -0.124 0.007 -0.135 -0.113
beta 1.417 0.044 1.346 1.49
q1 0.282 0.055 0.193 0.369
kvav 0.554 0.049 0.485 0.639
av 0.426 0.048 0.353 0.51
bv 1.674 0.023 1.642 1.716
kp 10.817 0.388 10.349 11.529
q2 0.27 0.059 0.182 0.375

# %% [markdown]
# accel2

# %%
bias -0.109 0.009 -0.122 -0.094
beta 1.669 0.079 1.548 1.805
q1 0.281 0.046 0.205 0.357
kvav 0.422 0.046 0.342 0.496
av 0.272 0.059 0.169 0.355
bv 1.656 0.03 1.608 1.708
kp 11.833 0.531 11.066 12.709
q2 0.12 0.042 0.062 0.198

# %%
