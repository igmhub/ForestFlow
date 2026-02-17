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
# - Get P1D data from ACCEL2 and compute best-fitting model using cup1d (done separately)
# - Read cup1d chain and evaluate ForestFlow to get Arinyo, P1D, and P3D
# - Compare ForestFlow P1D with lace-mpg P1D
# - Compare P1D and P3D from ForestFlow with ACCEL2 predictions
#
# - Compare ACCEL2 P1D with best-fitting P1D to DESIY1

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
# # Read chain using cup1d (DESI DR1)

# %% [markdown]
# #### Set cup1d

# %%
pip = Pipeline()

# %% [markdown]
# #### Load chain

# %%
# local
base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1"
folder = os.path.join(base, "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7/")
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

zs = pip.fitter.like.data.z.copy()
nn = 100

ind = np.random.permutation(np.arange(chain.shape[0]))[:nn]
pars_chain = {}

# pars_chain["As"] = np.zeros((ind.shape[0]))
# pars_chain["ns"] = np.zeros((ind.shape[0]))
pars_chain["z"] = zs
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
    nelem = len(p1d[0][jj])
    pars_chain["k_kms"][jj, :nelem] = pip.fitter.like.data.k_kms[jj]

# %%
# chi2_ev = np.zeros((500))
# for ii in range(500):
#     chi2_ev[ii] = pip.fitter.like.get_chi2(chain[ind[ii], :])

# %%
ii = 88
pip.fitter.like.plot_p1d(chain[ind[ii], :], print_chi2=False)

# pip.fitter.like.plot_p1d(p0, print_chi2=False)

# %% [markdown]
# #### From each point, we get the input parameters to ForestFlow
#
# Most of the time goes into calling CAMB

# %%
# only need to call it once since we are only chaing As and ns during inference
ii = 0
fid_cosmo = {
    "H0": 67.66,
    "mnu": 0,
    "omch2": 0.119,
    "ombh2": 0.0224,
    "omk": 0,
    'As': 2.105e-09,
    # "As": pars_chain["As"][ii],
    'ns': 0.9665,
    # "ns": pars_chain["ns"][ii],
    "nrun": 0.0,
    "pivot_scalar": 0.05,
    "w": -1.0,
}
sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(fid_cosmo)
# compute linear power parameters at each z (in Mpc units)
kp_Mpc = 0.7
linP_zs = fit_linP.get_linP_Mpc_zs(sim_cosmo, zs, kp_Mpc)
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
# ii = 0
# tar_cosmo = {
#     "H0": 67.66,
#     "mnu": 0,
#     "omch2": 0.119,
#     "ombh2": 0.0224,
#     "omk": 0,
#     # 'As': 2.105e-09,
#     "As": pars_chain["As"][ii],
#     # 'ns': 0.9665,
#     "ns": pars_chain["ns"][ii],
#     "nrun": 0.0,
#     "pivot_scalar": 0.05,
#     "w": -1.0,
# }

# %%
# res_linP_zs = rescale_linP(fid_cosmo, tar_cosmo, linP_zs)

# %%
# for ii in range(len(res_linP_zs)):
#     print()
#     print(res_linP_zs[ii])
#     print(linP_zs[ii])


# %%

# %%
for ii in range(ind.shape[0]):
    if ii % 10 == 0:
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
    
    tar_cosmo = {
        "As": chain_params[0].value_from_cube(chain[ind[ii], 0]),
        "ns": chain_params[1].value_from_cube(chain[ind[ii], 1]),
        "nrun":0
    }
    res_linP_zs = rescale_linP(fid_cosmo, tar_cosmo, linP_zs)
    for jj in range(zs.shape[0]):
        nelem = len(p1d[0][jj])
        pars_chain["p1d"][ii, jj, :nelem] = p1d[0][jj]
        pars_chain["p1d_nocont"][ii, jj, :nelem] = p1d_no[0][jj]
        
        pars_chain["Delta2_p"][ii, jj] = res_linP_zs[jj]["Delta2_p"]
        pars_chain["n_p"][ii, jj] = res_linP_zs[jj]["n_p"]

    # break

# %%
np.save("inter_chain.npy", pars_chain)

# %%
pars_chain = np.load("inter_chain.npy", allow_pickle=True).item()

# %% [markdown]
# load model to evaluate IGM parameters

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

# %% [markdown]
# ### old

# %%
from forestflow.archive import GadgetArchive3D
path_program = os.path.dirname(forestflow.__path__[0]) + '/'
folder_lya_data = path_program + "/data/best_arinyo/"

Archive3D = GadgetArchive3D(
    base_folder=path_program[:-1],
    folder_data=folder_lya_data,
    force_recompute_plin=False,
    average="both",
)
print(len(Archive3D.training_data))

# %%


from forestflow.old_code.paper_P3D_cINN import P3DEmulator as old_P3DEmulator


emulator = old_P3DEmulator(
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

# %% [markdown]
# ### new

# %%
emulator = P3DEmulator(
    model_path = path_repo + "/data/emulator_models/new_emu3",
)

# %% [markdown]
# #### Initiate output to store Arinyo parameters

# %%
pars = pars_chain.keys()
pars_ari = ["bias", "beta", "q1", "kvav", "av", "bv", "kp", "q2"]

out_ari = {}
for par in pars_ari:
    out_ari[par] = np.zeros_like(pars_chain["mF"])

out_ari["p1d"] = np.zeros_like(pars_chain["p1d_nocont"])

# %%

from forestflow.archive import get_camb_interp
from forestflow.model_p3d_arinyo import ArinyoModel

# %%
ii = 0
cosmo = {
    "H0": 67.66,
    "mnu": 0,
    "omch2": 0.119,
    "ombh2": 0.0224,
    "omk": 0,
    'As': 2.105e-09,
    # "As": pars_chain["As"][ii],
    'ns': 0.9665,
    # "ns": pars_chain["ns"][ii],
    "nrun": 0.0,
    "pivot_scalar": 0.05,
    "w": -1.0,
}
pk_interp = get_camb_interp({"cosmo_params": cosmo})
model_Arinyo = ArinyoModel(camb_pk_interp=pk_interp)

# %%
import types
from forestflow.camb_routines import P_camb

# %%
get_linpower = types.MethodType(P_camb, pk_interp)

# %%
# self.linP_interp.P(z, k_Mpc, grid=False)
k_Mpc = np.geomspace(1e-4, 1, 100)
pklin = get_linpower(2., k_Mpc, grid=False)

# %%
plt.loglog(k_Mpc, pklin)

# %%
ii = 0
tar_cosmo = {
    "As": chain_params[0].value_from_cube(chain[ind[ii], 0]),
    "ns": chain_params[1].value_from_cube(chain[ind[ii], 1]),
    # 'ns': 0.9665,
    "nrun":0
}

res_linP_zs = rescale_linP(fid_cosmo, tar_cosmo, linP_zs)

def rescale_pklin(zz, k_Mpc, fun_linpower, fid_cosmo, tar_cosmo, kp_Mpc=0.7, ks_Mpc=0.05):

    ratio_As = tar_cosmo["As"] / fid_cosmo["As"]
    delta_ns = tar_cosmo["ns"] - fid_cosmo["ns"]
    delta_nrun = tar_cosmo["nrun"] - fid_cosmo["nrun"]
    
    ln_kp_ks = np.log(kp_Mpc / ks_Mpc)
    delta_alpha_p = delta_nrun
    delta_n_p = delta_ns + delta_nrun * ln_kp_ks
    ln_ratio_A_p = (
        np.log(ratio_As)
        + (delta_ns + 0.5 * delta_nrun * ln_kp_ks) * ln_kp_ks
    )
    
    rotk = np.log(k_Mpc / kp_Mpc)    
    pklin = get_linpower(zz, k_Mpc, grid=False)
    
    pklin_rescaled = pklin * np.exp(
        ln_ratio_A_p
        + delta_n_p * rotk
        + 0.5 * delta_alpha_p * rotk**2
    )

    return pklin, pklin_rescaled


# %%
ratio_As

# %%
delta_ns

# %%
ii = 0
cosmo = {
    "H0": 67.66,
    "mnu": 0,
    "omch2": 0.119,
    "ombh2": 0.0224,
    "omk": 0,
    # 'As': 2.105e-09,
    "As": pars_chain["As"][ii],
    # 'ns': 0.9665,
    "ns": pars_chain["ns"][ii],
    "nrun": 0.0,
    "pivot_scalar": 0.05,
    "w": -1.0,
}
pk_interp = get_camb_interp({"cosmo_params": cosmo})
model_Arinyo = ArinyoModel(camb_pk_interp=pk_interp)
get_linpower = types.MethodType(P_camb, pk_interp)
pklin2 = get_linpower(2., k_Mpc, grid=False)

# %%
delta_ns

# %%

k_Mpc = np.geomspace(1e-4, 1, 100)
zz = 2
pklin, pklin_res = rescale_pklin(zz, k_Mpc, get_linpower, fid_cosmo, tar_cosmo)

# %%

# %%
plt.loglog(k_Mpc, pklin)
plt.loglog(k_Mpc, pklin2)
plt.loglog(k_Mpc, pklin_res)

# %%
np.mean(pklin_res/pklin2)

# %%
plt.loglog(k_Mpc, pklin)
plt.loglog(k_Mpc, pklin_new)
plt.loglog(k_Mpc, pklin2, alpha=0.5)
plt.loglog(k_Mpc, pklin_res, alpha=0.5)

# %%
np.mean(pklin_res/pklin2)

# %%
# plt.loglog(k_Mpc, pklin)
plt.plot(k_Mpc, pklin_new/pklin)
plt.plot(k_Mpc, pklin2/pklin, alpha=0.5)
plt.xscale("log")

# %%

# %%
# %%time
pk_interp = get_camb_interp({"cosmo_params": cosmo})

# %% [markdown]
# scale interpolator

# %%

# %%

# %%

# %%
# # %%time
# chain step
for ii in range(pars_chain["mF"].shape[0]):
    if ii % 10 == 0:
        print(ii)
    list_input_emu = []
    # redshift
    for jj in range(pars_chain["mF"].shape[1]):
        input_emu = {}
        for par in pars:
            if par not in ['Delta2_p', 'n_p', 'mF', 'gamma', 'sigT_Mpc', 'kF_Mpc']:
                continue
            input_emu[par] = pars_chain[par][ii, jj]
        list_input_emu.append(input_emu)

    # out = emulator.predict_Arinyos(
    #     emu_params=list_input_emu, Nrealizations=3000
    # )

    # for jj in range(pars_chain["mF"].shape[1]):
    #     for kk, par in enumerate(pars_ari):
    #         out_ari[par][ii, jj] = out[jj, kk]

    # cosmo = {
    #     "H0": 67.66,
    #     "mnu": 0,
    #     "omch2": 0.119,
    #     "ombh2": 0.0224,
    #     "omk": 0,
    #     # 'As': 2.105e-09,
    #     "As": pars_chain["As"][ii],
    #     # 'ns': 0.9665,
    #     "ns": pars_chain["ns"][ii],
    #     "nrun": 0.0,
    #     "pivot_scalar": 0.05,
    #     "w": -1.0,
    # }
    # sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(cosmo)

    for jj in range(pars_chain["mF"].shape[1]):
        _ = pars_chain["k_kms"][jj] != 0
        
        # dkms_dMpc = camb_cosmo.dkms_dMpc(sim_cosmo, z=np.array([pars_chain["z"][jj]]))[0]
        info_power = {
            "z": pars_chain["z"][jj],
            "k1d_Mpc": pars_chain["k_kms"][jj, _] * dkms_dMpc_zs[jj],
            "return_p1d": True,
            # "return_cov": True,
        }
        
        out = emulator.evaluate_arinyo(
            list_input_emu[jj],
            model_Arinyo,
            info_power=info_power,
            # natural_params=True,
            Nrealizations=3000
        )
        out_ari["p1d"][ii, jj, :np.sum(_)]  = out["p1d"] * dkms_dMpc_zs[jj]

        break
    # if ii > 10:
    break


# %%
list_input_emu

# %%
out["p1d"]

# %%
out_ari["p1d"][0, ii, _]

# %%
# for ii in range(2):
for ii in range(pars_chain["z"].shape[0]):
    _ = pars_chain["k_kms"][ii] != 0
    kk = pars_chain["k_kms"][ii, _]
    # plt.errorbar(
    #     kk, 
    #     kk/np.pi * np.mean(pars_chain["p1d_nocont"][:, ii, _], axis=0), 
    #     kk/np.pi * np.std(pars_chain["p1d_nocont"][:, ii, _], axis=0),
    #     alpha=0.5
    # )
    # plt.errorbar(
    #     kk, 
    #     kk/np.pi * np.mean(out_ari["p1d"][:, ii, _], axis=0), 
    #     kk/np.pi * np.std(out_ari["p1d"][:, ii, _], axis=0)
    # )
    # plt.plot(kk, kk/np.pi * pars_chain["p1d_nocont"][0, ii, _], "C"+str(ii) +"-", alpha=0.5)
    # plt.plot(kk, kk/np.pi * out_ari["p1d"][0, ii, _], "C"+str(ii) +"--")
    # plt.plot(kk, np.mean(out_ari["p1d"][:10, ii, _]/pars_chain["p1d_nocont"][:10, ii, _], axis=0)-1)
    plt.plot(kk, out_ari["p1d"][0, ii, _]/pars_chain["p1d_nocont"][0, ii, _]-1)
plt.xscale("log")

# %%
np.nanmean(out_ari["p1d"][0, :, _]/pars_chain["p1d_nocont"][0, :, _]-1)

# %%
0.06090075363636157

# %%
np.nanstd(out_ari["p1d"][0, :, _]/pars_chain["p1d_nocont"][0, :, _]-1)

# %%

# %%

# %%
0.015587160902410607

# %%
2%, 8%

# %%

# %%

# %% [markdown]
# ### huge errors!!!

# %%

# %%
out_ari["bias"] = -out_ari["bias"]

# %% [markdown]
# #### Store output for future use

# %%
save = True
if save:
    dict_out_all = {}
    dict_out_all["emu_params"] = pars_chain
    dict_out_all["ari_params"] = out_ari
    dict_out_all["zs"] = zs
    np.save("arinyo_from_p1d_new.npy", dict_out_all)

# %% [markdown]
# #### Load output

# %%
dict_out_all = np.load("arinyo_from_p1d_new.npy", allow_pickle=True).item()
zs = dict_out_all["zs"]
out_ari = dict_out_all["ari_params"]

# %%
dict_out_all.keys()

# %%
out_ari["bias"].shape

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


# err
mean = out_ari["bias"].mean(axis=0)
percen = np.percentile(out_ari["bias"], [16, 84], axis=0)

_ = zs < 3
ax[0].fill_between(zs[_], percen[0][_], percen[1][_], alpha=0.5, label="P1D DR1")
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

# err
mean = out_ari["beta"].mean(axis=0)
percen = np.percentile(out_ari["beta"], [16, 84], axis=0)
# low = mean - percen[0]
# high = percen[1] - mean
# err = np.zeros((2, mean.shape[0]))
# err[0] = low
# err[1] = high

_ = zs < 3
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

ax[0].legend(fontsize=ftsize)

ax[0].set_ylabel(r"$b$", fontsize=ftsize)
ax[1].set_ylabel(r"$\beta$", fontsize=ftsize)
ax[1].set_xlabel(r"$z$", fontsize=ftsize)

for ii in range(2):
    ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)

plt.tight_layout()
plt.savefig("bias_beta_BAOvsP1D.png")
plt.savefig("bias_beta_BAOvsP1D.pdf")

# %% [markdown]
# ### Get priors

# %%
zs[1]

# %%
fig, ax = plt.subplots(len(out_ari.keys()), 1, sharex=True, figsize=(8, 16))

print("par", "mean", "std", "min", "max")

for ii, par in enumerate(out_ari.keys()):
    percen = np.percentile(out_ari[par], [16, 84], axis=0)
    ax[ii].fill_between(zs, percen[0], percen[1])
    percen = np.percentile(out_ari[par], [5, 95], axis=0)
    cen = np.mean(out_ari[par][:, 1])
    std = np.std(out_ari[par][:, 1])
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
plt.savefig("Arinyo_with_z.pdf")
plt.savefig("Arinyo_with_z.png")

# %%
