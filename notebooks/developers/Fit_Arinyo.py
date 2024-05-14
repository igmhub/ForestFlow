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
# # Setting best configuration for the Arinyo model

# %% [markdown]
# In this notebook we investigate the optimal range of scales and the likelihood to be used for fitting P3D and P1D using the Arinyo model

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


# %%
from lace.cosmo import camb_cosmo
import forestflow
from forestflow.model_p3d_arinyo import get_linP_interp, ArinyoModel
from forestflow.fit_p3d import FitPk
from pyDOE2.doe_lhs import lhs
from forestflow.rebin_p3d import p3d_allkmu, get_p3d_modes, p3d_rebin_mu


from forestflow.archive import GadgetArchive3D
from forestflow.P3D_cINN import P3DEmulator

def ls_level(folder, nlevels):
    for ii in range(nlevels):
        folder = os.path.dirname(folder)
    folder += "/"
    return folder


path_program = ls_level(os.getcwd(), 2)
print(path_program)
sys.path.append(path_program)

# %% [markdown]
# ## Load archive

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
# ### Priors and seeds

# %%
def get_default_params():
    parameters_q = {
        "bias": 0.12,
        "beta": 1.4,
        "q1": 0.4,
        "kvav": 0.6,
        "av": 0.3,
        "bv": 1.5,
        "kp": 18.0,
    }
    parameters_q2 = {
        "bias": 0.12,
        "beta": 1.4,
        "q1": 0.4,
        "kvav": 0.6,
        "av": 0.3,
        "bv": 1.5,
        "kp": 18.0,
        "q2": 0.2,
    }

    priors_q = {
        "bias": [0, 1],
        "beta": [0, 5.0],
        "q1": [0, 5],
        "kvav": [0.1, 5.0],
        "av": [0, 2],
        "bv": [0, 5],
        "kp": [1, 50],
    }
    priors_q2 = {
        "bias": [0, 1],
        "beta": [0, 5.0],
        "q1": [0, 5],
        "kvav": [0.1, 5.0],
        "av": [0, 2],
        "bv": [0, 5],
        "kp": [1, 50],
        "q2": [0, 5],
    }

    return parameters_q, priors_q, parameters_q2, priors_q2

def get_input_data(data, kmax_fit):
    data_dict = {}
    data_dict["z"] = np.atleast_1d(data["z"])
    data_dict["kmu_modes"] = get_p3d_modes(kmax_fit)

    data_dict["k3d_Mpc"] = data["k3d_Mpc"]
    data_dict["mu3d"] = data["mu3d"]
    data_dict["p3d_Mpc"] = data["p3d_Mpc"]

    n_modes = np.zeros_like(data_dict["k3d_Mpc"])
    for ii in range(data_dict["k3d_Mpc"].shape[0]):
        for jj in range(data_dict["k3d_Mpc"].shape[1]):
            key = f"{ii}_{jj}_k"
            if key in data_dict["kmu_modes"]:
                n_modes[ii, jj] = data_dict["kmu_modes"][key].shape[0]

    data_dict["std_p3d"] = 1 / n_modes

    data_dict["k1d_Mpc"] = data["k_Mpc"]
    data_dict["p1d_Mpc"] = data["p1d_Mpc"]
    data_dict["std_p1d"] = np.pi / data["k_Mpc"] / np.mean(n_modes.sum(axis=0))

    model = data["model"]

    return data_dict, model


# %% [markdown]
# ### IC

# %%

kmax_3d = 3
kmax_1d = 3
use_q2 = True
fit_type = "both"

list_sim_use = Archive3D.get_testing_data("mpg_central")
# list_sim_use = Archive3D.get_testing_data("mpg_seed")

isim = 6
sim_use = list_sim_use[isim]
print(sim_use["z"])

# get initial parameters for fit
_ = get_default_params()
parameters_q, priors_q, parameters_q2, priors_q2 = _
if use_q2:
    parameters = parameters_q2
    priors = priors_q2
else:
    parameters = parameters_q
    priors = priors_q

for ii, par in enumerate(parameters):
    parameters[par] = np.abs(sim_use['Arinyo'][par])
# parameters["bias"] = 0.1
parameters["beta"] = 1.5

# %% [markdown]
# ### Fit

# %%
# get input data
data_dict, model = get_input_data(sim_use,  kmax_3d)

# set fitting model
fit = FitPk(
    data_dict,
    model,
    fit_type=fit_type,
    k3d_max=kmax_3d,
    k1d_max=kmax_1d,
    priors=priors,
)

chia = fit.get_chi2(parameters)
best_fit_params = parameters.copy()
print("Initial chi2", chia)
print("and best_params", best_fit_params)

results, _best_fit_params = fit.maximize_likelihood(best_fit_params)
chi2 = fit.get_chi2(_best_fit_params)
if chi2 < chia:
    chia = chi2
    best_fit_params = _best_fit_params.copy()
# the output is chia and best_fit_params
print("Final chi2", chia)
print("and best_params", best_fit_params)

val = np.array(list(best_fit_params.values()))
res_params = val
res_chi2 = chia


# %% [markdown]
# ### Check precision

# %%
n_mubins = 4
k3d_mask = data_dict["k3d_Mpc"][:, 0] <= kmax_3d
k1d_mask = (data_dict["k1d_Mpc"] <= kmax_1d) & (
    data_dict["k1d_Mpc"] > 0
)

nk = np.sum(k3d_mask)
model_p3d, plin = p3d_allkmu(
    sim_use["model"],
    data_dict["z"][0],
    best_fit_params,
    data_dict["kmu_modes"],
    nk=nk,
    nmu=16,
    compute_plin=True,
    minimize=True
)


_ = p3d_rebin_mu(sim_use["k3d_Mpc"][:nk], 
                 sim_use["mu3d"][:nk], 
                 sim_use["p3d_Mpc"][:nk], 
                 data_dict["kmu_modes"], 
                 n_mubins=n_mubins)
knew, munew, rebin_p3d, mu_bins = _

_ = p3d_rebin_mu(sim_use["k3d_Mpc"][:nk], 
                 sim_use["mu3d"][:nk], 
                 model_p3d[:nk], 
                 data_dict["kmu_modes"], 
                 n_mubins=n_mubins)
knew, munew, rebin_model_p3d, mu_bins = _


_ = p3d_rebin_mu(sim_use["k3d_Mpc"][:nk], 
                 sim_use["mu3d"][:nk], 
                 plin[:nk], 
                 data_dict["kmu_modes"], 
                 n_mubins=n_mubins)
knew, munew, rebin_plin, mu_bins = _

model_p1d = sim_use["model"].P1D_Mpc(
    data_dict["z"][0], 
    data_dict["k1d_Mpc"],
    parameters=best_fit_params,
    minimize=True,
)
model_p1d = model_p1d[k1d_mask]

# %%
jj = 0
fig, ax = plt.subplots(2,sharex=True)
for ii in range(0, 4):
    col = f"C{jj}"
    x = knew[:, ii]    
    _ = np.isfinite(x)
    y = rebin_p3d[:,ii]/rebin_plin[:, ii]
    ax[0].plot(x[_], y[_], col+"o:")
    
    y = rebin_model_p3d[:,ii]/rebin_plin[:,ii]
    ax[0].plot(x[_], y[_], col+"-")
    
    y = rebin_model_p3d[:,ii]/rebin_p3d[:,ii] - 1
    # mask = (((data_dict["mu3d"][:12])[fit.ind_fit3d] >= np.nanmin(data_dict["mu3d"][:,ii])) 
    #         & ((data_dict["mu3d"][:12])[fit.ind_fit3d] <= np.nanmax(data_dict["mu3d"][:,ii])))
    # y = (rebin_model_p3d[_,ii]-data_dict["p3d_Mpc"][k3d_mask, ii][_])/fit.err_p3d[mask]
    ax[1].plot(x[_], y[_], col+"-")
    jj += 1
ax[1].axhline(0, linestyle=":", color="k")
ax[1].axhline(0.1, linestyle=":", color="k")
ax[1].axhline(-0.1, linestyle=":", color="k")
ax[0].set_xscale("log")
# plt.yscale("log")

# %%
fig, ax = plt.subplots(2, sharex=True)
x = data_dict["k1d_Mpc"][k1d_mask]
ax[0].plot(x, x*data_dict["p1d_Mpc"][k1d_mask], "o:")
ax[0].plot(x, x*model_p1d, "-")

y = model_p1d/data_dict["p1d_Mpc"][k1d_mask]-1
# y = (model_p1d-data_dict["p1d_Mpc"][k1d_mask])/fit.err_p1d
ax[1].plot(x, y, "-")
ax[1].axhline(0, linestyle=":", color="k")
ax[1].axhline(0.01, linestyle=":", color="k")
ax[1].axhline(-0.01, linestyle=":", color="k")
ax[0].set_xscale("log")

# %%

# %% [markdown]
# # END HERE, characterize P3D and P1D below

# %% [markdown]
# ## Arinyo model from default cosmo and params
#
# For more details about the Arinyo model see Eq. 4.5 from Givans+22 (https://arxiv.org/abs/2205.00962)

# %%
zs = np.array([3]) # set target redshift
cosmo = camb_cosmo.get_cosmology() # set default cosmo
camb_results = camb_cosmo.get_camb_results(cosmo, zs=zs, camb_kmax_Mpc=200) # set default cosmo
arinyo = ArinyoModel(cosmo=cosmo, camb_results=camb_results, zs=zs, camb_kmax_Mpc=200) # set model
arinyo.default_params

# %% [markdown]
# ### Compute P3D & P1D

# %%
nn_k = 200 # number of k bins
nn_mu = 10 # number of mu bins
k = np.logspace(-1.5, 1, nn_k)
mu = np.linspace(0, 1, nn_mu)
k2d = np.tile(k[:, np.newaxis], nn_mu) # k grid for P3D
mu2d = np.tile(mu[:, np.newaxis], nn_k).T # mu grid for P3D

kpar = np.logspace(-1, np.log10(5), nn_k) # kpar for P1D

plin = arinyo.linP_Mpc(zs[0], k) # get linear power spectrum at target z
p3d = arinyo.P3D_Mpc(zs[0], k2d, mu2d, arinyo.default_params) # get P3D at target z
p1d = arinyo.P1D_Mpc(zs[0], kpar, parameters=arinyo.default_params) # get P1D at target z

# %% [markdown]
# #### Plot P3D

# %%
plot = False
if plot:
    for ii in range(p3d.shape[1]):
        col = 'C'+str(ii)
        if ii % 3 == 0:
            lab = r'$<\mu>=$'+str(np.round(mu[ii], 2))
        else:
            lab = None
        plt.loglog(k, p3d[:, ii]/plin, col, label=lab)
        plt.plot(k, p3d[0, ii]/plin[0]+k[:]*0, col+'--')
    plt.xlabel(r'$k$ [Mpc]')
    plt.ylabel(r'$P/P_{\rm lin}$')
    plt.legend(loc='upper left')

# %% [markdown]
# #### Plot P1D

# %%
plot = False
if plot:
    plt.plot(kpar, p1d)
    plt.xlabel(r'$k$ [Mpc]')
    plt.ylabel(r'$P_{\rm 1D}(k)$')
    plt.xscale('log')

# %% [markdown]
# ### Best-fitting Arinyo model to LaCE simulation
#
# The Arinyo model was optimized to reproduce both the P3D and P1D down to 4 Mpc (both)

# %%
ind_book = 6 # select a random simulation

zs = np.array([Archive3D.training_data[ind_book]['z']]) 

k3d_Mpc = Archive3D.training_data[ind_book]['k3d_Mpc']
mu3d = Archive3D.training_data[ind_book]['mu3d']
p3d_Mpc = Archive3D.training_data[ind_book]['p3d_Mpc']
Plin = Archive3D.training_data[ind_book]['Plin']

k1d_Mpc = Archive3D.training_data[ind_book]['k_Mpc']
p1d_Mpc = Archive3D.training_data[ind_book]['p1d_Mpc']

arinyo_params = Archive3D.training_data[ind_book]['Arinyo_minin'] # best-fitting Arinyo params
print(zs)
arinyo_params

# %% [markdown]
# #### Plot P3D from simulation

# %%
plot = False
if plot:
    for ii in range(0, p3d_Mpc.shape[1]):
        if ii % 2 == 0:
            lab = r'$<\mu>=$'+str(np.round(mu3d[-1,ii], 2))
        else:
            lab = None
        plt.loglog(k3d_Mpc[:, ii], p3d_Mpc[:, ii]/Plin[:, ii], label=lab)
    plt.xlabel(r'$k$ [Mpc]')
    plt.ylabel(r'$P/P_{\rm lin}$')
    plt.legend(loc='lower left')

# %% [markdown]
# #### Compare P3D from simulation and best-fitting Arinyo

# %%
model_p3d = Archive3D.training_data[ind_book]['model'].P3D_Mpc(zs, k3d_Mpc, mu3d, arinyo_params)

# %%
plot = False
if plot:
    jj = 0
    mask = k3d_Mpc[:,0] < 5
    for ii in range(0, p3d_Mpc.shape[1], 2):
        col = 'C'+str(jj)
        lab = r'$<\mu>=$'+str(np.round(np.nanmean(mu3d[:,ii]), 2))
        plt.plot(k3d_Mpc[mask, ii], p3d_Mpc[mask, ii]/Plin[mask, ii], col+'-', label=lab)
        plt.plot(k3d_Mpc[mask, ii], model_p3d[mask, ii]/Plin[mask, ii], col+'--')
        jj += 1
    plt.xscale('log')
    plt.xlabel(r'$k$ [Mpc]')
    plt.ylabel(r'$P/P_{\rm lin}$')
    plt.legend()

# %% [markdown]
# #### Ratio

# %%
plot = False
if plot:
    mask = k3d_Mpc[:,0] < 5
    for ii in range(0, p3d_Mpc.shape[1]):
        lab = r'$<\mu>=$'+str(np.round(np.nanmean(mu3d[:,ii]), 2))
        plt.plot(k3d_Mpc[mask, ii], p3d_Mpc[mask, ii]/model_p3d[mask, ii]-1, label=lab)
    plt.plot(k3d_Mpc[mask, 0], k3d_Mpc[mask, 0]*0, 'k--')
    plt.xscale('log')
    plt.xlabel(r'$k$ [Mpc]')
    plt.ylabel(r'$P/P_{\rm lin}$')
    plt.legend()

# %% [markdown]
# #### Same for P1D

# %%
model_p1d = Archive3D.training_data[ind_book]['model'].P1D_Mpc(zs, k1d_Mpc, parameters=arinyo_params)

# %%
plot = False
if plot:
    mask = k1d_Mpc < 5
    plt.plot(k1d_Mpc[mask], p1d_Mpc[mask]/model_p1d[mask]-1, '-', label='Sim/Model-1')
    plt.plot(k1d_Mpc[mask], k1d_Mpc[mask]*0, 'k--')
    plt.xscale('log')
    plt.xlabel(r'$k$ [Mpc]')
    plt.ylabel(r'$P_{\rm 1D}$')
    plt.legend()


# %% [markdown]
# ### Old fits

# %%
def get_std_kp1d(ind_sim, ind_tau, ind_z, err_p1d):
    _tau = np.argwhere(err_p1d["u_ind_tau"] == ind_tau)[0, 0]
    _z = np.argwhere(err_p1d["u_ind_z"] == ind_z)[0, 0]
    if(ind_sim != "mpg_central"):
        _sim = np.argwhere(err_p1d["u_ind_sim"] == int(ind_sim[4:]))[0, 0]
        av_pk = err_p1d["p1d_sim_tau_z"][_sim, _tau, _z] * err_p1d["k"] / np.pi
        std_kpk = err_p1d["sm_rel_err"] * av_pk
    else:
        av_pk = np.mean(err_p1d["p1d_sim_tau_z"][:, _tau, _z], axis=0) * err_p1d["k"] / np.pi
        std_kpk = err_p1d["sm_rel_err"] * av_pk
    return std_kpk


def get_std_kp3d(ind_sim, ind_tau, ind_z, err_p3d, sm_pk=False):
    _tau = np.argwhere(err_p3d["u_ind_tau"] == ind_tau)[0, 0]
    _z = np.argwhere(err_p3d["u_ind_z"] == ind_z)[0, 0]
    norm = err_p3d["k"] ** 3 / 2 / np.pi**2
    if(ind_sim != "mpg_central"):
        _sim = np.argwhere(err_p3d["u_ind_sim"] == int(ind_sim[4:]))[0, 0]
        if sm_pk:
            pk = err_p3d["sm_p3d_sim_tau_z"][_sim, _tau, _z]
        else:
            pk = err_p3d["p3d_sim_tau_z"][_sim, _tau, _z]    
        av_pk = pk * norm
    else:
        av_pk = np.mean(err_p3d["p3d_sim_tau_z"][:, _tau, _z], axis=0) * norm
    std_kpk = err_p3d["sm_rel_err"] * av_pk
    return std_kpk

