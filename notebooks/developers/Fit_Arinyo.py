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
from forestflow.fit_p3dz import FitPkz
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

# %%
list_central = Archive3D.get_testing_data("mpg_central")
list_seed = Archive3D.get_testing_data("mpg_seed")

list_merge = []
zlist = []
par_merge = ["mF", "T0", "gamma", "sigT_Mpc", "kF_Mpc"]
for ii in range(len(list_central)):
    cen = list_central[ii]
    seed = list_seed[ii]
    zlist.append(cen["z"])
    # print(cen["z"], seed["z"])

    tar = cen.copy()
    for par in par_merge:
        tar[par] = 0.5 * (cen[par] + seed[par])
        
    tar["p1d_Mpc"] = (cen["mF"]**2 * cen["p1d_Mpc"] + seed["mF"]**2 * seed["p1d_Mpc"]) / tar["mF"]**2 / 2
    tar["p3d_Mpc"] = (cen["mF"]**2 * cen["p3d_Mpc"] + seed["mF"]**2 * seed["p3d_Mpc"]) / tar["mF"]**2 / 2

    # print(cen["mF"], seed["mF"], tar["mF"])
    list_merge.append(tar)

# %%
from forestflow.utils import params_numpy2dict_minimizerz

def paramz_to_paramind(z, paramz):
    paramind = []
    for ii in range(len(z)):
        param = {}
        for key in paramz:
            param[key] = 10 ** np.poly1d(paramz[key])(z[ii])
        paramind.append(param)
    return paramind

file = path_program + "/data/best_arinyo/minimizer_z/fit_sim_label_combo_val_scaling_1_kmax3d_3_kmax1d_3.npz"
data = np.load(file, allow_pickle=True)
best_params = paramz_to_paramind(zlist, data["best_params"].item())

for ii in range(len(list_merge)):
    list_merge[ii]["Arinyo_minz"] = params_numpy2dict_minimizerz(best_params[ii])


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

def get_default_paramsz(sim_label, z, val_scaling=1):

    # names = ["bias", "beta", "q1", "kvav", "av", "bv", "kp", "q2"]
    names = ["bias", "beta", "q1", "kvav", "av", "bv", "kp"]
    
    priors = {
        "bias": [1e-3, 3],
        "beta": [5e-3, 7],
        "q1": [1e-2, 8],
        "kvav": [1e-3, 5],
        "av": [1e-3, 2],
        "bv": [1e-1, 5],
        "kp": [3, 30],
        # "q2": [0, 5],
    }

    folder = "/home/jchaves/Proyectos/projects/lya/ForestFlow/data/best_arinyo/minimizer/"
    file = f"fit_sim_label_{sim_label}_kmax3d_3_kmax1d_3.npz"
    dat = np.load(folder+file)
    in_parameters = {}
    _ = dat["val_scaling"] == val_scaling
    # param_ind = dat["best_params"][_, :, 0]
    # order = np.array([2, 2, 1, 1, 2, 0, 1, 0])
    param_ind = dat["best_params"][_, :-1, 0]
    order = np.array([2, 2, 1, 1, 2, 0, 1])
    for ii in range(param_ind.shape[1]):
        _ = param_ind[:,ii] < priors[names[ii]][0]
        param_ind[_,ii] = priors[names[ii]][0]
        
        _ = param_ind[:,ii] > priors[names[ii]][1]
        param_ind[_,ii] = priors[names[ii]][1]

        res = np.polyfit(z, np.log10(param_ind[:,ii]), deg=order[ii]) 
        p = 10 ** np.poly1d(res)(z)
        print(names[ii], priors[names[ii]][0], np.min(p), priors[names[ii]][1], np.max(p))
        in_parameters[names[ii]] = res

    return in_parameters, param_ind, order, priors

def get_input_dataz(list_data, kmax_fit):
    data_dict = {}
    
    data_dict["z"] = np.array([d["z"] for d in list_data])
    data_dict["kmu_modes"] = get_p3d_modes(kmax_fit)
    nz = data_dict["z"].shape[0]

    data_dict["k3d_Mpc"] = list_data[0]["k3d_Mpc"]
    data_dict["mu3d"] = list_data[0]["mu3d"]
    data_dict["k1d_Mpc"] = list_data[0]["k_Mpc"]
    
    n_modes = np.zeros_like(data_dict["k3d_Mpc"])
    for ii in range(data_dict["k3d_Mpc"].shape[0]):
        for jj in range(data_dict["k3d_Mpc"].shape[1]):
            key = f"{ii}_{jj}_k"
            if key in data_dict["kmu_modes"]:
                n_modes[ii, jj] = data_dict["kmu_modes"][key].shape[0]
    
    data_dict["p3d_Mpc"] = np.zeros((nz, *data_dict["k3d_Mpc"].shape))
    data_dict["std_p3d"] = np.zeros_like(data_dict["p3d_Mpc"])
    data_dict["p1d_Mpc"] = np.zeros((nz, data_dict["k1d_Mpc"].shape[0]))
    data_dict["std_p1d"] = np.zeros_like(data_dict["p1d_Mpc"])
    model = []
    for ii in range(nz):
        data_dict["p3d_Mpc"][ii] = list_data[ii]["p3d_Mpc"]
        data_dict["p1d_Mpc"][ii] = list_data[ii]["p1d_Mpc"]
        normz = data_dict["z"][ii]**3
        data_dict["std_p3d"][ii] = normz / n_modes * data_dict["k3d_Mpc"]**0.25
        data_dict["std_p1d"][ii] = normz * np.pi / data_dict["k1d_Mpc"] / np.mean(n_modes.sum(axis=0))
        model.append(list_data[ii]["model"])

    return data_dict, model


# %% [markdown]
# ### IC

# %%
Archive3D.list_sim_test

# %%

# %%
kmax_3d = 3
kmax_1d = 3
use_q2 = True
fit_type = "both"
sim_label = "mpg_central"
sim_label = "mpg_seed"
sim_label = "combo"

if(sim_label == "combo"):
    list_sim_use = list_merge
    sim_param_input = "mpg_central"
else:
    list_sim_use = Archive3D.get_testing_data(sim_label)
    sim_param_input = sim_label

data_dict, model = get_input_dataz(list_sim_use,  kmax_3d)

# get initial parameters for fit
parameters, param_ind, order, priors = get_default_paramsz(sim_param_input, data_dict["z"])

# parameters = best_fit_params.copy()
# parameters["bias"] = np.array([-0.02183114,  0.48450093, -1.90681541])

params_minimizer = np.concatenate(np.array(list(parameters.values())))
names = np.array(list(parameters.keys())).reshape(-1)

# %%
parameters = {'bias': np.array([-0.02612176,  0.49865012, -1.91246964]),
 'beta': np.array([-0.0739654 ,  0.2916148 , -0.05783508]),
 'q1': np.array([ 0.26445976, -0.91410792]),
 'kvav': np.array([ 0.18747969, -0.80355048]),
 'av': np.array([-0.16330166,  1.09833536, -2.28063046]),
 'bv': np.array([0.25569833]),
 'kp': np.array([0.06015869, 0.92459092])}

# %%
np.concatenate(list(parameters.values()))

# %% [markdown]
# ### Fit

# %%
# set fitting model
fit = FitPkz(
    data_dict,
    model,
    names=names,
    priors=priors,
    fit_type=fit_type,
    k3d_max=kmax_3d,
    k1d_max=kmax_1d,
    order=order,
    verbose=True
)

chia = fit.get_chi2(params_minimizer)
print("Initial chi2", chia)

# %%
# params_minimizer = np.concatenate(np.array(list(best_fit_params.values())))

# %%
results, best_fit_params = fit.maximize_likelihood(params_minimizer)
params_minimizer = np.concatenate(np.array(list(best_fit_params.values())))
chi2 = fit.get_chi2(params_minimizer)
print("Final chi2", chi2)
print("and best_params", best_fit_params)


# %%
def get_flag_out(ind_sim, val_scaling, kmax_3d, kmax_1d):
    flag = (
        "fit_sim_label_"
        + str(ind_sim)
        + "_val_scaling_"
        + str(np.round(val_scaling, 2))
        + "_kmax3d_"
        + str(kmax_3d)
        + "_kmax1d_"
        + str(kmax_1d)
    )
    return flag

folder_save = forestflow.__path__[0][:-10] + "/data/best_arinyo/minimizer_z/"
val_scaling = 1

out_file = folder_save + get_flag_out(
    "combo", val_scaling, kmax_3d, kmax_1d
)

print(out_file)

# %%
np.savez(out_file, chi2=chi2, best_params=best_fit_params)
print("Saved to", out_file)

# %% [markdown]
# #### check redshift dependence

# %%
# in_parameters = {}
# # 4?
order = np.array([2, 2, 1, 1, 2, 0, 1])

for ii, key in enumerate(best_fit_params):
    if(ii == 1):
        pass
    else:
        continue
    col = "C"+str(ii)
    res2 = best_fit_params[key]
    p2 = 10 ** np.poly1d(res2)(data_dict["z"])
    res = np.polyfit(data_dict["z"], np.log10(param_ind[:,ii]), deg=order[ii])
    p = 10 ** np.poly1d(res)(data_dict["z"])

    # res3 = np.polyfit(data_dict["z"], np.log10(p2), deg=2)
    res3 = best_fit_params_long[key]
    p3 = 10 ** np.poly1d(res3)(data_dict["z"])

    # print(p2/p3)
    
    plt.plot(data_dict["z"], p2, col+"-")
    plt.plot(data_dict["z"], p3, col+".-")
    plt.plot(data_dict["z"], p, col+"--")
    plt.plot(data_dict["z"], param_ind[:, ii], col+":o", label=key)
plt.yscale("log")
# plt.ylim(1e-1, 20)
plt.legend(ncol=2)

# %%
best_fit_params2 = best_fit_params.copy()

# %%
paramz = []
for ii in range(fit.nz):
    param = {}
    for jj, key in enumerate(best_fit_params2):
        param[key] = 10 ** np.poly1d(best_fit_params2[key])(fit.data["z"][ii])
    paramz.append(param)

# %% [markdown]
# ### Check precision

# %%
for iz in range(len(data_dict["z"])):

    if(data_dict["z"][iz] == 2) | (data_dict["z"][iz] == 3):
        pass
    else:
        continue

    n_mubins = 4
    k3d_mask = data_dict["k3d_Mpc"][:, 0] <= kmax_3d
    k1d_mask = (data_dict["k1d_Mpc"] <= kmax_1d) & (
        data_dict["k1d_Mpc"] > 0
    )
    
    nk = np.sum(k3d_mask)
    model_p3d, plin = p3d_allkmu(
        list_sim_use[iz]["model"],
        data_dict["z"][iz],
        paramz[iz],
        data_dict["kmu_modes"],
        nk=nk,
        nmu=16,
        compute_plin=True,
        minimize=True
    )
    
    
    _ = p3d_rebin_mu(list_sim_use[iz]["k3d_Mpc"][:nk], 
                     list_sim_use[iz]["mu3d"][:nk], 
                     list_sim_use[iz]["p3d_Mpc"][:nk], 
                     data_dict["kmu_modes"], 
                     n_mubins=n_mubins)
    knew, munew, rebin_p3d, mu_bins = _
    
    _ = p3d_rebin_mu(list_sim_use[iz]["k3d_Mpc"][:nk], 
                     list_sim_use[iz]["mu3d"][:nk], 
                     model_p3d[:nk], 
                     data_dict["kmu_modes"], 
                     n_mubins=n_mubins)
    knew, munew, rebin_model_p3d, mu_bins = _
    
    
    _ = p3d_rebin_mu(list_sim_use[iz]["k3d_Mpc"][:nk], 
                     list_sim_use[iz]["mu3d"][:nk], 
                     plin[:nk], 
                     data_dict["kmu_modes"], 
                     n_mubins=n_mubins)
    knew, munew, rebin_plin, mu_bins = _
    
    model_p1d = list_sim_use[iz]["model"].P1D_Mpc(
        data_dict["z"][iz], 
        data_dict["k1d_Mpc"],
        parameters=paramz[iz],
        minimize=True,
    )
    model_p1d = model_p1d[k1d_mask]
    
    jj = 0
    fig, ax = plt.subplots(3, sharex=True)
    for ii in range(0, 4):
        col = f"C{jj}"
        x = knew[:, ii]    
        _ = np.isfinite(x)
        y = rebin_p3d[:,ii]/rebin_plin[:, ii]
        ax[0].plot(x[_], y[_], col+"o:")
        
        y = rebin_model_p3d[:,ii]/rebin_plin[:,ii]
        ax[0].plot(x[_], y[_], col+"-")
        
        y = rebin_model_p3d[:,ii]/rebin_p3d[:,ii] - 1
        ax[1].plot(x[_], y[_], col+"-")

        # norm = 1
        # norm = (data_dict["z"][iz])**3.5
        y = (rebin_p3d[:,ii]-rebin_model_p3d[:,ii])/data_dict["std_p3d"][iz][:12, ii*5]
        ax[2].plot(x[_], y[_], col+"-")

        
        jj += 1
    ax[1].axhline(0, linestyle=":", color="k")
    ax[1].axhline(0.1, linestyle=":", color="k")
    ax[1].axhline(-0.1, linestyle=":", color="k")

    
    ax[2].axhline(0, linestyle=":", color="k")
    ax[2].axhline(1, linestyle=":", color="k")
    ax[2].axhline(-1, linestyle=":", color="k")
    
    ax[0].set_xscale("log")
    ax[0].set_title(data_dict["z"][iz])
    # plt.yscale("log")
    
    fig, ax = plt.subplots(3, sharex=True)
    x = data_dict["k1d_Mpc"][k1d_mask]
    ax[0].plot(x, x*list_sim_use[iz]["p1d_Mpc"][k1d_mask], "o:")
    ax[0].plot(x, x*model_p1d, "-")
    
    y = model_p1d/list_sim_use[iz]["p1d_Mpc"][k1d_mask]-1
    # y = (model_p1d-data_dict["p1d_Mpc"][k1d_mask])/fit.err_p1d
    ax[1].plot(x, y, "-")

    # norm = np.nanmax(list_sim_use[iz]["p1d_Mpc"][k1d_mask])
    # norm = 1
    # norm = (data_dict["z"][iz])**3*0.5
    y = (list_sim_use[iz]["p1d_Mpc"][k1d_mask]-model_p1d)/data_dict["std_p1d"][iz][k1d_mask]
    ax[2].plot(x, y, "-")

    
    ax[2].axhline(0, linestyle=":", color="k")
    ax[2].axhline(1, linestyle=":", color="k")
    ax[2].axhline(-1, linestyle=":", color="k")

    
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

