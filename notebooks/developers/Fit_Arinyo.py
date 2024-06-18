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

# %% [markdown]
# Estimate the contribution to the chi2 from each redshift! The redshift weighting is likely wrong

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

# %% [markdown]
# ### Compute average central and seed

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

file = path_program + "/data/best_arinyo/minimizer/fit_sim_label_combo_kmax3d_5_kmax1d_4.npz"
data = np.load(file, allow_pickle=True)
# best_params = paramz_to_paramind(zlist, data["best_params"].item())

for ii in range(len(list_merge)):
    # list_merge[ii]["Arinyo_minz"] = params_numpy2dict_minimizerz(best_params[ii])
    list_merge[ii]["Arinyo_minz"] = data["best_params"][ii]

# %% [markdown]
# ### Get normalization of P3D and P1D for fit

# %%
data_dict, model = get_input_data(cen, 3)
nmodes = 1/data_dict["std_p3d"][:,0]

# %%
rat_p1d_z = np.zeros((len(list_central), 2))
rat_p3d_z = np.zeros((len(list_central), 2))

rat_p3d_mu = np.zeros((len(list_central), 3, 16))
k_max = (cen["k_Mpc"] <=3) & (cen["k_Mpc"] > 0)
k_max_p1d = (cen["k_Mpc"] <=1) & (cen["k_Mpc"] > 0)
k_max_p3d = cen["k3d_Mpc"][:,0] <= 3
k_mu_p3d = cen["k3d_Mpc"][:, 0] <= 1
for ii in range(len(list_central)):
    # print(list_central[ii]["Arinyo_minz"])
    rat_p1d_z[ii, 0] = 1 + list_central[ii]["z"]
    rat_p1d_z[ii, 1] = np.median(list_central[ii]["p1d_Mpc"][k_max_p1d]/cen["p1d_Mpc"][k_max_p1d])
    # plt.plot(cen["k_Mpc"][k_max], list_central[ii]["p1d_Mpc"][k_max]/cen["p1d_Mpc"][k_max])
    norm = 1/np.exp((1 + list_central[ii]["z"]) * p1d_res[0] + p1d_res[1])*(1+cen["k_Mpc"][k_max]/2)**2 * 2000
    # plt.plot(cen["k_Mpc"][k_max], list_central[ii]["p1d_Mpc"][k_max]*norm)
    
    rat_p3d_z[ii, 0] = 1 + list_central[ii]["z"]
    rat_p3d_z[ii, 1] = np.median(list_central[ii]["p3d_Mpc"][k_max_p3d, 0]/cen["p3d_Mpc"][k_max_p3d, 0])
    # plt.plot(cen["k3d_Mpc"][k_max_p3d, 0], list_central[ii]["p3d_Mpc"][k_max_p3d, 0]/cen["p3d_Mpc"][k_max_p3d, 0])
    # norm = 1/np.exp((1 + list_central[ii]["z"]) * p3d_res[0] + p3d_res[1]) * nmodes[k_max_p3d]
    # norm *= 1/(1+cen["k3d_Mpc"][k_max_p3d, 0]/3)**2
    # plt.plot(cen["k3d_Mpc"][k_max_p3d, 0], list_central[ii]["p3d_Mpc"][k_max_p3d, 0]*norm)

    for jj in range(16):
        rat_p3d_mu[ii, 0, :] = 1 + list_central[ii]["z"]
        rat_p3d_mu[ii, 1, jj] = np.nanmedian(list_central[ii]["mu3d"][k_mu_p3d, jj])
        rat_p3d_mu[ii, 2, jj] = np.nanmedian(list_central[ii]["p3d_Mpc"][k_mu_p3d, jj]/list_central[ii]["p3d_Mpc"][k_mu_p3d, 0])
    # plt.plot(rat_p3d_mu[ii, 1, :], np.log(rat_p3d_mu[ii, 2, :]))

x = rat_p1d_z[:, 0]
y = np.log(rat_p1d_z[:, 1])
p1d_res = np.polyfit(x, y, 1)
# plt.plot(x, y)
# plt.plot(x, x*p1d_res[0]+p1d_res[1])

x = rat_p3d_z[:, 0]
y = np.log(rat_p3d_z[:, 1])
p3d_res = np.polyfit(x, y, 1)
# plt.plot(x, y)
# plt.plot(x, x*p3d_res[0]+p3d_res[1])


x = rat_p3d_mu[6, 1, :]
y = np.log(rat_p3d_mu[2, 2, :])
mu_res = np.polyfit(x, y, 2)
plt.plot(x, y)
plt.plot(x, x**2*mu_res[0]+x*mu_res[1]+mu_res[2])


print(p1d_res, p3d_res, mu_res)


# %% [markdown]
# ### Priors and seeds

# %%
def get_default_params():

    names = ["bias", "beta", "q1", "kvav", "av", "bv", "kp", "q2"]
    # names = ["bias", "beta", "q1", "kvav", "av", "bv", "kp"]
    
    parameters_q = {
        "bias": 0.12,
        "beta": 1.4,
        "q1": 0.4,
        "kvav": 0.6,
        "av": 0.3,
        "bv": 1.5,
        "kp": 18.0,
        "q2": 1e-2,
    }
    
    priors = {
        "bias": [1e-3, 3],
        "beta": [5e-3, 7],
        "q1": [1e-2, 8],
        "kvav": [1e-3, 5],
        "av": [1e-3, 2],
        "bv": [1e-1, 5],
        "kp": [3, 30],
        "q2": [1e-4, 8],
    }

    return parameters_q, priors


def get_input_data(z_use, list_data, kmax_fit, version=1):

    if(version == 1):
        p1d_res = [0.9887013,  -2.94078465]
        p3d_res = [1.11635295, -3.32955821]
        mu_res = [ 1.1985935,  -0.11867367,  0.00485981]
        alpha1d = 500
        alpha3d = 0.25
        k0_p1d = 2
        k0_p3d = 3
        
    data_dict = {}

    z_grid = np.array([d["z"] for d in list_data])
    mask_z = np.argmin(np.abs(z_use - z_grid))
    data_dict["z"] = z_grid[mask_z]
    data_dict["kmu_modes"] = get_p3d_modes(kmax_fit)
    nz = len(list_data)
    print(mask_z)

    data_dict["k3d_Mpc"] = list_data[0]["k3d_Mpc"]
    data_dict["mu3d"] = list_data[0]["mu3d"]
    data_dict["k1d_Mpc"] = list_data[0]["k_Mpc"]
    
    n_modes = np.zeros_like(data_dict["k3d_Mpc"])
    for ii in range(data_dict["k3d_Mpc"].shape[0]):
        for jj in range(data_dict["k3d_Mpc"].shape[1]):
            key = f"{ii}_{jj}_k"
            if key in data_dict["kmu_modes"]:
                n_modes[ii, jj] = data_dict["kmu_modes"][key].shape[0]
    
    data_dict["p3d_Mpc"] = np.zeros((data_dict["k3d_Mpc"].shape[0], data_dict["k3d_Mpc"].shape[1]))
    data_dict["std_p3d"] = np.zeros_like(data_dict["p3d_Mpc"])
    data_dict["p1d_Mpc"] = np.zeros((data_dict["k1d_Mpc"].shape[0]))
    data_dict["std_p1d"] = np.zeros_like(data_dict["p1d_Mpc"])
    
    for ii in range(nz):
        if(ii == mask_z):        
            data_dict["p3d_Mpc"] = list_data[ii]["p3d_Mpc"]
            data_dict["p1d_Mpc"] = list_data[ii]["p1d_Mpc"]
            model = list_data[ii]["model"]
            if(version == 0):
                normz = data_dict["z"][ii]**3
                data_dict["std_p3d"][ii] = normz / n_modes * data_dict["k3d_Mpc"]**0.25
                data_dict["std_p1d"][ii] = normz * np.pi / data_dict["k1d_Mpc"] / np.mean(n_modes.sum(axis=0))
            elif(version == 1):
                norm3d = alpha3d * n_modes / np.exp((1 + data_dict["z"]) * p3d_res[0] + p3d_res[1])
                norm3d /= np.exp(data_dict["mu3d"]**2*mu_res[0] + data_dict["mu3d"]*mu_res[1] + mu_res[2])
                data_dict["std_p3d"] = 1/norm3d
                norm1d = (1 + data_dict["k1d_Mpc"]/k0_p1d)**2 * alpha1d/np.exp((1 + data_dict["z"]) * p1d_res[0] + p1d_res[1])
                data_dict["std_p1d"] = 1/norm1d

    return data_dict, model


# %% [markdown]
# ### Fit at fixed z

# %%
kmax_3d = 5
kmax_1d = 4
fit_type = "both"
# sim_label = "mpg_central"
sim_label = "mpg_seed"
# sim_label = "combo"

if(sim_label == "combo"):
    list_sim_use = list_merge
else:
    list_sim_use = Archive3D.get_testing_data(sim_label, kmax_3d=3, kmax_1d=3)

# %%
z_grid = np.array([d["z"] for d in list_sim_use])
res_params = []
res_chi2 = []

for iz, z_use in enumerate(z_grid):
    # if(z_use == 4.5) | (z_use == 2.0):
    #     pass
    # else:
    #     continue
    print(z_use)
    print()

    data_dict, model = get_input_data(z_use, list_sim_use, kmax_3d)
    parameters, priors = get_default_params()
    parameters = list_merge[iz]["Arinyo_minz"].copy()
    
    params_minimizer = np.array(list(parameters.values()))
    names = np.array(list(parameters.keys())).reshape(-1)

    # set fitting model
    fit = FitPk(
        data_dict,
        model,
        names=names,
        priors=priors,
        fit_type=fit_type,
        k3d_max=kmax_3d,
        k1d_max=kmax_1d,
        verbose=True,
        maxiter = 400,
    )
    
    chia = fit.get_chi2(params_minimizer)
    # chia = fit.get_chi2(np.array(list(best_fit_params.values())))
    print("Initial chi2", chia)
    results, best_fit_params = fit.maximize_likelihood(params_minimizer)
    params_minimizer = np.array(list(best_fit_params.values()))
    chi2 = fit.get_chi2(params_minimizer)
    print("Final chi2", chi2)
    print("and best_params", best_fit_params)

    res_params.append(best_fit_params)
    res_chi2.append(chi2)


# %%
def get_flag_out(ind_sim, kmax_3d, kmax_1d):
    flag = (
        "fit_sim_label_"
        + str(ind_sim)
        # + "_val_scaling_"
        # + str(np.round(val_scaling, 2))
        + "_kmax3d_"
        + str(kmax_3d)
        + "_kmax1d_"
        + str(kmax_1d)
    )
    return flag

val_scaling = 1

folder_save = forestflow.__path__[0][:-10] + "/data/best_arinyo/minimizer/"
out_file = get_flag_out(sim_label, kmax_3d, kmax_1d)
# out_file = "combo_arinyo"
np.savez(
    folder_save + out_file,
    chi2=res_chi2,
    best_params=res_params,
    z=z_grid,
    val_scaling=val_scaling,
    # readme="sim, param, w/o w/ q2",
)

# %%
folder_save

# %%
for iz, z_use in enumerate(z_grid):
    # if(z_use == 2.0):
    #     pass
    # else:
    #     continue

    data_dict, model = get_input_data(z_use, list_sim_use,  kmax_3d)
    best_fit_params = res_params[iz]

    n_mubins = 4
    k3d_mask = data_dict["k3d_Mpc"][:, 0] <= kmax_3d
    k1d_mask = (data_dict["k1d_Mpc"] <= kmax_1d) & (
        data_dict["k1d_Mpc"] > 0
    )
    
    nk = np.sum(k3d_mask)
    model_p3d, plin = p3d_allkmu(
        model,
        data_dict["z"],
        best_fit_params,
        data_dict["kmu_modes"],
        nk=nk,
        nmu=16,
        compute_plin=True,
    )
    
    _ = p3d_rebin_mu(data_dict["k3d_Mpc"][:nk], 
                     data_dict["mu3d"][:nk], 
                     data_dict["p3d_Mpc"][:nk], 
                     data_dict["kmu_modes"], 
                     n_mubins=n_mubins)
    knew, munew, rebin_p3d, mu_bins = _
    
    _ = p3d_rebin_mu(data_dict["k3d_Mpc"][:nk], 
                     data_dict["mu3d"][:nk], 
                     model_p3d[:nk], 
                     data_dict["kmu_modes"], 
                     n_mubins=n_mubins)
    knew, munew, rebin_model_p3d, mu_bins = _
    
    
    _ = p3d_rebin_mu(data_dict["k3d_Mpc"][:nk], 
                     data_dict["mu3d"][:nk], 
                     plin[:nk], 
                     data_dict["kmu_modes"], 
                     n_mubins=n_mubins)
    knew, munew, rebin_plin, mu_bins = _
    
    model_p1d = model.P1D_Mpc(
        data_dict["z"], 
        data_dict["k1d_Mpc"],
        parameters=best_fit_params
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
        y = (model_p3d[:nk, ii*5]/data_dict["p3d_Mpc"][:nk, ii*5])-1
        ax[1].plot(x[_], y[_], col+"--")
    
        y = (model_p3d[:nk, ii*5] - data_dict["p3d_Mpc"][:nk, ii*5])/data_dict["std_p3d"][:nk, ii*5]
        ax[2].plot(x[_], y[_], col+"-")
    
        
        jj += 1
    ax[1].axhline(0, linestyle=":", color="k")
    ax[1].axhline(0.1, linestyle=":", color="k")
    ax[1].axhline(-0.1, linestyle=":", color="k")
    
    
    ax[2].axhline(0, linestyle=":", color="k")
    ax[2].axhline(1, linestyle=":", color="k")
    ax[2].axhline(-1, linestyle=":", color="k")
    
    ax[0].set_xscale("log")
    ax[0].set_title(data_dict["z"])
    # plt.yscale("log")
    
    fig, ax = plt.subplots(3, sharex=True)
    x = data_dict["k1d_Mpc"][k1d_mask]
    ax[0].plot(x, x*data_dict["p1d_Mpc"][k1d_mask], "o:")
    ax[0].plot(x, x*model_p1d, "-")
    
    y = model_p1d/data_dict["p1d_Mpc"][k1d_mask]-1
    # y = (model_p1d-data_dict["p1d_Mpc"][k1d_mask])/fit.err_p1d
    ax[1].plot(x, y, "-")
    
    # norm = np.nanmax(list_sim_use[iz]["p1d_Mpc"][k1d_mask])
    # norm = 1
    # norm = (data_dict["z"][iz])**3*0.5
    y = (data_dict["p1d_Mpc"][k1d_mask]-model_p1d)/data_dict["std_p1d"][k1d_mask]
    ax[2].plot(x, y, "-")
    
    
    ax[2].axhline(0, linestyle=":", color="k")
    ax[2].axhline(1, linestyle=":", color="k")
    ax[2].axhline(-1, linestyle=":", color="k")
    
    ax[1].axhline(0, linestyle=":", color="k")
    ax[1].axhline(0.01, linestyle=":", color="k")
    ax[1].axhline(-0.01, linestyle=":", color="k")

    ax[0].set_xscale("log")

# %%
order = np.array([2, 2, 1, 1, 2, 1, 1, 1])
z = np.array(z_grid)

for jj, par_name in enumerate(names):
    if(par_name in names[6:8]):
        pass
    else:
        continue
    col = "C"+str(jj)
    res = np.zeros(len(z))
    for iz, z_use in enumerate(z_grid):
        res[iz] = res_params[iz][par_name]

    resf = np.polyfit(z, np.log10(res), deg=order[jj])
    p = 10 ** np.poly1d(resf)(z)

    norm = 1
    if par_name == "kp":
        norm = 0.1

    plt.plot(z, res*norm, col, label=par_name)
    plt.plot(z, p*norm, col+"--")
    # plt.plot(z, res/p, col, label=par_name)
plt.legend()
plt.yscale("log")

# %%
