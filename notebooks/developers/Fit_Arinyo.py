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
# ### Start fit

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

def get_input_data(data, err_p3d, err_p1d):
    data_dict = {}
    data_dict["units"] = "N"
    data_dict["z"] = np.atleast_1d(data["z"])
    data_dict["k3d"] = data["k3d_Mpc"]
    data_dict["mu3d"] = data["mu3d"]
    data_dict["p3d"] = data["p3d_Mpc"] * data["k3d_Mpc"] ** 3 / 2 / np.pi**2
    std_p3d = get_std_kp3d(
        data["sim_label"], data["ind_rescaling"], data["ind_snap"], err_p3d, sm_pk=False
    )
    data_dict["std_p3d"] = std_p3d

    data_dict["k1d"] = data["k_Mpc"]
    data_dict["p1d"] = data["p1d_Mpc"] * data["k_Mpc"] / np.pi
    std_p1d = get_std_kp1d(data["sim_label"], data["ind_rescaling"], data["ind_snap"], err_p1d)
    data_dict["std_p1d"] = std_p1d

    # read cosmology
    # assert "LACE_REPO" in os.environ, "export LACE_REPO"
    # folder = os.environ["LACE_REPO"] + "/lace/emulator/sim_suites/post_768/"
    # genic_fname = (
    #     folder + "sim_pair_" + str(data["ind_sim"]) + "/sim_plus/paramfile.genic"
    # )
    # sim_cosmo_dict = read_genic.camb_from_genic(genic_fname)
    # cosmo = camb_cosmo.get_cosmology_from_dictionary(sim_cosmo_dict)

    # # get model
    # camb_results = camb_cosmo.get_camb_results(
    #     cosmo, zs=data_dict["z"], camb_kmax_Mpc=200
    # )
    # model = model_p3d_arinyo.ArinyoModel(cosmo, data_dict["z"][0], camb_results)
    # linp = (
    #     model.linP_Mpc(z=data_dict["z"][0], k_Mpc=data_dict["k3d"])
        
    # )
    linp = data['Plin'] * data["k3d_Mpc"] ** 3 / 2 / np.pi**2
    model = data['model']

    return data_dict, model, linp


# %% [markdown]
# same weight both, scale dependent for p3d, not for p1d
#
# when q2, q1+q2, q1-q2

# %%
err_p1d = np.load(path_program + "/data/p1d_4_fit.npz")
err_p3d = np.load(path_program + "/data/p3d_4_fit.npz")

# fit options
kmax_3d = 3
kmax_1d = 3
noise_3d = 0.01
noise_1d = 0.01
fit_type = "both"
use_q2 = True

list_sim_use = Archive3D.get_testing_data("mpg_central")

res_params = np.zeros((len(list_sim_use), 8, 2))
res_chi2 = np.zeros((len(list_sim_use), 2))

z_central = np.array([d["z"] for d in list_sim_use])

# %%
list_sim_use = []
sim_label = "mpg_0"
for isim in Archive3D.training_data:
    if(isim['sim_label'] == sim_label):
        list_sim_use.append(isim)

# %%
len(list_sim_use)

# %%
file = "/home/jchaves/Proyectos/projects/lya/data/forestflow/fits/fit_sim_label_mpg_10_kmax3d_3_noise3d_0.01_kmax1d_3_noise1d_0.01.npz"

# %%
data = np.load(file, allow_pickle=True)
data.files

# %%
data = np.load(file)
best_params = data["best_params"]
ind_snap = data["ind_snap"]
val_scaling = data["val_scaling"]

# %%
ii = 0
ind = (ind_snap == archive[ii]["ind_snap"]) & (
    val_scaling == archive[ii]["val_scaling"]
)

# %%
for ii in range(57):
    print(ii, archive[ii]["sim_label"])

# %%
archive = Archive3D.training_data

# %%

# %%
dat2["best_params"][0,:,1]

# %%
list_sim_use[0]["ind_snap"]

# %%
list_sim_use[0].keys()

# %%
# %%time

for isim, sim_use in enumerate(list_sim_use):
    for iq2, use_q2 in enumerate([False, True]):
        # if(isim == 10):
        #     pass
        # else: 
        #     continue
            
        print(isim, iq2)
        print()

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
            # parameters[par] = res_params[isim, ii, 0]

        # if use_q2 == False:
        #     parameters["q1"] -= 0.2
        #     parameters["beta"] -= 0.1
        #     parameters["kvav"] -= 0.2
        #     parameters["av"] -= 0.2
        # parameters["q2"] = 0

        # folder and name of output file
        # folder_out_file = "/data/desi/scratch/jchavesm/p3d_fits/"
        # out_file = get_flag_out(ind_sim, kmax_3d, noise_3d, kmax_1d, noise_1d)
        
        # get input data
        data_dict, model, linp = get_input_data(
            sim_use, err_p3d, err_p1d
        )
        
        # set fitting model
        fit = FitPk(
            data_dict,
            model,
            fit_type=fit_type,
            k3d_max=kmax_3d,
            k1d_max=kmax_1d,
            noise_3d=noise_3d,
            noise_1d=noise_1d,
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
        res_params[isim, :val.shape[0], iq2] = val
        res_chi2[isim, iq2] = chia
        
    



# %%
plt.plot(z_central, res_chi2[:,0], label="q1 and q2")
plt.plot(z_central, res_chi2[:,1], label="q1")
plt.ylabel("chi2")
plt.xlabel("z")
plt.legend()
plt.savefig("chi2_for_central.png")

# %%
best_fit_params.keys()

# %%

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# folder_fig = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"

ftsize = 20

# Create a 2x1 grid for plotting
fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
name_params = ['bias', 'beta', 'q1', 'kvav', 'av', 'bv', 'kp', "q2"]
# name_params = list(Arinyo_emu[0].keys())

name2label = {
    'bias':r"$-b_\delta$", 
    'bias_eta':r"$-b_\eta$", 
    'q1':r"$q_1$", 
    'q2':r"$q_2$",
    'kv':r"$k_\mathrm{v}$", 
    'av':r"$a_\mathrm{v}$", 
    'bv':r"$b_\mathrm{v}$", 
    'kp':r"$k_\mathrm{p}$", 
}

# Plot the original and emulator data in the upper panel
for i in range(len(name_params)):
    if(i < 4):
        ax1 = ax[0]
    else:
        ax1 = ax[1]
    col = "C"+str(i)
    # ari_emu = np.array([d[name_params[i]] for d in Arinyo_emu])
    # ari_emu_std = np.array([d[name_params[i]] for d in Arinyo_emu_std])
    # ari_cen = np.array([d[name_params[i]] for d in Arinyo_sim])
    ari_cen = res_params[:, i, 0]
    ari_emu = res_params[:, i, 1]

    # print(name_params[i])
    # print(np.mean(np.abs(ari_emu)/np.abs(ari_cen)-1))
    # print(np.std(np.abs(ari_emu)/np.abs(ari_cen)-1))
    # if i != 6:
    ax1.plot(
        z_central,
        np.abs(ari_cen),
        "o:",
        color=col,
        lw=2
        # label=name2label[name_params[i]],
    )
    ax1.plot(
        z_central,
        np.abs(ari_emu),
        color=col,
        ls="-",
    )

    # ax1.fill_between(
    #     z_central, 
    #     np.abs(ari_emu)-0.5*ari_emu_std, 
    #     np.abs(ari_emu)+0.5*ari_emu_std,
    #     color=col,
    #     alpha=0.2
    # )
    # ax2.plot(z_central, np.abs(ari_cen)
    # / np.abs(ari_emu)
    # - 1, color=colors[i], ls="-")

for ii in range(2):
    ax[ii].set_ylabel("Parameter", fontsize=ftsize)
    ax[ii].set_yscale("log")
    ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)

ax[0].set_ylim(0.02, 2)
ax[1].set_ylim(0.02, 25)
    
ax[-1].set_xlabel("$z$", fontsize=ftsize)


hand = []
for i in range(4):
    col = "C"+str(i)
    hand.append(mpatches.Patch(color=col, label=name_params[i]))
legend1 = ax[0].legend(fontsize=ftsize-2, loc="lower left", handles=hand, ncols=4)

line1 = Line2D([0], [0], label='q1 and q2', color='k', ls=":", marker="o")
line2 = Line2D([0], [0], label='q1', color='k', ls="-")
hand = [line1, line2]
ax[0].legend(fontsize=ftsize-2, loc="upper left", handles=hand, ncols=2)
ax[0].add_artist(legend1)

hand = []
for i in range(4, len(name_params)):
    col = "C"+str(i)
    hand.append(mpatches.Patch(color=col, label=name_params[i]))
legend1 = ax[1].legend(fontsize=ftsize-2, loc="lower left", handles=hand, ncols=4)

# plt.gca().add_artist(legend1)
# Adjust layout
plt.tight_layout()

# plt.savefig("arinyo_z.png")
# plt.savefig("arinyo_z.pdf")

# Show the plot
# plt.show()

# %%
plot = True

for isim in range(11):
    name_params = ['bias', 'beta', 'q1', 'kvav', 'av', 'bv', 'kp', "q2"]
    print(z_central[isim])
    # if(isim == 10):
    #     pass
    # else:
    #     continue
    
    p1 = {}
    p2 = {}
    for ii, par in enumerate(name_params):
        p1[par] = res_params[isim, ii, 1]
        if(par != "q2"):
            p2[par] = res_params[isim, ii, 0]
    # print(p1)
    # print(p2)
    
    p3d_Mpc = list_sim_use[isim]["p3d_Mpc"].copy()
    p1d_Mpc = list_sim_use[isim]["p1d_Mpc"].copy()
    
    best_p3d = list_sim_use[isim]["model"].P3D_Mpc(z_central[isim], data_dict["k3d"], data_dict["mu3d"], p1)
    pred_p3d = list_sim_use[isim]["model"].P3D_Mpc(z_central[isim], data_dict["k3d"], data_dict["mu3d"], p2)
    
    best_p1d = list_sim_use[isim]["model"].P1D_Mpc(z_central[isim], data_dict["k1d"], parameters=p1)
    pred_p1d = list_sim_use[isim]["model"].P1D_Mpc(z_central[isim], data_dict["k1d"], parameters=p2)
    if plot:
        fig, ax = plt.subplots(2, sharex=True, figsize=(10, 10))
        jj = 0
        mask = k3d_Mpc[:,0] < kmax_3d
        ax[0].plot(k3d_Mpc[mask, 0], p3d_Mpc[mask, 0][:]*0, 'k:')
        for ii in range(0, p3d_Mpc.shape[1], 3):
            col = 'C'+str(jj)
            lab = r'$<\mu>=$'+str(np.round(np.nanmean(mu3d[:,ii]), 2))
            # ax[0].plot(k3d_Mpc[mask, ii], p3d_Mpc[mask, ii]/Plin[mask, ii], col+'-', label=lab)
            # ax[0].plot(k3d_Mpc[mask, ii], best_p3d[mask, ii]/Plin[mask, ii], col+'--')
            # ax[0].plot(k3d_Mpc[mask, ii], pred_p3d[mask, ii]/Plin[mask, ii], col+':')
            # ax[0].plot(k3d_Mpc[mask, ii], p3d_Mpc[mask, ii]/Plin[mask, ii], col+'-', label=lab)
            
            ax[0].plot(k3d_Mpc[mask, ii], best_p3d[mask, ii]/p3d_Mpc[mask, ii]-1, col+'-', label=lab)
            ax[0].plot(k3d_Mpc[mask, ii], pred_p3d[mask, ii]/p3d_Mpc[mask, ii]-1, col+'--')
            jj += 1
        ax[0].set_xscale('log')
        ax[1].set_xlabel(r'$k$ [Mpc]')
        ax[0].set_ylabel(r'Residual')
        ax[1].set_ylabel(r'Residual')
        ax[0].legend()
    
        mask = (k1d_Mpc < kmax_1d) & (k1d_Mpc > 0)
        ax[1].plot(k1d_Mpc[mask], p1d_Mpc[mask][:]*0, 'k:')
        ax[1].plot(k1d_Mpc[mask], best_p1d[mask]/p1d_Mpc[mask]-1, '-', label="q1 and q2")
        ax[1].plot(k1d_Mpc[mask], pred_p1d[mask]/p1d_Mpc[mask]-1, '--', label="q1")
        ax[1].legend()
        plt.tight_layout()
        ax[1].set_ylim([-0.03, 0.03])
        ax[0].set_ylim([-0.2, 0.2])
        plt.savefig("fit_z"+str(isim)+"_new2.png")
        # plt.savefig("fit_z"+str(isim)+"_k3d_4_k1d_4.png")

# %%

# %%

# %%

# %%
# %%time
## fit data ##
# get initial solution for sampler
# we extract 5 samples from the priors
parameter_names = list(parameters.keys())
nsam = 5
seed = 0
nparams = len(parameter_names)
design = lhs(
    nparams,
    samples=nsam,
    criterion="c",
    random_state=seed,
)

for ii in range(nparams):
    buse = priors[parameter_names[ii]]
    design[:, ii] = (buse[1] - buse[0]) * design[:, ii] + buse[0]

chia = 1e10
for ii in range(design.shape[0]):
    pp = {}
    for jj in range(nparams):
        pp[parameter_names[jj]] = design[ii, jj]
    results, _best_fit_params = fit.maximize_likelihood(pp)
    chi2 = fit.get_chi2(_best_fit_params)
    if chi2 < chia:
        chia = chi2
        best_fit_params = _best_fit_params.copy()

# and values that we know work well for some samples
results, _best_fit_params = fit.maximize_likelihood(parameters)
chi2 = fit.get_chi2(_best_fit_params)
if chi2 < chia:
    chia = chi2
    best_fit_params = _best_fit_params.copy()
# the output is chia and best_fit_params
print("Initial chi2", chia)
print("and best_params", best_fit_params)

# %%

chia = 10
best_fit_params = ptarget.copy()
best_fit_params["q2"] = 0

# %%
# %%time
results, _best_fit_params = fit.maximize_likelihood(best_fit_params)
chi2 = fit.get_chi2(_best_fit_params)
if chi2 < chia:
    chia = chi2
    best_fit_params = _best_fit_params.copy()
# the output is chia and best_fit_params
print("Initial chi2", chia)
print("and best_params", best_fit_params)

# %%

# %%
best_fit_params

# %%

ptarget

# %%
results, _best_fit_params = fit.maximize_likelihood(ptarget)
# fit.get_chi2(ptarget)

# %%
_best_fit_params

# %%

# %%
