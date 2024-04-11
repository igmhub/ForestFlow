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
from forestflow.model_p3d_arinyo import get_linP_interp
from forestflow.model_p3d_arinyo import ArinyoModel

from forestflow.archive import GadgetArchive3D
from forestflow.P3D_cINN import P3DEmulator

def ls_level(folder, nlevels):
    for ii in range(nlevels):
        folder = os.path.dirname(folder)
    folder += "/"
    return folder


path_program = ls_level(os.getcwd(), 1)
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
mask = k1d_Mpc < 4
plt.plot(k1d_Mpc[mask], p1d_Mpc[mask]/model_p1d[mask]-1, '-', label='Sim/Model-1')
plt.plot(k1d_Mpc[mask], k1d_Mpc[mask]*0, 'k--')
plt.xscale('log')
plt.xlabel(r'$k$ [Mpc]')
plt.ylabel(r'$P_{\rm 1D}$')
plt.legend()


# %%

# %%
def get_default_params():
    parameters_q = {
        "bias": -0.12,
        "beta": 1.4,
        "d1_q1": 0.4,
        "d1_kvav": 0.6,
        "d1_av": 0.3,
        "d1_bv": 1.5,
        "d1_kp": 18.0,
    }
    parameters_q2 = {
        "bias": -0.12,
        "beta": 1.4,
        "d1_q1": 0.4,
        "d1_kvav": 0.6,
        "d1_av": 0.3,
        "d1_bv": 1.5,
        "d1_kp": 18.0,
        "d1_q2": 0.2,
    }

    priors_q = {
        "bias": [-1, 0.5],
        "beta": [0, 5.0],
        "d1_q1": [0, 5],
        "d1_kvav": [0.1, 5.0],
        "d1_av": [0, 2],
        "d1_bv": [0, 5],
        "d1_kp": [1, 50],
    }
    priors_q2 = {
        "bias": [-1, 0.5],
        "beta": [0, 5.0],
        "d1_q1": [0, 5],
        "d1_kvav": [0.1, 5.0],
        "d1_av": [0, 2],
        "d1_bv": [0, 5],
        "d1_kp": [1, 50],
        "d1_q2": [0, 5],
    }

    return parameters_q, priors_q, parameters_q2, priors_q2


# %%
def get_std_kp1d(ind_sim, ind_tau, ind_z, err_p1d):
    _sim = np.argwhere(err_p1d["u_ind_sim"] == ind_sim[4:])[0, 0]
    _tau = np.argwhere(err_p1d["u_ind_tau"] == ind_tau)[0, 0]
    _z = np.argwhere(err_p1d["u_ind_z"] == ind_z)[0, 0]
    av_pk = err_p1d["p1d_sim_tau_z"][_sim, _tau, _z] * err_p1d["k"] / np.pi
    std_kpk = err_p1d["sm_rel_err"] * av_pk
    return std_kpk


def get_std_kp3d(ind_sim, ind_tau, ind_z, err_p3d, sm_pk=False):
    print(ind_sim[4:])
    print(err_p3d["u_ind_sim"])
    _sim = np.argwhere(err_p3d["u_ind_sim"] == ind_sim[4:])[0, 0]
    _tau = np.argwhere(err_p3d["u_ind_tau"] == ind_tau)[0, 0]
    _z = np.argwhere(err_p3d["u_ind_z"] == ind_z)[0, 0]
    if sm_pk:
        pk = err_p3d["sm_p3d_sim_tau_z"][_sim, _tau, _z]
    else:
        pk = err_p3d["p3d_sim_tau_z"][_sim, _tau, _z]

    av_pk = pk * err_p3d["k"] ** 3 / 2 / np.pi**2
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
    linp = data['Plin'] * data_dict["k3d_Mpc"] ** 3 / 2 / np.pi**2
    model = data['model']

    return data_dict, model, linp


# %%
Archive3D.training_data[ind_book].keys()

# %%
err_p1d['u_ind_sim']

# %%
Archive3D.training_data[ind_book]["sim_label"][4:]

# %%
err_p1d = np.load(path_program + "/data/p1d_4_fit.npz")
err_p3d = np.load(path_program + "/data/p3d_4_fit.npz")

# fit options
kmax_3d = 5
noise_3d = 0.075
kmax_1d = 5
noise_1d = 0.01
fit_type = "both"
use_q2 = True

# get initial parameters for fit
_ = get_default_params()
parameters_q, priors_q, parameters_q2, priors_q2 = _
if use_q2:
    parameters = parameters_q2
    priors = priors_q2
else:
    parameters = parameters_q
    priors = priors_q

# folder and name of output file
# folder_out_file = "/data/desi/scratch/jchavesm/p3d_fits/"
# out_file = get_flag_out(ind_sim, kmax_3d, noise_3d, kmax_1d, noise_1d)

# get input data
data_dict, model, linp = get_input_data(
    Archive3D.training_data[ind_book], err_p3d, err_p1d
)

# set fitting model
fit = fit_p3d.FitPk(
    data_dict,
    model,
    fit_type=fit_type,
    k3d_max=kmax_3d,
    k1d_max=kmax_1d,
    noise_3d=noise_3d,
    noise_1d=noise_1d,
    priors=priors,
)



# %%

# %%
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
