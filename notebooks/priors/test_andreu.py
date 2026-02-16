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

# cosmology functions from LaCE
import camb
from lace.cosmo import camb_cosmo, fit_linP

# cup1d functions to reconstruct models from chain
from cup1d.likelihood.input_pipeline import Args
from cup1d.likelihood.pipeline import Pipeline

# ForestFlow emulator
from forestflow.model_p3d_arinyo import ArinyoModel
from forestflow.archive import GadgetArchive3D
from forestflow.P3D_cINN import P3DEmulator


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
#

# %%
# Setup cup1d pipeline to read and interpret P1D chains
def get_cup1d_pipeline():
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

    return pip


# %%
pip = get_cup1d_pipeline()


# %%
def get_p1d_chain():
    folder = '/global/cfs/cdirs/desi/users/jjchaves/p1d/'
    chain = np.array(np.load(folder + "chain.npy"))
    chain = chain.reshape(-1, 53)
    return chain


# %%
chain = get_p1d_chain()


# %%
def get_As_ns_from_sample(pip, sample):
    like_params = pip.fitter.like.parameters_from_sampling_point(sample)
    As = like_params[0].value_from_cube(sample[0])
    ns = like_params[1].value_from_cube(sample[1])
    return As, ns


# %%
def get_cosmo_from_sample(pip, sample):
    # get As, ns from sample
    As, ns = get_As_ns_from_sample(pip, sample)
    # setup CAMB object
    cosmo_params = {
        'H0': 67.66,
        'mnu': 0,
        'omch2': 0.119,
        'ombh2': 0.0224,
        'As': As, 
        'ns': ns, 
    }
    cosmo = camb_cosmo.get_cosmology_from_dictionary(cosmo_params)
    
    return cosmo


# %%
def get_growth_at_z(cosmo, z=2.33): 

    # will ask for quantities at z=2.33
    cosmo.set_matter_power(redshifts=[z], nonlinear=False) #, silent=True)
    # compute linear power quantities
    camb_results = camb.get_results(cosmo)
    sig_8 = np.array(camb_results.get_sigma8())
    f_sig_8 = camb_results.get_fsigma8()

    return sig_8[0], f_sig_8[0]


# %%
sample = chain[0]
As, ns = get_As_ns_from_sample(pip, sample)
print(As, ns)
cosmo = get_cosmo_from_sample(pip, sample)
sig_8, f_sig_8 = get_growth_at_z(cosmo, z=2.33)
print(sig_8, f_sig_8)


# %%
def get_params_from_sample(pip, sample, z=2.33):

    like_params = pip.fitter.like.parameters_from_sampling_point(sample)
    
    # get cosmo first
    cosmo = get_cosmo_from_sample(pip, sample) 
    ### HOW DO I GET THE STAR PARAMETERS FOR THIS COSMO?

    sig_8, f_sig_8 = get_growth_at_z(cosmo, z)
    params = {'sigma_8': sig_8, 'f_sigma_8': f_sig_8}    
    dkms_dMpc = camb_cosmo.dkms_dMpc(cosmo, z=z)
    #print(z, dkms_dMpc)
    
    # compute linear power parameters (in Mpc units)
    linP = fit_linP.get_linP_Mpc_zs(cosmo, zs=[z], kp_Mpc=0.7)

    # collect emu params
    params["Delta2_p"] = linP[0]["Delta2_p"]
    params["n_p"] = linP[0]["n_p"]

    # mean flux model
    mf_model = pip.fitter.like.theory.model_igm.models["F_model"]
    params["mF"] = mf_model.get_mean_flux(z, like_params=like_params)

    # thermal model
    T_model = pip.fitter.like.theory.model_igm.models["T_model"]
    params["gamma"] = T_model.get_gamma(z, like_params=like_params)
    sigT_kms = T_model.get_sigT_kms(z, like_params=like_params)
    params["sigT_Mpc"] = sigT_kms / dkms_dMpc

    # pressure model
    P_model = pip.fitter.like.theory.model_igm.models["P_model"]
    kF_kms = P_model.get_kF_kms(z, like_params=like_params)
    params["kF_Mpc"] = kF_kms * dkms_dMpc
    
    return params


# %%
params = get_params_from_sample(pip, sample, z=2.33)
print(params)


# %%
# For each point in the P1D chain, compute emulator parameters at z=2.33
def get_params_from_chain(pip, chain, z=2.33, nn=100):   
    ind = np.random.permutation(np.arange(2739200))[:nn]
    new_chain = [] 

    for ii in range(nn):
        sample = chain[ind[ii], :]
        params = get_params_from_sample(pip, sample, z)
        new_chain.append(params)

    return new_chain


# %%
new_chain = get_params_from_chain(pip, chain, z=2.33, nn=100)

# %%
sig8 = [sample["sigma_8"] for sample in new_chain]
print(np.mean(sig8))
print(np.sqrt(np.var(sig8)))

# %%
f_sig8 = [sample["f_sigma_8"] for sample in new_chain]
print(np.mean(f_sig8))
print(np.sqrt(np.var(f_sig8)))

# %%
DL2 = [sample["Delta2_p"] for sample in new_chain]
print(np.mean(DL2))
print(np.sqrt(np.var(DL2)))

# %%
plt.scatter(sig8, DL2)

# %% [markdown]
# ## LOAD P3D ARCHIVE (needed for forestflow)

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
# ## Load ForestFlow

# %%
training_type = "Arinyo_min"
model_path = path_program + "/data/emulator_models/mpg_hypercube.pt"

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
for sample in new_chain:
    info = emulator.evaluate(emu_params=sample, Nrealizations=10000)
    #print(sample['mF'], info['coeffs_Arinyo']['bias'])
    
    ### ForestFlow MUST be deterministic! At least allow user to use always the same random seed 

    for key, value in info['coeffs_Arinyo'].items():
        sample[key] = value

# %%
new_chain[1]

# %%
bias = np.array([sample["bias"] for sample in new_chain])
print(np.mean(bias))
print(np.sqrt(np.var(bias)))

# %%
beta = np.array([sample["beta"] for sample in new_chain])
print(np.mean(beta))
print(np.sqrt(np.var(beta)))

# %%
b_eta = beta * bias * sig_8 / f_sig_8
print(np.mean(b_eta))
print(np.sqrt(np.var(b_eta)))


# %%
b_sig_8 = bias * sig_8
print(np.mean(b_sig_8))
print(np.sqrt(np.var(b_sig_8)))

# %%
b_eta_f_sig_8 = b_eta * f_sig_8
print(np.mean(b_eta_f_sig_8))
print(np.sqrt(np.var(b_eta_f_sig_8)))

# %%
