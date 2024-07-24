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
# # Sensitivity plots for ForestFlow

# %%
# %load_ext autoreload
# %autoreload 2

from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


import os, sys
import numpy as np
from matplotlib import pyplot as plt

# our modules
from lace.archive import gadget_archive, nyx_archive
from lace.emulator.emulator_manager import set_emulator
from lace.emulator.nn_emulator import NNEmulator
from lace.emulator.gp_emulator import GPEmulator
from lace.utils import poly_p1d
import lace

from cup1d.p1ds.data_Chabanier2019 import P1D_Chabanier2019

import forestflow
from forestflow.archive import GadgetArchive3D
from forestflow.P3D_cINN import P3DEmulator
import copy


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
# ## Load data and emulator

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
# ### Power spectra

# %%
k_Mpc = np.zeros((1000, 2))
k_Mpc[:, 0] = np.geomspace(0.03, 6, 1000)
k_Mpc[:, 1] = np.geomspace(0.03, 6, 1000)
mu = np.zeros((1000, 2))
mu[:,0] = 0
mu[:,1] = 1

kpar_Mpc = np.geomspace(0.03, 5, 1000)

# %%
# target redshift
z_test = 3

# target cosmology
cosmo = {
    'H0': 67.0,
    'omch2': 0.12,
    'ombh2': 0.022,
    'mnu': 0.0,
    'omk': 0,
    'As': 2.006055e-09,
    'ns': 0.967565,
    'nrun': 0.0,
    'w': -1.0
}

# random (approx central)
input_params = {
    'Delta2_p': 0., # not used if you provide cosmology
    'n_p': 0., # not used if you provide cosmology
    'mF': 0.66,
    'sigT_Mpc': 0.13,
    'gamma': 1.5,
    'kF_Mpc': 10.5
}

var_input = {
    "As": cosmo["As"] + 0.05 * cosmo["As"],
    "omch2": cosmo["omch2"] + 0.05 * cosmo["omch2"],
    "mF": input_params["mF"] + 0.01 * input_params["mF"],
    "sigT_Mpc": input_params["sigT_Mpc"] + 0.05 * input_params["sigT_Mpc"], 
}

info_power = {
    "cosmo": cosmo,
    "k3d_Mpc": k_Mpc,
    "mu": mu,
    "k1d_Mpc": kpar_Mpc,
    "return_p3d": True,
    "return_p1d": True,
    "z": z_test,
}

out = emulator.evaluate(
    emu_params=input_params,
    info_power=info_power,
    Nrealizations=1000
)

orig_p1d = out['p1d']
orig_p3d = out['p3d']


# %%
var_p1d = np.zeros((4, orig_p1d.shape[0]))
var_p3d = np.zeros((4, orig_p3d.shape[0], orig_p3d.shape[1]))

# %%
for ii in range(4):
    cosmo = {
        'H0': 67.0,
        'omch2': 0.12,
        'ombh2': 0.022,
        'mnu': 0.0,
        'omk': 0,
        'As': 2.006055e-09,
        'ns': 0.967565,
        'nrun': 0.0,
        'w': -1.0
    }

    input_params = {
        'Delta2_p': 0., # not used if you provide cosmology
        'n_p': 0., # not used if you provide cosmology
        'mF': 0.66,
        'sigT_Mpc': 0.13,
        'gamma': 1.5,
        'kF_Mpc': 10.5
    }
    
    if(ii == 0):
        cosmo["As"] = var_input["As"]
    elif(ii == 1):
        cosmo["omch2"] = var_input["omch2"]
    elif(ii == 2):
        input_params["mF"] = var_input["mF"]
    elif(ii == 3):
        input_params["sigT_Mpc"] = var_input["sigT_Mpc"]


    info_power = {
        "cosmo": cosmo,
        "k3d_Mpc": k_Mpc,
        "mu": mu,
        "k1d_Mpc": kpar_Mpc,
        "return_p3d": True,
        "return_p1d": True,
        "z": z_test,
    }
    
    out = emulator.evaluate(
        emu_params=input_params,
        info_power=info_power,
        Nrealizations=1000
    )
    print(ii, out["linP_zs"])

    var_p1d[ii] = out["p1d"]
    var_p3d[ii] = out["p3d"]

# %%
fid_dp = 0.3501265780565268
fid_np = -2.3000471164022493

ratio_Ap = 0.37831021916336866/fid_dp
delta_np = -2.2891612818319254-fid_np

kp = 0.7
ks = 0.05
ln_kp_ks = np.log(kp/ks)

delta_ns = delta_np
ratio_As = np.exp(np.log(ratio_Ap) - delta_np * ln_kp_ks)

# %%
# deltapar = As_sam[ii]/cosmo_params0['As']
# cosmo_params['As'] = cosmo_params0['As'] * deltapar


# deltapar = cosmo_params0['ns'] - ns_sam[ii]
# cosmo_params['ns'] = ns_sam[ii]
# kp = 0.7
# ks = 0.05
# cosmo_params['As'] = cosmo_params0['As'] * (kp/ks)**deltapar

# %%
input_params = {
    'Delta2_p': 0.,
    'n_p': 0., 
    'mF': 0.66,
    'sigT_Mpc': 0.13,
    'gamma': 1.5,
    'kF_Mpc': 10.5
}

cosmo = {
    'H0': 67.0,
    'omch2': 0.12,
    'ombh2': 0.022,
    'mnu': 0.0,
    'omk': 0,
    'As': 2.006055e-09,
    'ns': 0.967565,
    'nrun': 0.0,
    'w': -1.0
}

cosmo["As"] *= ratio_As
cosmo["ns"] += delta_ns

info_power = {
    "cosmo":cosmo,
    "k3d_Mpc": k_Mpc,
    "mu": mu,
    "k1d_Mpc": kpar_Mpc,
    "return_p3d": True,
    "return_p1d": True,
    "z": z_test,
}

out = emulator.evaluate(
    emu_params=input_params,
    info_power=info_power,
    Nrealizations=1000
)

omh2_p1d = out['p1d']
omh2_p3d = out['p3d']

print(out["linP_zs"])

# %%
ii = 1
plt.plot(k_Mpc[:,0], var_p3d[ii, :,0]/omh2_p3d[:,0])
plt.plot(k_Mpc[:,0], var_p3d[ii, :,1]/omh2_p3d[:,1])
plt.plot(kpar_Mpc, orig_p1d/omh2_p1d)

plt.xscale("log")

# %%
1 {'Delta2_p': 0.37831021916336866, 'n_p': -2.2891612818319254, 'alpha_p': -0.22072016609048092, 'f_p': 0.9824077977871849}

# %%
path_fig = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"

fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 4*3))
ls = ["-", "--", ":", "-."]
labs = [r"$P_\mathrm{3D}(k, \mu=0)$", r"$P_\mathrm{3D}(k, \mu=1)$", r"$P_\mathrm{1D}$"]
labsleg = [r'$\Delta A_\mathrm{s}=+5\%$', r'$\Delta \Omega_\mathrm{M} h^2=+5\%$', r'$\Delta \bar_{F}=+1\%$', r'$\Delta \sigma_\mathrm{T}=+5\%$']
lw = 2.5
fontsize=20

for ii in range(4):
    ax[0].plot(k_Mpc[:,0], var_p3d[ii, :,0]/orig_p3d[:,0], ls[ii], label=labsleg[ii], lw=lw)
    ax[1].plot(k_Mpc[:,0], var_p3d[ii, :,1]/orig_p3d[:,1], ls[ii], lw=lw)
    
    ax[2].plot(kpar_Mpc, var_p1d[ii]/orig_p1d, ls[ii], lw=lw)

ax[0].plot(k_Mpc[:,0], omh2_p3d[:,0]/orig_p3d[:,0], ls[ii], label="Compensated Omh2", lw=lw)
ax[1].plot(k_Mpc[:,0], omh2_p3d[:,1]/orig_p3d[:,1], ls[ii], lw=lw)

ax[2].plot(kpar_Mpc, omh2_p1d/orig_p1d, ls[ii], lw=lw)


# ax[0].legend(loc="lower right", fontsize=fontsize-2)

for ii in range(3):
    ax[ii].axhline(1, linestyle=":", color="k", alpha=0.5, lw=2)
    ax[ii].tick_params(axis="both", which="major", labelsize=fontsize)
    
ax[0].axvline(5, linestyle="--", color="k", alpha=0.5, lw=2)
ax[1].axvline(5, linestyle="--", color="k", alpha=0.5, lw=2)
ax[2].axvline(4, linestyle="--", color="k", alpha=0.5, lw=2)

ax[0].set_xscale("log")

            
ax[0].set_xlabel(r'$k\, [\mathrm{Mpc}^{-1}]$', fontsize=fontsize)
ax[1].set_xlabel(r'$k\, [\mathrm{Mpc}^{-1}]$', fontsize=fontsize)
ax[2].set_xlabel(r'$k_\parallel\, [\mathrm{Mpc}^{-1}]$', fontsize=fontsize)

ax[0].set_ylabel(r"$\left(P_{\rm 3D}/P_{\rm 3D}^{\rm fid}\right)(k, \mu=0)$", fontsize=fontsize)
ax[1].set_ylabel(r"$\left(P_{\rm 3D}/P_{\rm 3D}^{\rm fid}\right)(k, \mu=1)$", fontsize=fontsize)
ax[2].set_ylabel(r"$\left(P_{\rm 1D}/P_{\rm 1D}^{\rm fid}\right)(k_\parallel)$", fontsize=fontsize)

plt.tight_layout()

plt.savefig(path_fig+'/sensitivity_power.png')
plt.savefig(path_fig+'/sensitivity_power.pdf')

# %% [markdown]
# ### Linear biases

# %%
# target redshift
nz = 10
z_test = np.linspace(2, 4.5, 10)
nrel = 2000

# target cosmology
cosmo = {
    'H0': 67.0,
    'omch2': 0.12,
    'ombh2': 0.022,
    'mnu': 0.0,
    'omk': 0,
    'As': 2.006055e-09,
    'ns': 0.967565,
    'nrun': 0.0,
    'w': -1.0
}

# random (approx central)
input_params = {
    'Delta2_p': 0., # not used if you provide cosmology
    'n_p': 0., # not used if you provide cosmology
    'mF': 0.66,
    'sigT_Mpc': 0.13,
    'gamma': 1.5,
    'kF_Mpc': 10.5
}

var_input = {
    "As": cosmo["As"] + 0.05 * cosmo["As"],
    "omch2": cosmo["omch2"] + 0.05 * cosmo["omch2"],
    "mF": input_params["mF"] + 0.01 * input_params["mF"],
    "sigT_Mpc": input_params["sigT_Mpc"] + 0.05 * input_params["sigT_Mpc"], 
}

lybias = np.zeros((nz, 3))

for ii, z in enumerate(z_test):

    info_power = {
        "cosmo": cosmo,
        "z": z,
    }
    
    out = emulator.evaluate(
        emu_params=input_params,
        info_power=info_power,
        Nrealizations=nrel
    )

    lybias[ii, 0] = out["coeffs_Arinyo"]["beta"]
    lybias[ii, 1] = out["coeffs_Arinyo"]["bias"]
    
    out = emulator.evaluate(
        emu_params=input_params,
        info_power=info_power,
        Nrealizations=nrel,
        natural_params=True
    )
    lybias[ii, 2] = out["coeffs_Arinyo"]["bias_eta"]


# %%
lybias_var = np.zeros((4, nz, 3))


for ii in range(4):
    print(ii)
    cosmo = {
        'H0': 67.0,
        'omch2': 0.12,
        'ombh2': 0.022,
        'mnu': 0.0,
        'omk': 0,
        'As': 2.006055e-09,
        'ns': 0.967565,
        'nrun': 0.0,
        'w': -1.0
    }

    input_params = {
        'Delta2_p': 0., # not used if you provide cosmology
        'n_p': 0., # not used if you provide cosmology
        'mF': 0.66,
        'sigT_Mpc': 0.13,
        'gamma': 1.5,
        'kF_Mpc': 10.5
    }
    
    if(ii == 0):
        cosmo["As"] = var_input["As"]
    elif(ii == 1):
        cosmo["omch2"] = var_input["omch2"]
    elif(ii == 2):
        input_params["mF"] = var_input["mF"]
    elif(ii == 3):
        input_params["sigT_Mpc"] = var_input["sigT_Mpc"]


    for jj, z in enumerate(z_test):

        info_power = {
            "cosmo": cosmo,
            "z": z,
        }
        
        out = emulator.evaluate(
            emu_params=input_params,
            info_power=info_power,
            Nrealizations=nrel
        )
    
        lybias_var[ii, jj, 0] = out["coeffs_Arinyo"]["beta"]
        lybias_var[ii, jj, 1] = out["coeffs_Arinyo"]["bias"]
        
        out = emulator.evaluate(
            emu_params=input_params,
            info_power=info_power,
            Nrealizations=nrel,
            natural_params=True
        )
        lybias_var[ii, jj, 2] = out["coeffs_Arinyo"]["bias_eta"]

# %%
path_fig = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"

fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 4*3))
ls = ["-", "--", ":", "-."]
labsleg = [r'$\Delta A_\mathrm{s}=+5\%$', r'$\Delta \Omega_\mathrm{M} h^2=+5\%$', r'$\Delta m_\mathrm{F}=+1\%$', r'$\Delta \sigma_\mathrm{T}=+5\%$']
lw = 2.5
fontsize=20

for jj in range(3):
    for ii in range(4):
        ax[jj].plot(z_test, lybias_var[ii, :, jj]/lybias[:, jj], ls[ii], lw=3, label=labsleg[ii])

ax[0].legend(loc="upper right", fontsize=fontsize-2)

for ii in range(3):
    ax[ii].axhline(1, linestyle=":", color="k", alpha=0.5, lw=2)
    ax[ii].tick_params(axis="both", which="major", labelsize=fontsize)

ax[2].set_xlabel(r'$z$', fontsize=fontsize)

ax[0].set_ylabel(r"$\left(\beta/\beta^{\rm fid}\right)(z)$", fontsize=fontsize)
ax[1].set_ylabel(r"$\left(b_\delta/b_\delta^{\rm fid}\right)(z)$", fontsize=fontsize)
ax[2].set_ylabel(r"$\left(b_\eta/b_\eta^{\rm fid}\right)(z)$", fontsize=fontsize)

plt.tight_layout()

plt.savefig(path_fig+'/sensitivity_lbias.png')
plt.savefig(path_fig+'/sensitivity_lbias.pdf')

# %%

# %%

# %%

# %%
labs = [r"$P_\mathrm{3D}(k, \mu=0)$", r"$P_\mathrm{3D}(k, \mu=1)$", r"$P_\mathrm{1D}$"]
lab_par = [r'$\Delta^2_\mathrm{p}$', r'$n_\mathrm{p}$', r'$m_\mathrm{F}$', r'$\gamma$', r'$\sigma_\mathrm{T}$', r'$k_\mathrm{F}$']
fontsize = 22

fig, ax = plt.subplots(3, 2, sharex=True, figsize=(8, 4*2))
ax = ax.reshape(-1)
for jj in range(2):
    for kk, par in enumerate(emu_params0):       
        ii0 = 0
        for ii in range(0, nn, 2):
            if(ii == 2):
                continue
            leg = par + '= ' + str(np.round(all_dp_vals[kk][ii], 2))
                
            if(jj == 0):
                k = k_Mpc[:,0]
                dat = var_p3d[kk, ii, :, 0] / orig_p3d[:, 0] - 1
                lss = "-"
                col = "C"+str(ii0)
            elif(jj == 1):
                k = k_Mpc[:,0]
                dat = var_p3d[kk, ii, :, 1] / orig_p3d[:, 1] - 1
                lss = "--"
                col = "C"+str(ii0)
                    
            ax[kk].plot(k, dat, ls=lss, c=col, label=leg, lw=2, alpha=0.75)
            ii0 += 1
            
        # ax[kk].legend(ncol=4)            
        ax[kk].axhline(color='k', ls=":")
        ax[kk].set_title(lab_par[kk],
            fontsize=fontsize)
        # ax[kk].set_ylabel('var/orig-1')
        if(jj == 0):
            ymax = np.max(np.abs(dp3d_range[kk, :]))*1.1
            ax[kk].set_ylim(-ymax, ymax)
            ax[kk].tick_params(axis="both", which="major", labelsize=fontsize-4)

ax[-2].set_xlabel(r'$k\, [\mathrm{Mpc}^{-1}]$',
    fontsize=fontsize)
ax[-1].set_xlabel(r'$k\, [\mathrm{Mpc}^{-1}]$',
    fontsize=fontsize)
ax[-1].set_xscale('log')
ax[-2].set_ylabel("\n ")


# create manual symbols for legend
patch1 = mpatches.Patch(color='C0', label='Min')
patch2 = mpatches.Patch(color='C1', label='Max')
ax[0].legend(handles=[patch1, patch2], ncol=2, fontsize=fontsize-6, loc="lower left")

line1 = Line2D([0], [0], label=r'$\mu=0$', color='gray', ls="-", lw=2)
line2 = Line2D([0], [0], label=r'$\mu=1$', color='gray', ls="--", lw=2)
ax[1].legend(handles=[line1, line2], ncol=2, fontsize=fontsize-6, loc="lower left")

# plt.suptitle(labs[jj], fontsize=20)
plt.tight_layout()
fig.text(
    0.005,
    0.5,
    r"$P_{\rm 3D}^\mathrm{var}/P_{\rm 3D}^\mathrm{central}-1$",
    va="center",
    rotation="vertical",
    fontsize=fontsize,
)
# plt.savefig(path_fig+'/sensitivity_P3D.png')
# plt.savefig(path_fig+'/sensitivity_P3D.pdf')

# %% [markdown]
# #### Compute Dp derivative

# %% [markdown]
# Get cosmo params from mpg_central at z=3

# %%
zcen = 3
k_Mpc = np.zeros((100, 2))
k_Mpc[:, 0] = np.geomspace(0.03, 5, 100)
k_Mpc[:, 1] = np.geomspace(0.03, 5, 100)
mu = np.zeros((100, 2))
mu[:,0] = 0
mu[:,1] = 1

kpar_Mpc = np.geomspace(0.03, 5, 100)

info_power = {
    "sim_label": "mpg_central",
    "k3d_Mpc": k_Mpc,
    "mu": mu,
    "k1d_Mpc": kpar_Mpc,
    "return_p3d": True,
    "return_p1d": True,
    # "return_cov": True,
    "z": zcen,
}

# random, we overwrite it below with z=3 data
emu_params0 = {
    'Delta2_p': 0.6179245757601503,
    'n_p': -2.348387407902965,
    'mF': 0.8711781695077168,
    'gamma': 1.7573934933568836,
    'sigT_Mpc': 0.1548976469710291,
    'kF_Mpc': 7.923157506298608,
}
range_par = {
    'Delta2_p': 0.07,
    'n_p': 0.1,
    'mF': 0.02,
    'gamma': 0.2,
    'sigT_Mpc': 0.02,
    'kF_Mpc': 2
}
emu_params = list(emu_params0.keys())

testing_data = Archive3D.get_testing_data(sim_label="mpg_central")
for ind_book in range(len(testing_data)):
    if(
        (testing_data[ind_book]['z'] == zcen)
    ):
        _id = ind_book
Ap_cen = testing_data[_id][emu_params[0]]
np_cen = testing_data[_id][emu_params[1]]
for par in emu_params:
    emu_params0[par] = testing_data[_id][par]

cosmo_params0 = copy.deepcopy(testing_data[_id]['cosmo_params'])

nn = 5
As_sam = np.linspace(cosmo_params0["As"]*(1-range_par["Delta2_p"]), cosmo_params0["As"]*(1+range_par["Delta2_p"]), nn)
ns_sam = np.linspace(cosmo_params0["ns"]-range_par["n_p"], cosmo_params0["ns"]+range_par["n_p"], nn)
print(As_sam)
print(ns_sam)

out = emulator.evaluate(
    emu_params=emu_params0,
    info_power=info_power,
    Nrealizations=100
)

orig_p1d = out['p1d']
orig_p3d = out['p3d']

# emu_params0["Delta2_p"] = out["linP_zs"]['Delta2_p']
# emu_params0["n_p"] = out["linP_zs"]['n_p']


# %%
all_dp_vals = []
var_p1d = np.zeros((6, nn, orig_p1d.shape[0]))
var_p3d = np.zeros((6, nn, orig_p3d.shape[0], orig_p3d.shape[1]))

for jj, par in enumerate(emu_params0):

    if((par == "Delta2_p") | (par == "n_p")):
        dp_vals = np.zeros(nn)
    else:
        dp_vals = np.linspace(emu_params0[par]-range_par[par], emu_params0[par]+range_par[par], nn)
    
    for ii in range(nn):
    
        cosmo_params = copy.deepcopy(cosmo_params0)
        emu_params = copy.deepcopy(emu_params0)

        if(par == "Delta2_p"):
            deltapar = As_sam[ii]/cosmo_params0['As']
            cosmo_params['As'] = cosmo_params0['As'] * deltapar
        elif(par == "n_p"):
            deltapar = cosmo_params0['ns'] - ns_sam[ii]
            cosmo_params['ns'] = ns_sam[ii]
            kp = 0.7
            ks = 0.05
            cosmo_params['As'] = cosmo_params0['As'] * (kp/ks)**deltapar
        else:
            emu_params[par] = dp_vals[ii]
            
        out = p3d_emu.predict_P3D_Mpc(
            cosmo=cosmo_params,
            z=z, 
            emu_params=emu_params,
            k_Mpc=k_Mpc,
            mu=mu,
            kpar_Mpc = kpar_Mpc
        )

        if(par == "Delta2_p"):
            dp_vals[ii] = out['linP_zs']["Delta2_p"]
        elif(par == "n_p"):
            dp_vals[ii] = out['linP_zs']["n_p"]
        var_p1d[jj, ii] = out['p1d']
        var_p3d[jj, ii] = out['p3d'] 

        # if((par == "Delta2_p") | (par == "n_p")):
        #     if(par == "Delta2_p"):
        #         deltapar1 = out['linP_zs'][par] / emu_params0[par]
        #     elif(par == "n_p"):
        #         deltapar1 = emu_params0[par] - out['linP_zs'][par]
            
        #     print(out['linP_zs']["Delta2_p"], out['linP_zs']["n_p"], deltapar, deltapar1)
    
    
    all_dp_vals.append(dp_vals)
all_dp_vals = np.array(all_dp_vals)
print(all_dp_vals)    

# %%
dp3d_range = np.zeros((6, 2))
dp1d_range = np.zeros((6, 2))
for ii in range(dp3d_range.shape[0]):
    dp3d_range[ii, 0] = np.max(var_p3d[ii]/orig_p3d[None, ...])-1
    dp3d_range[ii, 1] = np.min(var_p3d[ii]/orig_p3d[None, ...])-1
    dp1d_range[ii, 0] = np.max(var_p1d[ii]/orig_p1d[None, ...])-1
    dp1d_range[ii, 1] = np.min(var_p1d[ii]/orig_p1d[None, ...])-1

# %%
labs = [r"$P_\mathrm{3D}(k, \mu=0)$", r"$P_\mathrm{3D}(k, \mu=1)$", r"$P_\mathrm{1D}$"]
lab_par = [r'$\Delta^2_\mathrm{p}$', r'$n_\mathrm{p}$', r'$m_\mathrm{F}$', r'$\gamma$', r'$\sigma_\mathrm{T}$', r'$k_\mathrm{F}$']
fontsize = 22

fig, ax = plt.subplots(3, 2, sharex=True, figsize=(8, 4*2))
ax = ax.reshape(-1)
for jj in range(2):
    for kk, par in enumerate(emu_params0):       
        ii0 = 0
        for ii in range(0, nn, 2):
            if(ii == 2):
                continue
            leg = par + '= ' + str(np.round(all_dp_vals[kk][ii], 2))
                
            if(jj == 0):
                k = k_Mpc[:,0]
                dat = var_p3d[kk, ii, :, 0] / orig_p3d[:, 0] - 1
                lss = "-"
                col = "C"+str(ii0)
            elif(jj == 1):
                k = k_Mpc[:,0]
                dat = var_p3d[kk, ii, :, 1] / orig_p3d[:, 1] - 1
                lss = "--"
                col = "C"+str(ii0)
                    
            ax[kk].plot(k, dat, ls=lss, c=col, label=leg, lw=2, alpha=0.75)
            ii0 += 1
            
        # ax[kk].legend(ncol=4)            
        ax[kk].axhline(color='k', ls=":")
        ax[kk].set_title(lab_par[kk],
            fontsize=fontsize)
        # ax[kk].set_ylabel('var/orig-1')
        if(jj == 0):
            ymax = np.max(np.abs(dp3d_range[kk, :]))*1.1
            ax[kk].set_ylim(-ymax, ymax)
            ax[kk].tick_params(axis="both", which="major", labelsize=fontsize-4)

ax[-2].set_xlabel(r'$k\, [\mathrm{Mpc}^{-1}]$',
    fontsize=fontsize)
ax[-1].set_xlabel(r'$k\, [\mathrm{Mpc}^{-1}]$',
    fontsize=fontsize)
ax[-1].set_xscale('log')
ax[-2].set_ylabel("\n ")


# create manual symbols for legend
patch1 = mpatches.Patch(color='C0', label='Min')
patch2 = mpatches.Patch(color='C1', label='Max')
ax[0].legend(handles=[patch1, patch2], ncol=2, fontsize=fontsize-6, loc="lower left")

line1 = Line2D([0], [0], label=r'$\mu=0$', color='gray', ls="-", lw=2)
line2 = Line2D([0], [0], label=r'$\mu=1$', color='gray', ls="--", lw=2)
ax[1].legend(handles=[line1, line2], ncol=2, fontsize=fontsize-6, loc="lower left")

# plt.suptitle(labs[jj], fontsize=20)
plt.tight_layout()
fig.text(
    0.005,
    0.5,
    r"$P_{\rm 3D}^\mathrm{var}/P_{\rm 3D}^\mathrm{central}-1$",
    va="center",
    rotation="vertical",
    fontsize=fontsize,
)
plt.savefig(path_fig+'/sensitivity_P3D.png')
plt.savefig(path_fig+'/sensitivity_P3D.pdf')

# %%
labs = [r"$P_\mathrm{3D}(k, \mu=0)$", r"$P_\mathrm{3D}(k, \mu=1)$", r"$P_\mathrm{1D}$"]
lab_par = [r'$\Delta^2_\mathrm{p}$', r'$n_\mathrm{p}$', r'$m_\mathrm{F}$', r'$\gamma$', r'$\sigma_\mathrm{T}$', r'$k_\mathrm{F}$']
fontsize = 22

fig, ax = plt.subplots(3, 2, sharex=True, figsize=(8, 4*2))
ax = ax.reshape(-1)
for kk, par in enumerate(emu_params0):        
    for ii in range(0, nn, 2):
        if(ii == 2):
            continue
        leg = par + '= ' + str(np.round(all_dp_vals[kk][ii], 2))
        
        k = kpar_Mpc
        dat = var_p1d[kk, ii] / orig_p1d - 1
                
        ax[kk].plot(k, dat, label=leg, lw=2)
        
    # ax[kk].legend(ncol=4)            
    ax[kk].axhline(color='k', ls=":")
    ax[kk].set_title(lab_par[kk],
            fontsize=fontsize)
    ymax = np.max(np.abs(dp1d_range[kk, :]))*1.1
    ax[kk].set_ylim(-ymax, ymax)
    ax[kk].tick_params(axis="both", which="major", labelsize=fontsize-4)
        
ax[-2].set_xlabel(r'$k_\parallel\, [\mathrm{Mpc}^{-1}]$', fontsize=fontsize)
ax[-1].set_xlabel(r'$k_\parallel\, [\mathrm{Mpc}^{-1}]$', fontsize=fontsize)
ax[-1].set_xscale('log')
ax[-2].set_ylabel("\n ")
# plt.suptitle(labs[jj], fontsize=20)
fig.text(
    0.005,
    0.5,
    r"$P_{\rm 1D}^\mathrm{var}/P_{\rm 1D}^\mathrm{central}-1$",
    va="center",
    rotation="vertical",
    fontsize=fontsize,
)
plt.tight_layout()
plt.savefig(path_fig+'/sensitivity_P1D.png')
plt.savefig(path_fig+'/sensitivity_P1D.pdf')

# %% [markdown]
# ## Parameter sensitivity

# %%
# random, we overwrite it below with z=3 data
emu_params0 = {
    'Delta2_p': 0.6179245757601503,
    'n_p': -2.348387407902965,
    'mF': 0.8711781695077168,
    'gamma': 1.7573934933568836,
    'sigT_Mpc': 0.1548976469710291,
    'kF_Mpc': 7.923157506298608,
}
range_par = {
    'Delta2_p': 0.07,
    'n_p': 0.1,
    'mF': 0.02,
    'gamma': 0.2,
    'sigT_Mpc': 0.02,
    'kF_Mpc': 2
}
emu_params = list(emu_params0.keys())

testing_data = Archive3D.get_testing_data(sim_label="mpg_central")
for ind_book in range(len(testing_data)):
    if(
        (testing_data[ind_book]['z'] == 3)
    ):
        _id = ind_book
Ap_cen = testing_data[_id][emu_params[0]]
np_cen = testing_data[_id][emu_params[1]]
for par in emu_params:
    emu_params0[par] = testing_data[_id][par]

fp_cen = testing_data[_id]['f_p']

z = 3
cosmo_params0 = copy.deepcopy(testing_data[_id]['cosmo_params'])


# %%
emu_params = copy.deepcopy(emu_params0)
emu_params["f_p"] = fp_cen

out = p3d_emu.predict_P3D_Mpc(
    cosmo=cosmo_params0,
    z=z, 
    emu_params=emu_params,
    natural_params=True,
)


# %%
nn = 25
nparams = len(out['coeffs_Arinyo'].keys())
ninput = len(emu_params0.keys())
param_evol = np.zeros((nn, ninput, nparams))
param_input = np.zeros((nn, ninput))

As_sam = np.linspace(cosmo_params0["As"]*(1-range_par["Delta2_p"]), cosmo_params0["As"]*(1+range_par["Delta2_p"]), nn)
ns_sam = np.linspace(cosmo_params0["ns"]-range_par["n_p"], cosmo_params0["ns"]+range_par["n_p"], nn)

for jj, par in enumerate(emu_params0):
    dp2_vals = np.linspace(emu_params0[par]-range_par[par], emu_params0[par]+range_par[par], nn)
    
    for ii in range(nn):
        cosmo_params = copy.deepcopy(cosmo_params0)
        emu_params = copy.deepcopy(emu_params0)

        if(par == "Delta2_p"):
            deltapar = As_sam[ii]/cosmo_params0['As']
            cosmo_params['As'] = cosmo_params0['As'] * deltapar
        elif(par == "n_p"):
            deltapar = cosmo_params0['ns'] - ns_sam[ii]
            cosmo_params['ns'] = ns_sam[ii]
            kp = 0.7
            ks = 0.05
            cosmo_params['As'] = cosmo_params0['As'] * (kp/ks)**deltapar
        else:
            emu_params[par] = dp2_vals[ii]
            emu_params["f_p"] = fp_cen
            param_input[ii, jj] = dp2_vals[ii]
            
        out = p3d_emu.predict_P3D_Mpc(
            cosmo=cosmo_params,
            z=z, 
            emu_params=emu_params,
            natural_params=True,
            verbose=False,
            return_cov=False
        )
        param_evol[ii, jj] = np.array(list(out['coeffs_Arinyo'].values()))
            
        if(par == "Delta2_p"):
            param_input[ii, jj] = out['linP_zs']["Delta2_p"]
        elif(par == "n_p"):
            param_input[ii, jj] = out['linP_zs']["n_p"]
        else:
            param_input[ii, jj] = dp2_vals[ii]

# %%
param_evol.shape

# %%
# name_params = list(Arinyo_emu[0].keys())
name_params = ['bias', 'bias_eta', 'q1', 'q2', 'kv', 'av', 'bv', 'kp']
name2label = {
    'bias':r"$b_\delta$", 
    'bias_eta':r"$b_\eta$", 
    'q1':r"$q_1$", 
    'q2':r"$q_2$",
    'kv':r"$k_\mathrm{v}$", 
    'av':r"$a_\mathrm{v}$", 
    'bv':r"$b_\mathrm{v}$", 
    'kp':r"$k_\mathrm{p}$", 
}
par2label = {
    'Delta2_p': r'$\Delta^2_\mathrm{p}$', 
    'n_p': r'$n_\mathrm{p}$', 
    'mF': r'$m_\mathrm{F}$', 
    'gamma': r'$\gamma$', 
    'sigT_Mpc': r'$\sigma_\mathrm{T}$', 
    'kF_Mpc': r'$k_\mathrm{F}$',
}
index2axis = [0, 1, 2, 4, 5, 6, 7, 3] 

# %%
print(emu_params0.keys())
print(out['coeffs_Arinyo'].keys())

# %%

# %%
fig, ax = plt.subplots(4, 2, sharex=True, figsize=(8, 4*2))
ax = ax.reshape(-1)
fontsize = 16
ls = ['-', '--', ':', '-.', (0, (5, 5)), (0, (3, 1, 1, 1, 1, 1))]

for jj, par2 in enumerate(out['coeffs_Arinyo']):
    jj0 = index2axis[jj]
    ax[jj0].set_ylabel(name2label[par2], fontsize=fontsize)  
    for ii, par1 in enumerate(emu_params0):
        col = "C" + str(ii)
        if(ii == jj0):
            lab = "x = "+par2label[par1] 
        else:
            lab = None
            
        xmin = param_input[:, ii].min()
        xmax = param_input[:, ii].max()
        x = (param_input[:, ii]-xmin)/(xmax-xmin)*2-1
        y = param_evol[:, ii, jj]
        pfit = np.polyfit(x, y , 1)
        yy = pfit[0] * x + pfit[1]
        # ax[jj0].plot(x, y, label=lab, lw=2, c=col)
        ax[jj0].plot(x, yy, label=lab, lw=2, c=col, ls=ls[ii], alpha=0.75)
ax[-2].set_xlabel("Variation in x", fontsize=fontsize)
ax[-1].set_xlabel("Variation in x", fontsize=fontsize)

for ii in range(len(emu_params0.keys())):
    ax[ii].legend(ncol=2, loc="upper left", fontsize=fontsize)
plt.tight_layout()
plt.savefig(path_fig+'/sensitivity_arinyo.png')
plt.savefig(path_fig+'/sensitivity_arinyo.pdf')

# %%
