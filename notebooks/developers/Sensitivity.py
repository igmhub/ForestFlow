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
#
# Need to be updated

# %% [markdown]
# For paper 
#     
# %%%%%%%%%%%%%%%%%%%
# \begin{figure*}
# \includegraphics[width=\columnwidth]{figures/sensitivity_P3D.pdf}
# \includegraphics[width=\columnwidth]{figures/sensitivity_P1D.pdf}
# \centering
# \caption{Dependence of \pthreed (left panel) and \poned (right panel) on \forestflow parameters. Each panel displays the results for variations in the parameter quoted at its top. The blue and orange colors show the relative difference between \forestflow predictions for the \simcentral simulation at $z=3$ and \forestflow predictions for the minimum and maximum value of the target parameter for the \lacehc simulations at $z=3$, respectively. In the left panel, solid and dashed lines depict the results for $\mu=0$ and 1, respectively. As we can see, the dependence of \pthreed and \poned upon $n_\mathrm{p}$ and \mflux is the strongest among the cosmological and IGM parameters, respectively.
# }
# \label{fig:sensitivity}
# \end{figure*}
# %%%%%%%%%%%%%%%%%%%
#
# \section{Sensitivity of \forestflow to input parameters (WIP)}
# \label{sec:results_sensitivity}
#
# \jch{Start with dependence of Arinyo on input parameters, then statistics.}
#
# In Fig.~\ref{fig:sensitivity}, we display the dependence \pthreed and \poned on each of the \forestflow parameters at a time. The blue and orange colors show the relative difference between \forestflow predictions for the \simcentral simulation at $z=3$ and \forestflow predictions for the minimum and maximum value of the target parameter for the \lacehc simulations at $z=3$, respectively. Specifically, the minimum, central, and maximum values of the parameters are $\Delta_\mathrm{p}^2=0.32$, 0.35, and 0.37; $n_\mathrm{p}=-2.40$, -2.30, and -2.20; $\mflux=0.64$, 0.66, and 0.68; $\sigma_\mathrm{T}=0.11$, 0.13, and 0.15; $\gamma=1.31$, 1.51, and 1.71; and $k_\mathrm{F}=8.63$, 10.63, and 12.63. As we can see, the dependence of \pthreed and \poned upon $n_\mathrm{p}$ and \mflux is the strongest. In what follows, we compare the previous results with measurements from a set of hydrodynamical simulations run by varying one of the input parameters at a time while holding fixed all others \citep{mcdonald2003MeasurementCosmologicalGeometry, mcdonald2005LinearTheoryPower}:
#
# \begin{itemize}
#     \item $\Delta_\mathrm{p}^2$. The perpendicular modes of \pthreed increase with this property on all scales, while parallel modes and \poned do so only on scales larger than $\simeq1.3\iMpc$ and decrease on smaller scales. The first behavior is explained by the linear relation between $\Delta_\mathrm{p}^2$ and $A_\mathrm{s}$, the large-scale amplitude of the power spectrum. The second trend is driven by the increasing strength of nonlinear peculiar velocities when the power increases; this is because larger velocities induce stronger suppression of small-scale power along the line of sight. \afr{I don't understand why you talk about $A_s$ here.} \jch{I am following the same line of thought as in McDonald's papers, but I am happy to change it if you have a better explanation in mind.}
#     These results are fully compatible with measurements from simulations ran varying $\Delta_\mathrm{p}^2$ while holding fixed all other parameters \citep{mcdonald2003MeasurementCosmologicalGeometry, mcdonald2005LinearTheoryPower}.
#
#     \item $n_\mathrm{p}$. We naively expect that an increase in the slope decreases the power on large scales while increasing it on small scales. However, as discussed above, an increase in the power also increases the amplitude of nonlinear velocities, thereby leading to a stronger suppression of the power on small scales. Taken together, these physical effects explain the trends displayed by all statistics, which also agree with those found in simulations.
#
#     \item \mflux. The amplitude of \pthreed and \poned decreases with increasing flux across all scales; this is because \jch{Andreu: could you explain this?}. 
#     \afr{Yes, I think we can write a simple toy model to understand this. We can talk about this in person.}
#     These results are also in line with those found in simulations.
#
#     \item $\sigma_\mathrm{T}$. This parameter controls the strength of thermal broadening (see \S\ref{sec:strategy_params}), which explains why its increase primarily suppresses the line-of-sight power on small scales. It also induces small changes in the amplitude on large scales, also found in simulations. \jch{Andreu: any explanation for this behavior?}
#
#     \item $\gamma$. An increase in this parameter decreases the optical depth and increases the thermal broadening, which explains the decreased power on large scales and the stronger line-of-sight suppression. 
#     \afr{I thought you were varying this parameter while keeping the others fixed? If so, there should be no change in optical depth or in thermal broadening...} \jch{Explain in detail the dependence with the size of the systems.}
#     Both trends are in agreement with simulations.
#
#     \item $k_\mathrm{F}$. This parameter controls the pressure smoothing scale, and thus it primarily controls the small-scale behavior of the power spectra. As expected, an increase leads to an increment of the small-scale power for perpendicular scales and \poned. On the other hand, the power decreases for parallel scales, which is explained because \jch{Andreu: could you explain this trend?} 
#     \afr{I'm not sure the large-scale behaviour in P3D is easy to explain intuitively. You are messing with the mapping between matter fluctuation and flux when changing very high-k, you are changing the biases. Why in this direction? I don't know. It is also possible that the emulator is not capturing very well the dependency with $k_F$, but the plot is a bit busy and difficult to read.}. \jch{Plot fixed, check again same scaling as in sims} No pairs of simulations have been run while holding fixed this parameter, but a change in redshift together with the rescaling of other parameters led to similar results \citep{mcdonald2003MeasurementCosmologicalGeometry}.
# \end{itemize}

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
# ## Load stuff

# %%
path_fig = '/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/'

folder_lya_data = os.path.dirname(forestflow.__path__[0]) + "/data/best_arinyo/"

Archive3D = GadgetArchive3D(
    base_folder=os.path.dirname(forestflow.__path__[0]),
    folder_data=folder_lya_data,
    force_recompute_plin=False,
    average="both",
)
print(len(Archive3D.training_data))

# %%
training_type = "Arinyo_minz"
model_path=path_program+"/data/emulator_models/mpg_jointz.pt"

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
# ## Pk sensitivity

# %% [markdown]
# #### Compute Dp derivative

# %% [markdown]
# Get cosmo params from mpg_central at z=3

# %%
# Nsim = 30
# Nz = 11
# zs = np.flip(np.arange(2, 4.6, 0.25))

# k_Mpc = Archive3D.training_data[0]["k3d_Mpc"]
# mu = Archive3D.training_data[0]["mu3d"]

# k_mask = (k_Mpc < 5) & (k_Mpc > 0)

# k_Mpc = k_Mpc[k_mask]
# mu = mu[k_mask]


# k_p1d_Mpc_all = Archive3D.training_data[0]["k_Mpc"]
# k_p1d_Mpc = Archive3D.training_data[0]["k_Mpc"]
# k1d_mask = (k_p1d_Mpc < 5) & (k_p1d_Mpc > 0)
# k_p1d_Mpc = k_p1d_Mpc[k1d_mask]
# norm = k_p1d_Mpc / np.pi
# norm_all = k_p1d_Mpc_all / np.pi

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
