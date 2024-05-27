# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Running simulation
#
# The aim of this notebook is to understand why ForestFlow does not work for a simulation with nonzero running of the spectral index

# +
# %load_ext autoreload
# %autoreload 2

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.integrate import simpson

from forestflow.model_p3d_arinyo import ArinyoModel
from forestflow.archive import GadgetArchive3D
from forestflow.P3D_cINN import P3DEmulator
from forestflow.rebin_p3d import p3d_allkmu, get_p3d_modes, p3d_rebin_mu
from lace.cosmo import camb_cosmo

from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

def ls_level(folder, nlevels):
    for ii in range(nlevels):
        folder = os.path.dirname(folder)
    folder += "/"
    return folder


path_program = ls_level(os.getcwd(), 2)
print(path_program)
sys.path.append(path_program)
# -

# ## LOAD P3D ARCHIVE

# +
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
# -

# ### Load emulator

# +
training_type = "Arinyo_minz"
nparams = 8
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
# -

# ### General stuff

# +

zcen = 3


n_mubins = 4
kmax_3d_plot = 4
kmax_1d_plot = 4
kmax_fit = 3

sim = Archive3D.training_data[0]

k3d_Mpc = sim['k3d_Mpc']
mu3d = sim['mu3d']
p3d_Mpc = sim['p3d_Mpc']
kmu_modes = get_p3d_modes(kmax_3d_plot)

mask_3d = k3d_Mpc[:, 0] <= kmax_3d_plot

mask_1d = (sim['k_Mpc'] <= kmax_1d_plot) & (sim['k_Mpc'] > 0)
k1d_Mpc = sim['k_Mpc'][mask_1d]
p1d_Mpc = sim['p1d_Mpc'][mask_1d]
# -

# #### Central simulation

# +


sim_label = "mpg_central"

info_power = {
    "sim_label": sim_label,
    "k3d_Mpc": k3d_Mpc[mask_3d, :],
    "mu": mu3d[mask_3d, :],
    "kmu_modes": kmu_modes,
    "k1d_Mpc": k1d_Mpc,
    "return_p3d": True,
    "return_p1d": True,
    "return_cov": True,
    "z": zcen,
}

sim_label = info_power["sim_label"]
test_sim = Archive3D.get_testing_data(
    sim_label, force_recompute_plin=False
)
test_sim_z = [d for d in test_sim if d["z"] == info_power["z"]]
emu_params_cen = test_sim_z[0]

out = emulator.evaluate(
    emu_params=emu_params_cen,
    info_power=info_power,
    natural_params=True,
    Nrealizations=100
)

_ = p3d_rebin_mu(out["k_Mpc"], out["mu"], test_sim_z[0]["p3d_Mpc"][mask_3d], kmu_modes, n_mubins=n_mubins)
knew, munew, rebin_p3d_simcen, mu_bins = _

_ = p3d_rebin_mu(out["k_Mpc"], out["mu"], out["p3d"], kmu_modes, n_mubins=n_mubins)
knew, munew, rebin_p3d_emucen, mu_bins = _

_ = p3d_rebin_mu(out["k_Mpc"], out["mu"], out["p3d_std"], kmu_modes, n_mubins=n_mubins)
knew, munew, rebin_p3d_std_emucen, mu_bins = _

_ = p3d_rebin_mu(out["k_Mpc"], out["mu"], out["Plin"], kmu_modes, n_mubins=n_mubins)
knew, munew, rebin_plincen, mu_bins = _

norm_p1d = out["k1d_Mpc"]/np.pi
p1d_emucen = norm_p1d * out["p1d"]
p1d_std_emucen = norm_p1d * out["p1d_std"]
p1d_simcen = norm_p1d * test_sim_z[0]["p1d_Mpc"][mask_1d]
# -

# #### Running simulation

# +


sim_label = "mpg_running"

info_power = {
    "sim_label": sim_label,
    "k3d_Mpc": k3d_Mpc[mask_3d, :],
    "mu": mu3d[mask_3d, :],
    "kmu_modes": kmu_modes,
    "k1d_Mpc": k1d_Mpc,
    "return_p3d": True,
    "return_p1d": True,
    "return_cov": True,
    "z": zcen,
}

sim_label = info_power["sim_label"]
test_sim = Archive3D.get_testing_data(
    sim_label, force_recompute_plin=False
)
test_sim_z = [d for d in test_sim if d["z"] == info_power["z"]]
emu_params_run = test_sim_z[0]

out = emulator.evaluate(
    emu_params=emu_params_run,
    info_power=info_power,
    natural_params=True,
    Nrealizations=100
)

_ = p3d_rebin_mu(out["k_Mpc"], out["mu"], test_sim_z[0]["p3d_Mpc"][mask_3d], kmu_modes, n_mubins=n_mubins)
knew, munew, rebin_p3d_simrun, mu_bins = _

_ = p3d_rebin_mu(out["k_Mpc"], out["mu"], out["p3d"], kmu_modes, n_mubins=n_mubins)
knew, munew, rebin_p3d_emurun, mu_bins = _

_ = p3d_rebin_mu(out["k_Mpc"], out["mu"], out["p3d_std"], kmu_modes, n_mubins=n_mubins)
knew, munew, rebin_p3d_std_emurun, mu_bins = _

_ = p3d_rebin_mu(out["k_Mpc"], out["mu"], out["Plin"], kmu_modes, n_mubins=n_mubins)
knew, munew, rebin_plinrun, mu_bins = _

norm_p1d = out["k1d_Mpc"]/np.pi
p1d_emurun = norm_p1d * out["p1d"]
p1d_std_emurun = norm_p1d * out["p1d_std"]
p1d_simrun = norm_p1d * test_sim_z[0]["p1d_Mpc"][mask_1d]
# -

# #### Get P1D

emu_params_run.keys()

# +

zs = [zcen]
cosmo_cen = camb_cosmo.get_cosmology_from_dictionary(emu_params_cen["cosmo_params"])
cosmo_run = camb_cosmo.get_cosmology_from_dictionary(emu_params_run["cosmo_params"])

camb_results_cen = camb_cosmo.get_camb_results(cosmo_cen, zs=zs, camb_kmax_Mpc=200)
camb_results_run = camb_cosmo.get_camb_results(cosmo_run, zs=zs, camb_kmax_Mpc=200)

ks, zs, pk_cen = camb_results_cen.get_linear_matter_power_spectrum(
        var1=8, var2=8, hubble_units=False, nonlinear=False, k_hunit=False
    )
ks, zs, pk_run = camb_results_run.get_linear_matter_power_spectrum(
        var1=8, var2=8, hubble_units=False, nonlinear=False, k_hunit=False
    )
# -

# #### Ratio Plin

_ = (ks > 0.01) & (ks < 50)
plt.plot(ks[_], pk_run[0][_]/pk_cen[0][_])
plt.xscale("log")
# plt.savefig("ratio_pklin_cen_run.png")

# #### Integrate Plins to get P1Ds

def int_plin(k_par, k_Mpc, pk_Mpc):

    k_perp_min=0.001
    k_perp_max=100
    n_k_perp=99
    
    ln_k_perp = np.linspace(
        np.log(k_perp_min), np.log(k_perp_max), n_k_perp
    )
    
    dlnk = ln_k_perp[1] - ln_k_perp[0]
    
    k_perp = np.exp(ln_k_perp)
    k = np.sqrt(k_par**2 + k_perp**2)
    mu = k_par / k

    inter_pk = (1 / (2 * np.pi)) * k_perp**2 * np.exp(np.interp(np.log(k), np.log(k_Mpc), np.log(pk_Mpc)))
    inte = simpson(inter_pk, ln_k_perp, dx=dlnk)
    
    return k, inter_pk, inte


# +
k_par = 1
    
k, inter_pk_cen, int_cen = int_plin(k_par, ks, pk_cen[0])
k, inter_pk_run, int_run = int_plin(k_par, ks, pk_run[0])
print((int_run/int_cen-1) * 100)

plt.loglog(k, inter_pk_cen)
plt.loglog(k, inter_pk_run)

plt.ylim(0.01)
# -

# Let's compute the ratio between the P1Ds

p1d_lin_cen = np.zeros((out["k1d_Mpc"].shape[0]))
p1d_lin_run = np.zeros((out["k1d_Mpc"].shape[0]))
for ii, kpar in enumerate(out["k1d_Mpc"]):
    _, _, p1d_lin_cen[ii] = int_plin(kpar, ks, pk_cen[0])
    _, _, p1d_lin_run[ii] = int_plin(kpar, ks, pk_run[0])
    

plt.plot(out["k1d_Mpc"], p1d_emurun/p1d_emucen-1, label="ForestFlow")
plt.plot(out["k1d_Mpc"], p1d_lin_run/p1d_lin_cen-1, label="P1D from lin th")
plt.legend()
plt.xscale("log")

# We thus expect percent-level differences for P1Ds from cosmologies with the same amplitude and slope of the power spectrum but different runnings. However, this is not the case in the simulations. In fact, for large scales, it is the opposite.

plt.plot(out["k1d_Mpc"], p1d_emurun/p1d_emucen-1, label="ForestFlow")
plt.plot(out["k1d_Mpc"], p1d_lin_run/p1d_lin_cen-1, label="P1D from lin th")
plt.plot(out["k1d_Mpc"], p1d_simrun/p1d_simcen-1, label="Simulation")
plt.legend()
plt.xscale("log")
# plt.savefig("ratio_P1D_sims.png")

# For P3D, we also have more discrepancy between simulation and linear theory than between simulation and emulator

for ii in range(4):
    col = f"C{ii}"
    plt.plot(knew[:,ii], rebin_p3d_simrun[:,ii]/rebin_p3d_simcen[:,ii], col)
    plt.plot(knew[:,ii], rebin_p3d_emurun[:,ii]/rebin_p3d_emucen[:,ii], col+'--')
ii = 0
plt.plot(knew[:,ii], rebin_plinrun[:,ii]/rebin_plincen[:,ii], "k")
    # plt.loglog(knew[:,ii], rebin_plin2[:,ii])
plt.xscale("log")


