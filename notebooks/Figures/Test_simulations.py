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
# # Central simulation at z=3, l10 test simulations

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from forestflow.model_p3d_arinyo import ArinyoModel
from forestflow.archive import GadgetArchive3D
from forestflow.P3D_cINN import P3DEmulator
from forestflow.plots.test_sims import (
    plot_p1d_test_sims, 
    plot_p3d_test_sims, 
    get_modes, 
    plot_p1d_snap,
    plot_p3d_snap
)
from forestflow.utils import params_numpy2dict
from forestflow.rebin_p3d import p3d_allkmu, get_p3d_modes, p3d_rebin_mu

from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"


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
# ## LOAD P3D ARCHIVE

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


# %% [markdown]
# ## Load emulator

# %%
# training_type = "Arinyo_min_q1"
# training_type = "Arinyo_min_q1_q2"
# training_type = "Arinyo_min"
training_type = "Arinyo_minz"

if (training_type == "Arinyo_min_q1"):
    nparams = 7
    model_path = path_program+"/data/emulator_models/mpg_q1/mpg_hypercube.pt"
elif(training_type == "Arinyo_min"):
    nparams = 8
    # model_path = path_program+"/data/emulator_models/mpg_q1_q2/mpg_hypercube.pt"
    model_path=path_program+"/data/emulator_models/mpg_last.pt"
elif(training_type == "Arinyo_minz"):
    nparams = 8
    # model_path = path_program+"/data/emulator_models/mpg_q1_q2/mpg_hypercube.pt"
    model_path=path_program+"/data/emulator_models/mpg_minz.pt"

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
# #### General stuff

# %%
Nsim = 30
zs = np.flip(np.arange(2, 4.6, 0.25))
Nz = zs.shape[0]


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

# %% [markdown]
# ### Central simulation

# %%
zcen = 3

info_power = {
    "sim_label": "mpg_central",
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
emu_params = test_sim_z[0]

out = emulator.evaluate(
    emu_params=emu_params,
    info_power=info_power,
    natural_params=True,
    Nrealizations=1000
)

# %% [markdown]
# #### Rebin data

# %%
_ = p3d_rebin_mu(out["k_Mpc"], out["mu"], test_sim_z[0]["p3d_Mpc"][mask_3d], kmu_modes, n_mubins=n_mubins)
knew, munew, rebin_p3d_sim, mu_bins = _

_ = p3d_rebin_mu(out["k_Mpc"], out["mu"], out["p3d"], kmu_modes, n_mubins=n_mubins)
knew, munew, rebin_p3d_emu, mu_bins = _

_ = p3d_rebin_mu(out["k_Mpc"], out["mu"], out["p3d_std"], kmu_modes, n_mubins=n_mubins)
knew, munew, rebin_p3d_std_emu, mu_bins = _

_ = p3d_rebin_mu(out["k_Mpc"], out["mu"], out["Plin"], kmu_modes, n_mubins=n_mubins)
knew, munew, rebin_plin, mu_bins = _

# %%
norm_p1d = out["k1d_Mpc"]/np.pi
p1d_emu = norm_p1d * out["p1d"]
p1d_std_emu = norm_p1d * out["p1d_std"]
p1d_sim = norm_p1d * test_sim_z[0]["p1d_Mpc"][mask_1d]

# %%
folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"
plot_p3d_snap(
    folder, 
    knew, 
    munew,
    rebin_p3d_sim/rebin_plin,
    rebin_p3d_emu/rebin_plin,
    rebin_p3d_std_emu/rebin_plin,
    mu_bins,
)

# %%
plot_p1d_snap(
    folder, 
    out["k1d_Mpc"], 
    p1d_sim,
    p1d_emu,
    p1d_std_emu,
)

# %%

# %% [markdown]
# ## TEST SIMULATIONS

# %%
sim_labels = [
    "mpg_central",    
    "mpg_seed",
    "mpg_growth",
    "mpg_neutrinos",
    "mpg_curved",
    "mpg_running",
    "mpg_reio",
]

# %%
arr_p3d_sim = np.zeros((len(sim_labels), Nz, np.sum(mask_3d), n_mubins))
arr_p3d_emu = np.zeros((len(sim_labels), Nz, np.sum(mask_3d), n_mubins))
arr_p1d_sim = np.zeros((len(sim_labels), Nz, np.sum(mask_1d)))
arr_p1d_emu = np.zeros((len(sim_labels), Nz, np.sum(mask_1d)))

for isim, sim_label in enumerate(sim_labels):    
    test_sim = Archive3D.get_testing_data(
        sim_label, force_recompute_plin=False
    )

    z_grid = [d["z"] for d in test_sim]
    for iz, z in enumerate(z_grid):
        print(sim_label, z)
        test_sim_z = [d for d in test_sim if d["z"] == z]

        info_power = {
            "sim_label": sim_label,
            "k3d_Mpc": k3d_Mpc[mask_3d, :],
            "mu": mu3d[mask_3d, :],
            "kmu_modes": kmu_modes,
            "k1d_Mpc": k1d_Mpc,
            "return_p3d": True,
            "return_p1d": True,
            # "return_cov": True,
            "z": z,
        }

        emu_params = test_sim_z[0]
        
        out = emulator.evaluate(
            emu_params=emu_params,
            info_power=info_power,
            # natural_params=True,
            Nrealizations=100
        )
        
        _ = p3d_rebin_mu(out["k_Mpc"], out["mu"], test_sim_z[0]["p3d_Mpc"][mask_3d], kmu_modes, n_mubins=n_mubins)
        knew, munew, arr_p3d_sim[isim, iz], mu_bins = _
        
        _ = p3d_rebin_mu(out["k_Mpc"], out["mu"], out["p3d"], kmu_modes, n_mubins=n_mubins)
        knew, munew, arr_p3d_emu[isim, iz], mu_bins = _
        
        arr_p1d_emu[isim, iz] = out["p1d"]
        arr_p1d_sim[isim, iz] = test_sim_z[0]["p1d_Mpc"][mask_1d]

# %%
rat_p3d = arr_p3d_emu/arr_p3d_sim - 1
rat_p1d = arr_p1d_emu/arr_p1d_sim - 1

# %%
folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"

# %%
savename = folder + "test_cosmo/test_cosmo_P3D"
for ext in [".png", ".pdf"]:
    plot_p3d_test_sims(
        sim_labels,
        knew,
        munew,
        rat_p3d,
        mu_bins=mu_bins,
        savename=savename+ext,
        fontsize=20,
    )

# %%

# %%
savename = folder + "test_cosmo/test_cosmo_P1D"
for ext in [".png", ".pdf"]:
    plot_p1d_test_sims(
        sim_labels,
        out["k1d_Mpc"],
        rat_p1d,
        savename=savename+ext,
        fontsize=20,
    );

# %%
# savename = folder + "test_cosmo/test_cosmo_P1D_smooth_ideal"
# for ext in [".png", ".pdf"]:
#     plot_p1d_test_sims(
#         sim_labels,
#         k_p1d_Mpc,
#         fractional_error_P1D_arinyo,
#         savename=savename+ext,
#         fontsize=20,
#     )

# %%
# savename = folder + "test_cosmo/test_cosmo_P1D_fit_ideal"
# for ext in [".png", ".pdf"]:
#     plot_p1d_test_sims(
#         sim_labels,
#         k_p1d_Mpc,
#         fractional_error_P1D_bench,
#         savename=savename+ext,
#         fontsize=20,
#     )

# %% [markdown]
# ## PLOT P3D

# %%

# %%
savename = folder + "test_cosmo/test_cosmo_P3D_smooth.pdf"
plot_p3d_test_sims(
    sim_labels,
    k_Mpc,
    mu,    
    k_mask,
    fractional_error_P3D_arinyo,
    savename=savename,
    fontsize=20,
)
savename = folder + "test_cosmo/test_cosmo_P3D_smooth.png"
plot_p3d_test_sims(
    sim_labels,
    k_Mpc,
    mu,    
    k_mask,
    fractional_error_P3D_arinyo,
    savename=savename,
    fontsize=20,
)

# %%
savename = folder + "test_cosmo/test_cosmo_P3D_fitq2.pdf"
plot_p3d_test_sims(
    sim_labels,
    k_Mpc,
    mu,    
    k_mask,
    fractional_error_P3D_bench,
    savename=savename,
    fontsize=20,
)
# savename = folder + "test_cosmo/test_cosmo_P3D_fit.png"
# plot_p3d_test_sims(
#     sim_labels,
#     k_Mpc,
#     mu,    
#     k_mask,
#     fractional_error_P3D_bench,
#     savename=savename,
#     fontsize=20,
# )

# %%

# %% [markdown]
# ## CHECK RUNNING SIMULATION

# %%
index = sim_labels.index('mpg_running')

# %%
for iz,z in enumerate(z_grid):
    plt.plot(p1d_k, fractional_error_P1D_sims[index, iz], label = f'z={z_grid[iz]}')
    
plt.legend(fontsize=10)

plt.axhspan(-0.01, 0.01, color="gray", alpha=0.3)
plt.ylim(-0.02, 0.07)

plt.ylabel(r"Error $P_{\rm 1D}$ [%]",  fontsize=16)
    
plt.xlabel('$k$ [1/Mpc]', fontsize=14 )   

# %%
mu_mask = (mu >= 0.31) & (mu <= 0.38)
k_masked = k_Mpc[mu_mask]
for iz,z in enumerate(z_grid):
    plt.plot(k_masked, fractional_error_P3D_sims[index, iz][mu_mask], label = f'z={z_grid[iz]}')
    
plt.legend(fontsize=10)

plt.axhspan(-0.10, 0.10, color="gray", alpha=0.3)
plt.ylim(-0.15, 0.15)

plt.ylabel(r"Error $P_{\rm 3D}$ [%]",  fontsize=16)
    
plt.xlabel('$k$ [1/Mpc]', fontsize=14 )   

# %% [markdown]
# ### Look at Arinyo parameters

# %%
sim_label ='mpg_running'
arinyo_emu = np.zeros(shape=(len(z_grid),8))
arinyo_mcmc = np.zeros(shape=(len(z_grid),8))
test_sim = Archive3D.get_testing_data(sim_label, force_recompute_plin=True)
                       
for iz, z in enumerate(z_grid):

    test_sim_z = [d for d in test_sim if d["z"] == z]
    
    arinyo_emu[iz] = p3d_emu.predict_Arinyos(test_sim=test_sim_z)
                         
    arinyo_mcmc[iz] = np.fromiter(test_sim_z[0]["Arinyo"].values(), dtype=float) 
    



# %% [markdown]
# #### plot values

# %%
params =  [r"$b$", r"$\beta$", "$q_1$", "$k_{vav}$", "$a_v$", "$b_v$", "$k_p$", "$q_2$"]
colors = plt.cm.tab10(np.linspace(0, 1, len(params)))


for ip, param in enumerate(params):
    plt.scatter(np.arange(8), arinyo_emu[ip, :], marker='_', label='ForestFlow' if ip == 0 else "", color = colors[ip])
    plt.scatter(np.arange(0.3, 8.3, 1), arinyo_mcmc[ip, :], marker='x',  label='MCMC' if ip == 0 else "", color = colors[ip])

plt.xticks(np.arange(8), params)
plt.legend(fontsize=14)
plt.ylabel('Parameter value', fontsize=16)
plt.show()


# %% [markdown]
# #### plot ratios

# %%
sim_label ='mpg_central'
arinyo_emu_central = np.zeros(shape=(len(z_grid),8))
arinyo_mcmc_central = np.zeros(shape=(len(z_grid),8))
test_sim = Archive3D.get_testing_data(sim_label, force_recompute_plin=True)
                       
for iz, z in enumerate(z_grid):
    test_sim_z = [d for d in test_sim if d["z"] == z]
    arinyo_emu_central[iz] = p3d_emu.predict_Arinyos(test_sim=test_sim_z)                
    arinyo_mcmc_central[iz] = np.fromiter(test_sim_z[0]["Arinyo"].values(), dtype=float) 
ratio_central = arinyo_emu_central / arinyo_mcmc_central


# %%
sim_label ='mpg_running'
arinyo_emu_running = np.zeros(shape=(len(z_grid),8))
arinyo_mcmc_running = np.zeros(shape=(len(z_grid),8))
test_sim = Archive3D.get_testing_data(sim_label, force_recompute_plin=True)
                       
for iz, z in enumerate(z_grid):
    test_sim_z = [d for d in test_sim if d["z"] == z]
    arinyo_emu_running[iz] = p3d_emu.predict_Arinyos(test_sim=test_sim_z)                
    arinyo_mcmc_running[iz] = np.fromiter(test_sim_z[0]["Arinyo"].values(), dtype=float) 
ratio_running = arinyo_emu_running / arinyo_mcmc_running


# %%
params =  [r"$b$", r"$\beta$", "$q_1$", "$k_{vav}$", "$a_v$", "$b_v$", "$k_p$", "$q_2$"]
colors = plt.cm.tab10(np.linspace(0, 1, len(params)))


for ip, param in enumerate(params):
    plt.scatter(np.arange(8), ratio_central[ip, :], marker='_', label='Central' if ip == 0 else "", color = colors[ip])
    plt.scatter(np.arange(0.3, 8.3, 1), ratio_running[ip, :], marker='x',  label='Running' if ip == 0 else "", color = colors[ip])

plt.xticks(np.arange(8), params)
plt.legend(fontsize=14)
plt.ylabel('Emulated / MCMC  parameters', fontsize=16)

plt.ylim(0,2)
plt.show()

# %% [markdown]
# ### Check running, move below

# %%
snap = 5
index = [0, -2]
for ii, sim_label in enumerate(["mpg_central", "mpg_running"]):
    # plt.plot(k_p1d_Mpc, P1D_testsims_true[index[ii], snap]/P1D_testsims_true[index[0], snap], ".")
    plt.plot(k_p1d_Mpc_all, P1D_testsims_true_all[index[ii], snap]/P1D_testsims_true_all[index[0], snap], ".:", label=sim_label+" sim")
    # plt.plot(k_p1d_Mpc, P1D_testsims_Arinyo[index[ii], snap]/P1D_testsims_true[index[0], snap]/norm)
    plt.plot(k_p1d_Mpc_all, P1D_testsims_Arinyo_all[index[ii], snap]/P1D_testsims_true_all[index[0], snap], "--", label=sim_label+" Arinyo")
plt.xscale('log')
# plt.yscale('log')
plt.xlim(0.1, 10)
plt.ylim(0.6, 1.1)
plt.legend()
plt.savefig("test_arinyo_10.png")

# %%

# %%
params =  [r"$b$", r"$\beta$", "$q_1$", "$k_{vav}$", "$a_v$", "$b_v$", "$k_p$", "$q_2$"]
print(arinyo_testsims[0, 5])
print(arinyo_testsims[-2, 5])
index = [0, -2]
snap = 5
z = z_grid[snap]
model_Arinyo = []
div = P1D_testsims_true[index[0], snap]
for ii, sim_label in enumerate(["mpg_central", "mpg_running"]):
    # Find the index of the underscore
    underscore_index = sim_label.find("_")
    lab = sim_label[underscore_index + 1 :]

    # test_sim = Archive3D.get_testing_data(sim_label, force_recompute_plin=False)
    # z_grid = [d["z"] for d in test_sim]

    # load arinyo module
    # lab = "central"
    flag = f"Plin_interp_sim{lab}.npy"
    file_plin_inter = folder_interp + flag
    pk_interp = np.load(file_plin_inter, allow_pickle=True).all()
    model_Arinyo.append(ArinyoModel(camb_pk_interp=pk_interp))
    
for ii, sim_label in enumerate(["mpg_central", "mpg_running"]):
    col = "C"+str(ii)
    BF_arinyo = params_numpy2dict(arinyo_testsims[index[ii], snap])
    # print(sim_label)
    # BF_arinyo = test_sim_z[0]["Arinyo_minin"]

    # test_p3d_arinyo = model_Arinyo.P3D_Mpc(z, k_Mpc, mu, BF_arinyo)        
    test_p1d_arinyo = model_Arinyo[ii].P1D_Mpc(z, k_p1d_Mpc, parameters=BF_arinyo) * norm

    plt.plot(k_p1d_Mpc, test_p1d_arinyo/div, col, label=sim_label+" emu")
    plt.plot(k_p1d_Mpc, P1D_testsims_true[index[ii], snap]/div, col+ ".:", label=sim_label+" sim")

test_p1d_arinyo = model_Arinyo[0].P1D_Mpc(z, k_p1d_Mpc, parameters=BF_arinyo) * norm

plt.plot(k_p1d_Mpc, test_p1d_arinyo/div, "C2--", label=sim_label+" Plin central")
BF_arinyo = params_numpy2dict(arinyo_testsims[index[0], snap])
test_p1d_arinyo = model_Arinyo[1].P1D_Mpc(z, k_p1d_Mpc, parameters=BF_arinyo) * norm
plt.plot(k_p1d_Mpc, test_p1d_arinyo/div, "C3-.", label=sim_label+" param central")
plt.legend()
plt.savefig("test_running.png")


# %%

# %%

# %%

# %%
from lace.setup_simulations import read_genic
from lace.cosmo import camb_cosmo, fit_linP
kp_Mpc = 0.7
z = z_grid[snap]

# %%
pair_dir = '/home/jchaves/Proyectos/projects/lya/LaCE/data/sim_suites/Australia20/running_sim/'

genic_fname = pair_dir + "/sim_plus/paramfile.genic"
cosmo_params = read_genic.camb_from_genic(genic_fname)

# setup CAMB object
sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(cosmo_params)

# compute linear power parameters at each z (in Mpc units)
linP_zs = fit_linP.get_linP_Mpc_zs(sim_cosmo, [z], kp_Mpc)

# %%
linP_zs_central = linP_zs
linP_zs_central

# %%
linP_zs_running = linP_zs
linP_zs_running

# %%
test_central[5]['Delta2_p']

# %%
test_running[5]['Delta2_p']

# %%
test_running[5]['Plin_for_p1d'].shape

# %%
test_central[5].keys()

# %%
plt.plot(test_central[5]['k_Mpc'], test_central[5]['p1d_Mpc']/test_running[5]['p1d_Mpc'])
plt.xlim(0.1, 4)
plt.xscale("log")
plt.ylim(0.99, 1.01)

# %%
test_central[5]['Plin']

# %%
plt.plot(test_central[5]['k3d_Mpc'][:,0], 
         (test_central[5]['Plin']/test_running[5]['Plin'])[:,0])
# plt.xlim(0.1, 10)
plt.xscale("log")
# plt.ylim(0.96, 1.02)
plt.savefig("plin_running.png")

# %%
test_running[5]['Plin_for_p1d']

# %%
test_central = Archive3D.get_testing_data(
    "mpg_central", force_recompute_plin=False
)
test_running = Archive3D.get_testing_data(
    "mpg_running", force_recompute_plin=False
)

# %%

# %%
