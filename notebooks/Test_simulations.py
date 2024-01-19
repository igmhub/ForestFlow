# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: ForestFlow
#     language: python
#     name: forestflow
# ---

# %% [markdown]
# # NOTEBOOK PRODUCING FIGURE X, Y P3D PAPER
#

# %%
import sys
import os
import matplotlib.pyplot as plt
import matplotlib

font = {"size": 22}
matplotlib.rc("font", **font)
plt.rc("text", usetex=False)
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rc("xtick", labelsize=15)
matplotlib.rc("ytick", labelsize=15)

import numpy as np

# %%
from forestflow.archive import GadgetArchive3D
from forestflow.P3D_cINN import P3DEmulator
from forestflow.model_p3d_arinyo import ArinyoModel
from forestflow.utils import sigma68


# %%
def ls_level(folder, nlevels):
    for ii in range(nlevels):
        folder = os.path.dirname(folder)
    folder += "/"
    return folder


path_program = ls_level(os.getcwd(), 1)
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
    force_recompute_plin=True,
    average="both",
)
print(len(Archive3D.training_data))


# %% [markdown]
# ## TRAIN EMULATOR

# %%
p3d_emu = P3DEmulator(
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
    Nrealizations=1000,
    model_path="../data/emulator_models/mpg_hypercube.pt",
)

# %% [markdown]
# ## TEST EMULATOR TEST SIMULATIONS

# %%
# Extract data from Archive3D
k_Mpc = Archive3D.training_data[0]["k3d_Mpc"]
mu = Archive3D.training_data[0]["mu3d"]

# Apply a mask to select relevant k values
k_mask = (k_Mpc < 4) & (k_Mpc > 0)
k_Mpc = k_Mpc[k_mask]
mu = mu[k_mask]


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

# %% [markdown]
# #### Loop over test sims, emulator prediction

# %%
P3D_testsims = np.zeros((len(sim_labels), 11, 148))
P1D_testsims = np.zeros((len(sim_labels), 11, 53))

for ii, sim_label in enumerate(sim_labels):
    test_sim = central = Archive3D.get_testing_data(
        sim_label, force_recompute_plin=True
    )
    z_grid = [d["z"] for d in test_sim]

    for iz, z in enumerate(z_grid):
        
        test_sim_z = [d for d in test_sim if d["z"] == z]
        p3d_arinyo_mean = p3d_emu.predict_P3D_Mpc(
            sim_label=sim_label, z=z, test_sim=test_sim_z, return_cov=False
        )
        p1d_arinyo_mean = p3d_emu.predict_P1D_Mpc(
            sim_label=sim_label, z=z, test_sim=test_sim_z, return_cov=False
        )

        P3D_testsims[ii, iz] = p3d_arinyo_mean
        P1D_testsims[ii, iz] = p1d_arinyo_mean

# %% [markdown]
# #### Loop over test sims, true P1D and P3D

# %%
P3D_testsims_true = np.zeros((len(sim_labels), 11, 148))
P1D_testsims_true = np.zeros((len(sim_labels), 11, 53))

for ii, sim_label in enumerate(sim_labels):
    test_sim =  Archive3D.get_testing_data(
        sim_label, force_recompute_plin=True
    )
    z_grid = [d["z"] for d in test_sim]

    for iz, z in enumerate(z_grid):
        test_sim_z = [d for d in test_sim if d["z"] == z]
        
        p1d_sim, p1d_k = p3d_emu.get_p1d_sim(test_sim_z)
        p3d_sim = [d for d in test_sim if d['z'] == z][0]['p3d_Mpc'][p3d_emu.k_mask]

        p3d_sim = test_sim_z[0]["p3d_Mpc"][p3d_emu.k_mask]
        p3d_sim = np.array(p3d_sim)

        P3D_testsims_true[ii, iz] = p3d_sim
        P1D_testsims_true[ii, iz] = p1d_sim


# %% [markdown]
# #### Loop over test sims, P1D and P3D from MCMC Arinyo

# %% jupyter={"outputs_hidden": true}
P3D_testsims_Arinyo = np.zeros((len(sim_labels), 11, 148))
P1D_testsims_Arinyo = np.zeros((len(sim_labels), 11, 53))

for ii, sim_label in enumerate(sim_labels):
    # Find the index of the underscore
    underscore_index = sim_label.find("_")
    lab = sim_label[underscore_index + 1 :]

    test_sim = Archive3D.get_testing_data(sim_label, force_recompute_plin=True)
    z_grid = [d["z"] for d in test_sim]

    for iz, z in enumerate(z_grid):
        test_sim_z = [d for d in test_sim if d["z"] == z]

        # load arinyo module
        flag = f"Plin_interp_sim{lab}.npy"
        file_plin_inter = folder_interp + flag
        pk_interp = np.load(file_plin_inter, allow_pickle=True).all()
        model_Arinyo = ArinyoModel(camb_pk_interp=pk_interp)

        BF_arinyo = test_sim_z[0]["Arinyo"]
        p3d_arinyo = model_Arinyo.P3D_Mpc(z, k_Mpc, mu, BF_arinyo)
        
        p1d_arinyo =  p3d_emu.predict_P1D_Mpc(sim_label=sim_label, 
                                              z=z, 
                                              test_sim=test_sim_z, 
                                              test_arinyo=np.fromiter(BF_arinyo.values(), dtype='float').reshape(1,8),
                                              return_cov=False)

        P3D_testsims_Arinyo[ii, iz] = p3d_arinyo
        P1D_testsims_Arinyo[ii, iz] = p1d_arinyo#[k1d_mask]

# %% [markdown]
# ### Define fractional errors

# %%
# here we can change P3D_testsims_true by P3D_testsims_Arinyo (and same for P1D)

# %%
fractional_error_P3D = (P3D_testsims / P3D_testsims_Arinyo - 1) * 100
fractional_error_P1D = (P1D_testsims / P1D_testsims_Arinyo - 1) * 100

# %%

# %% [markdown]
# ## PLOT P1D

# %%
colors = ["crimson"]
fig, ax = plt.subplots(
    ncols=1,
    nrows=len(sim_labels),
    figsize=(10, 2 * len(sim_labels)),
    sharey=True,
    sharex=True,
)

for c in range(len(sim_labels)):
    ax[c].plot(
        z_grid,
        np.nanmedian(fractional_error_P1D[c], 1),
        ls="--",
        marker="o",
        markersize=2,
        color=colors[0],
    )
    ax[c].fill_between(
        z_grid,
        np.nanmedian(fractional_error_P1D[c], 1)
        - sigma68(fractional_error_P1D[c].flatten()),
        np.nanmedian(fractional_error_P1D[c], 1)
        + sigma68(fractional_error_P1D[c].flatten()),
        color=colors[0],
        alpha=0.4,
    )

    ax[c].axhspan(-1, 1, color="gray", alpha=0.3)
    ax[c].axhline(y=0, color="black", ls="--", alpha=0.4)


ax[0].set_ylim(-5, 5)

# plt.ylabel(r'Percent error P1D', fontsize = 14)

fig.text(0.04, 0.5, r"Error $P_{\rm 1D}$ [%]", va="center", rotation="vertical")
fig.text(0.8, 0.86, r"Central", fontsize=15)
fig.text(0.8, 0.75, r"Seed", fontsize=15)
fig.text(0.8, 0.64, r"Growth", fontsize=15)
fig.text(0.8, 0.52, r"Neutrinos", fontsize=15)
fig.text(0.8, 0.41, r"Curved", fontsize=15)
fig.text(0.8, 0.3, r"Running", fontsize=15)
fig.text(0.78, 0.19, r"Reionization", fontsize=15)
plt.xlabel(r"$z$")


# plt.savefig('other_cosmologies.pdf', bbox_inches = 'tight')

# %% [markdown]
# ## PLOT P3D

# %%
# Define mu bins
mu_lims = [[0, 0.06], [0.31, 0.38], [0.62, 0.69], [0.94, 1]]

# Define colors for different mu bins
colors = ["navy", "crimson", "forestgreen", "goldenrod"]

fig, ax = plt.subplots(
    ncols=1,
    nrows=len(sim_labels),
    figsize=(10, 2 * len(sim_labels)),
    sharey=True,
    sharex=True,
)

for c in range(len(sim_labels)):
    ax[c].axhspan(-1, 1, color="gray", alpha=0.3)
    ax[c].axhline(y=0, color="black", ls="--", alpha=0.4)
    # Loop through mu bins
    for mi in range(int(len(mu_lims))):
        mu_mask = (mu >= mu_lims[mi][0]) & (mu <= mu_lims[mi][1])
        k_masked = k_Mpc[mu_mask]

        # Calculate fractional error statistics
        frac_err = np.nanmedian(fractional_error_P3D[c, :, :], 0)
        frac_err_err = sigma68(fractional_error_P3D[c, :, :])

        frac_err_masked = frac_err[mu_mask]
        frac_err_err_masked = frac_err_err[mu_mask]

        ax[c].plot(
            k_masked,
            frac_err_masked,
            ls="--",
            marker="o",
            markersize=2,
            color=colors[mi],
            label=f"${mu_lims[mi][0]}\leq \mu \leq {mu_lims[mi][1]}$",
        )
        ax[c].fill_between(
            k_masked,
            frac_err_masked - frac_err_err_masked,
            frac_err_masked + frac_err_err_masked,
            color=colors[mi],
            alpha=0.4,
        )


ax[0].set_ylim(-15, 15)
ax[0].legend(fontsize=10)


# plt.ylabel(r'Percent error P1D', fontsize = 14)

fig.text(0.04, 0.5, r"Error $P_{\rm 3D}$ [%]", va="center", rotation="vertical")
fig.text(0.67, 0.86, r"Central", fontsize=15)
fig.text(0.8, 0.75, r"Seed", fontsize=15)
fig.text(0.8, 0.64, r"Growth", fontsize=15)
fig.text(0.8, 0.52, r"Neutrinos", fontsize=15)
fig.text(0.8, 0.41, r"Curved", fontsize=15)
fig.text(0.8, 0.3, r"Running", fontsize=15)
fig.text(0.78, 0.19, r"Reionization", fontsize=15)
plt.xlabel(r"$k$ [1/Mpc]")


# plt.savefig('other_cosmologies.pdf', bbox_inches = 'tight')

# %% [markdown]
# ## CHECK RUNNING SIMULATION

# %%
index = sim_labels.index('mpg_running')

# %%
for iz,z in enumerate(z_grid):
    plt.plot(p1d_k, fractional_error_P1D[index, iz], label = f'z={z_grid[iz]}')
    
plt.legend(fontsize=10)

plt.axhspan(-1, 1, color="gray", alpha=0.3)
plt.ylim(-2, 7)

plt.ylabel(r"Error $P_{\rm 1D}$ [%]",  fontsize=16)
    
plt.xlabel('$k$ [1/Mpc]', fontsize=14 )   

# %%
mu_mask = (mu >= 0.31) & (mu <= 0.38)
k_masked = k_Mpc[mu_mask]
for iz,z in enumerate(z_grid):
    plt.plot(k_masked, fractional_error_P3D[index, iz][mu_mask], label = f'z={z_grid[iz]}')
    
plt.legend(fontsize=10)

plt.axhspan(-10, 10, color="gray", alpha=0.3)
plt.ylim(-15, 15)

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


# %%
