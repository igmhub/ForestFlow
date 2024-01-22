# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: emulators
#     language: python
#     name: emulators
# ---

# %%
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"
import matplotlib.cm as cm

# %%
from forestflow.archive import GadgetArchive3D
from forestflow.plots_v0 import plot_test_p3d
from forestflow.P3D_cINN import P3DEmulator


# %%
def ls_level(folder, nlevels):
    for ii in range(nlevels):
        folder = os.path.dirname(folder)
    folder += "/"
    return folder

path_program = ls_level(os.getcwd(), 1)
print(path_program)
sys.path.append(path_program)

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
k_Mpc = Archive3D.training_data[0]["k3d_Mpc"]
mu = Archive3D.training_data[0]["mu3d"]

k_mask = (k_Mpc < 4) & (k_Mpc > 0)

k_Mpc = k_Mpc[k_mask]
mu = mu[k_mask]

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
    model_path="../data/emulator_models/mpg_hypercube.pt",
)

# %% [markdown]
# ## CONTRIBUTION TO P3D ON CENTRAL AT z=3

# %%
sim_label = "mpg_central"
z_test = 3



# %%
test_sim = Archive3D.get_testing_data(
    sim_label, force_recompute_plin=False
)

# %%
# define test sim
dict_sim = [d for d in test_sim if d["z"] == z_test and d["val_scaling"] == 1]

# %%
p1d, k1d = p3d_emu.get_p1d_sim(dict_sim)

# %% jupyter={"outputs_hidden": true}
Npoints = 200
params_list = ['bias', 'beta', 'q1', 'kvav', 'av', 'bv', 'kp', 'q2']
p1ds = np.zeros(shape=(len(params_list), Npoints, 53))
for ip, param in enumerate(params_list):
    arinyo_mcmc = dict_sim[0]["Arinyo"]
    
    arinyo_ip = arinyo_mcmc.copy()
    for iq,q in enumerate(np.linspace(-0.1,0.1,Npoints)):
        arinyo_ip[param] = arinyo_mcmc[param] + arinyo_mcmc[param]*q
        
        p1ds[ip, iq] = p3d_emu.predict_P1D_Mpc(sim_label=sim_label,
                                     z=z_test,
                                     test_sim=dict_sim,
                                     test_arinyo=np.fromiter(arinyo_ip.values(), dtype=float).reshape(1,8),
                                     return_cov=False)
        
p1d_mcmc = p3d_emu.predict_P1D_Mpc(sim_label=sim_label,
                                     z=z_test,
                                     test_sim=dict_sim,
                                     test_arinyo=np.fromiter(arinyo_mcmc.values(), dtype=float).reshape(1,8),
                                     return_cov=False)

variatoin_p1d = p1ds / p1d_mcmc[None, None, :]

# %% jupyter={"outputs_hidden": true}
Npoints = 200
params_list = ['bias', 'beta', 'q1', 'kvav', 'av', 'bv', 'kp', 'q2']
p3ds = np.zeros(shape=(len(params_list), Npoints, 148))
for ip, param in enumerate(params_list):
    arinyo_mcmc = dict_sim[0]["Arinyo"]
    
    arinyo_ip = arinyo_mcmc.copy()
    for iq,q in enumerate(np.linspace(-0.1,0.1,Npoints)):
        arinyo_ip[param] = arinyo_mcmc[param] + arinyo_mcmc[param]*q
        
        p3ds[ip, iq] = p3d_emu.predict_P3D_Mpc(sim_label=sim_label,
                                     z=z_test,
                                     test_sim=dict_sim,
                                     test_arinyo=np.fromiter(arinyo_ip.values(), dtype=float).reshape(1,8),
                                     return_cov=False)
        
p3d_mcmc = p3d_emu.predict_P3D_Mpc(sim_label=sim_label,
                                     z=z_test,
                                     test_sim=dict_sim,
                                     test_arinyo=np.fromiter(arinyo_mcmc.values(), dtype=float).reshape(1,8),
                                     return_cov=False)

variatoin_p3d = p3ds / p3d_mcmc[None, None, :]

# %%
arinyo_mcmc_plot = np.array(list(arinyo_mcmc.values())).copy()
# Set up the figure and axis
fig, axes = plt.subplots(4, 2, figsize=(12, 9), sharex=True, sharey=False)
for ii, ax in enumerate(axes.flat):
    for j,q in enumerate(np.linspace(0.9,1.11,Npoints)):
        colors = np.linspace(arinyo_mcmc_plot[ii]-arinyo_mcmc_plot[ii]*0.1, arinyo_mcmc_plot[ii]+arinyo_mcmc_plot[ii]*0.1, 200)
        norm = plt.Normalize(vmin=np.min(colors), vmax=np.max(colors))
        color_val = colors[j]
        color_map = cm.ScalarMappable(norm=norm, cmap='RdYlBu') # Create a colormap instance
        color = color_map.to_rgba(color_val)  # Map the value to a color from the RdYlBu colormap

        ax.plot(k1d , variatoin_p1d[ii,j], color=color)
        
    ax.set_xscale('log')
    y_ticks = np.round(np.linspace(np.min(variatoin_p1d[ii,:]), np.max(variatoin_p1d[ii,:]), 5),3)
    ax.set_yticks(y_ticks)
    
# Add a colorbar to show the color mapping
    cbar = plt.colorbar(color_map, ax=ax, ticks=colors)

    #cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='RdYlBu'), ax=ax)
    cbar.set_label(params_list[ii], fontsize = 14)  # Replace 'Colorbar Label' with an appropriate label
    
    # Set the tick locations and labels of the colorbar to show the real parameter values
    cbar.set_ticks(np.round(np.linspace(arinyo_mcmc_plot[ii]-arinyo_mcmc_plot[ii]*0.1, arinyo_mcmc_plot[ii]+arinyo_mcmc_plot[ii]*0.1,5),2))  # Set the tick locations to the minimum and maximum parameter values
    cbar.set_ticklabels(np.round(np.linspace(arinyo_mcmc_plot[ii]-arinyo_mcmc_plot[ii]*0.1, arinyo_mcmc_plot[ii]+arinyo_mcmc_plot[ii]*0.1,5),2))  # Set the tick labels to the minimum and maximum parameter values


    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # Add labels, title, etc. if desired
    fig.text(0.5, 0.05, r'$k$ [1/Mpc]', ha='center', fontsize=16)
    fig.text(0.05, 0.5, r'Variation in the $P_{\rm 1D}$', va='center', rotation='vertical', fontsize=16)

# Show the plot
#plt.savefig('smoothness_p1d.pdf',bbox_inches='tight')
plt.show()


# %%
arinyo_mcmc_plot = np.array(list(arinyo_mcmc.values())).copy()
# Set up the figure and axis
fig, axes = plt.subplots(4, 2, figsize=(12, 9), sharex=True, sharey=False)
for ii, ax in enumerate(axes.flat):
    for j,q in enumerate(np.linspace(0.9,1.11,Npoints)):
        colors = np.linspace(arinyo_mcmc_plot[ii]-arinyo_mcmc_plot[ii]*0.1, arinyo_mcmc_plot[ii]+arinyo_mcmc_plot[ii]*0.1, 200)
        norm = plt.Normalize(vmin=np.min(colors), vmax=np.max(colors))
        color_val = colors[j]
        color_map = cm.ScalarMappable(norm=norm, cmap='RdYlBu') # Create a colormap instance
        color = color_map.to_rgba(color_val)  # Map the value to a color from the RdYlBu colormap

        ax.plot(k1d , variatoin_p1d[ii,j], color=color)
        
    ax.set_xscale('log')
    y_ticks = np.round(np.linspace(np.min(variatoin_p1d[ii,:]), np.max(variatoin_p1d[ii,:]), 5),3)
    ax.set_yticks(y_ticks)
    
# Add a colorbar to show the color mapping
    cbar = plt.colorbar(color_map, ax=ax, ticks=colors)

    #cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='RdYlBu'), ax=ax)
    cbar.set_label(params_list[ii], fontsize = 14)  # Replace 'Colorbar Label' with an appropriate label
    
    # Set the tick locations and labels of the colorbar to show the real parameter values
    cbar.set_ticks(np.round(np.linspace(arinyo_mcmc_plot[ii]-arinyo_mcmc_plot[ii]*0.1, arinyo_mcmc_plot[ii]+arinyo_mcmc_plot[ii]*0.1,5),2))  # Set the tick locations to the minimum and maximum parameter values
    cbar.set_ticklabels(np.round(np.linspace(arinyo_mcmc_plot[ii]-arinyo_mcmc_plot[ii]*0.1, arinyo_mcmc_plot[ii]+arinyo_mcmc_plot[ii]*0.1,5),2))  # Set the tick labels to the minimum and maximum parameter values


    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # Add labels, title, etc. if desired
    fig.text(0.5, 0.05, r'$k$ [1/Mpc]', ha='center', fontsize=16)
    fig.text(0.05, 0.5, r'Variation in the $P_{\rm 1D}$', va='center', rotation='vertical', fontsize=16)

# Show the plot
#plt.savefig('smoothness_p1d.pdf',bbox_inches='tight')
plt.show()


# %%
k_Mpc = Archive3D.training_data[0]["k3d_Mpc"]
mu = Archive3D.training_data[0]["mu3d"]

k_mask = (k_Mpc < 4) & (k_Mpc > 0)

k_Mpc = k_Mpc[k_mask]
mu = mu[k_mask]


mu_min, mu_max = 0.31,0.38
mu_mask = (mu>mu_min) &( mu<mu_max)
k_p3d = k_Mpc[mu_mask]


# %%
arinyo_mcmc_plot = np.array(list(arinyo_mcmc.values())).copy()
# Set up the figure and axis
fig, axes = plt.subplots(4, 2, figsize=(12, 9), sharex=True, sharey=False)
for ii, ax in enumerate(axes.flat):
    for j,q in enumerate(np.linspace(0.9,1.11,Npoints)):
        colors = np.linspace(arinyo_mcmc_plot[ii]-arinyo_mcmc_plot[ii]*0.1, arinyo_mcmc_plot[ii]+arinyo_mcmc_plot[ii]*0.1, 200)
        norm = plt.Normalize(vmin=np.min(colors), vmax=np.max(colors))
        color_val = colors[j]
        color_map = cm.ScalarMappable(norm=norm, cmap='RdYlBu') # Create a colormap instance
        color = color_map.to_rgba(color_val)  # Map the value to a color from the RdYlBu colormap

        ax.plot(k_p3d , variatoin_p3d[ii,j][mu_mask], color=color)
        
    ax.set_xscale('log')
    y_ticks = np.round(np.linspace(np.min(variatoin_p3d[ii,:][:,mu_mask]), np.max(variatoin_p3d[ii,:][:,mu_mask]), 5),3)
    ax.set_yticks(y_ticks)
    
# Add a colorbar to show the color mapping
    cbar = plt.colorbar(color_map, ax=ax, ticks=colors)

    #cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='RdYlBu'), ax=ax)
    cbar.set_label(params_list[ii], fontsize = 14)  # Replace 'Colorbar Label' with an appropriate label
    
    # Set the tick locations and labels of the colorbar to show the real parameter values
    cbar.set_ticks(np.round(np.linspace(arinyo_mcmc_plot[ii]-arinyo_mcmc_plot[ii]*0.1, arinyo_mcmc_plot[ii]+arinyo_mcmc_plot[ii]*0.1,5),2))  # Set the tick locations to the minimum and maximum parameter values
    cbar.set_ticklabels(np.round(np.linspace(arinyo_mcmc_plot[ii]-arinyo_mcmc_plot[ii]*0.1, arinyo_mcmc_plot[ii]+arinyo_mcmc_plot[ii]*0.1,5),2))  # Set the tick labels to the minimum and maximum parameter values


    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # Add labels, title, etc. if desired
    fig.text(0.5, 0.05, r'$k$ [1/Mpc]', ha='center', fontsize=16)
    fig.text(0.05, 0.5, r'Variation in the $P_{\rm 3D}$', va='center', rotation='vertical', fontsize=16)

# Show the plot
#plt.savefig('smoothness_p1d.pdf',bbox_inches='tight')
plt.show()


# %%
