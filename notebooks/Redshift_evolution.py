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
from matplotlib import gridspec
import numpy as np

# %%
from ForestFlow.archive import GadgetArchive3D
from ForestFlow.plots_v0 import plot_test_p3d
from ForestFlow.P3D_cINN import P3DEmulator
from ForestFlow.model_p3d_arinyo import ArinyoModel
from ForestFlow import model_p3d_arinyo


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
folder_lya_data = path_program +  "/data/best_arinyo/"
#folder_interp = path_program+"/data/plin_interp/"

Archive3D = GadgetArchive3D(
    base_folder=path_program[:-1], 
    folder_data=folder_lya_data, 
    force_recompute_plin=True,
    average='both'
)
print(len(Archive3D.training_data))


# %% [markdown]
# ## LOAD EMULATOR

# %%
p3d_emu = P3DEmulator(
    Archive3D.training_data,
    Archive3D.emu_params,
    nepochs=300,
    lr=0.001,#0.005
    batch_size=20,
    step_size=200,
    gamma=0.1,
    weight_decay=0,
    adamw=True,
    nLayers_inn=12,#15
    Archive=Archive3D,
    use_chains=False,
    chain_samp=100_000,
    Nrealizations=1000,
    input_space='Arinyo',
    folder_chains='/data/desi/scratch/jchavesm/p3d_fits_new/',
    model_path='../data/emulator_models/mpg_hypercube.pt'
)

# %% [markdown]
# ## LOAD CENTRAL SIMULATION

# %%
sim_label = "mpg_central"

# %%
test_sim = central = Archive3D.get_testing_data(
        sim_label, 
        force_recompute_plin=True
        )


# %%
Arinyo_coeffs_central =  np.array( [list(test_sim[i]["Arinyo"].values()) for i in range(len(test_sim))] )


# %% [markdown]
# ## LOOP OVER REDSHIFTS PREDICTING THE ARINYO PARAMETERS

# %%
z_central = [d['z'] for d in test_sim]

# %%
Arinyo_coeffs_central_emulator = np.zeros_like(Arinyo_coeffs_central)
for iz,z in enumerate(z_central):
    
    test_sim_z = [d for d in test_sim if d['z']==z]
    
    
    testing_condition = p3d_emu._get_test_condition(test_sim_z)
    Arinyo_mean = p3d_emu.predict_Arinyos(testing_condition, 
                                                true_coeffs=None, 
                                                plot=False,                                                 
                                                return_all_realizations=False)
    
    Arinyo_coeffs_central_emulator[iz] = Arinyo_mean


# %% [markdown]
# ## PLOT

# %% jupyter={"outputs_hidden": true}
test_sim[0]

# %%
colors = ['navy', 'crimson', 'forestgreen', 'purple', 'darkorange', 'deepskyblue', 'grey', 'pink']
labels = [r'$b$', r'$\beta$', r'$q_1$', r'$k_{\rm vav}$', r'$a_{v}$', r'$b_{v}$', r'$k_p$', r'$q_2$']

# Create a 2x1 grid for plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]})

# Plot the original and emulator data in the upper panel
for i in range(8):
    if i!=6:
        ax1.plot(z_central, np.abs(Arinyo_coeffs_central[:, i]), color=colors[i], ls='-', label=labels[i])
        ax1.plot(z_central, np.abs(Arinyo_coeffs_central_emulator[:, i]), color=colors[i], ls='--')

ax1.set_ylabel('Arinyo Parameter', fontsize=16)
ax1.legend(fontsize=14)

# Calculate relative difference
relative_differences = [
    np.abs(Arinyo_coeffs_central[:, i]) / np.abs(Arinyo_coeffs_central_emulator[:, i]) - 1 for i in range(8)
]

# Plot relative difference in the lower panel
for i in range(8):
    ax2.plot(z_central, relative_differences[i], color=colors[i], ls='-')

ax2.set_xlabel('$z$', fontsize=16)
ax2.set_ylabel('Relative Difference', fontsize=16)
#ax2.legend()

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# %%
