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
#     display_name: emulators2
#     language: python
#     name: emulators2
# ---

# %% [markdown]
# # NOTEBOOK PRODUCING FIGURE X, Y P3D PAPER
#

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# %%
from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

# %%
from forestflow.archive import GadgetArchive3D
from forestflow.plots_v0 import plot_err_uncertainty
from forestflow.P3D_cINN import P3DEmulator
from forestflow.model_p3d_arinyo import ArinyoModel

#from forestflow.model_p3d_arinyo import ArinyoModel
#from forestflow import model_p3d_arinyo
#from forestflow.likelihood import Likelihood

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
folder_interp = path_program+"/data/plin_interp/"
folder_chains='/pscratch/sd/l/lcabayol/P3D/p3d_fits_new/'

Archive3D = GadgetArchive3D(
    base_folder=path_program[:-1], 
    folder_data=folder_lya_data, 
    force_recompute_plin=False,
    average='both'
)
print(len(Archive3D.training_data))


# %% [markdown]
# ## TRAIN EMULATOR

# %%
p3d_emu = P3DEmulator(
    Archive3D.training_data,
    Archive3D.emu_params,
    nepochs=250,#300,
    lr=0.001,#0.005
    batch_size=20,#20,
    step_size=200,
    gamma=0.1,
    weight_decay=0,
    adamw=True,
    nLayers_inn=12,#15
    Archive=Archive3D,
    Nrealizations=5000,
    model_path='../data/emulator_models/mpg_hypercube.pt'
)

# %% [markdown]
# ## PLOT COVARIANCE FOR A FIXED MU

# %%
sim_label = 'mpg_central'
z_test = 3.0
val_scaling=1.0


# %%
test_sim =  Archive3D.get_testing_data(
        sim_label, 
        force_recompute_plin=False
        )
test_sim = [d for d in test_sim if d['z']==z_test and d['val_scaling'] ==val_scaling]
k_Mpc =test_sim[0]["k_Mpc"]
k_mask = (k_Mpc < 4) & (k_Mpc > 0)
k_Mpc = k_Mpc[k_mask]

test_params = {param: test_sim[0][param] for param in p3d_emu.emuparams}



# %%
mask = (test_sim[0]['k_Mpc'] > 0) & (test_sim[0]['k_Mpc'] < 4)
kMpc = test_sim[0]['k_Mpc'][mask]


# %%
out = p3d_emu.predict_P3D_Mpc(
    sim_label=sim_label,
    z=z_test,
    emu_params=test_params,
    k_Mpc=kMpc,
    mu = np.ones(42),
    kpar_Mpc = kMpc,
    return_cov=True,
)

# %%
plt.imshow(np.log(out['p3d_cov']))
xticks_indices = np.linspace(0, kMpc.shape[0] - 1, 10, dtype=int)
plt.xticks(xticks_indices, [f'{k:.1f}' for k in kMpc[xticks_indices]], fontsize=12)
plt.yticks(xticks_indices, [f'{k:.1f}' for k in kMpc[xticks_indices]], fontsize=12)

# Add labels to x and y axis
plt.xlabel('$k$ [1/Mpc]', fontsize=16)
plt.ylabel('$k$ [1/Mpc]', fontsize=16)

# Add colorbar with label log(cov)
cbar = plt.colorbar()
cbar.set_label('log(Cov)', fontsize=16)

# Show plot
plt.show()

# %% [markdown]
# ## PLOT TEST SIMULATION AT z=z_test

# %%
sim_label = 'mpg_central'
z_test = 4
val_scaling=1.0


# %%
test_sim =  Archive3D.get_testing_data(
        sim_label, 
        force_recompute_plin=False
        )
test_sim = [d for d in test_sim if d['z']==z_test and d['val_scaling'] ==val_scaling]
k_Mpc =test_sim[0]["k_Mpc"]
k_mask = (k_Mpc < 4) & (k_Mpc > 0)
k_Mpc = k_Mpc[k_mask]

test_params = {param: test_sim[0][param] for param in p3d_emu.emuparams}



p3d_true = test_sim[0]['p3d_Mpc'][p3d_emu.k_mask]

# %%
out = p3d_emu.predict_P3D_Mpc(
    sim_label=sim_label,
    z=z_test, 
    emu_params=test_params
)
out.keys()

# %%
err_cov = np.sqrt(np.diagonal(out['p3d_cov']))

# %%
err_pred = out['p3d'] - p3d_true

# %%
frac_diff = (out['p3d'] / p3d_true - 1)*100
err_cov_diff = err_cov/p3d_true*100

# %%
plt.figure(figsize=(13,5))
plt.errorbar(x = np.arange(len(p3d_emu.k_Mpc_masked)),y=frac_diff, yerr=err_cov_diff, color='goldenrod' )
plt.axhline(y=0, ls='--', color='black')
plt.ylim(-20,20)
plt.xticks([],fontsize=66)
plt.yticks(fontsize=16)

plt.xlabel('$k$', fontsize=18)
plt.ylabel('Percent error', fontsize=18)


# %%
# Calculate the lower and upper bounds
lower_bound = frac_diff - err_cov/p3d_true*100
upper_bound = frac_diff + err_cov/p3d_true*100

# Count the number of measurements within the range
num_compatible = np.sum((lower_bound <= 0) & (upper_bound >= 0))

print(f"Number of measurements within +-error of 0: {num_compatible} out of {len(err_cov)}, which corresponds to {num_compatible/len(err_cov) *100}%")

# %% [markdown]
# ### Same for a limited range of mu

# %%
mu_min, mu_max = 0.31,0.38

# %%
frac_diff_masked = frac_diff[(p3d_emu.mu_masked>mu_min)&(p3d_emu.mu_masked<mu_max)]
err_cov_diff_masked = err_cov_diff[(p3d_emu.mu_masked>mu_min)&(p3d_emu.mu_masked<mu_max)]
kmasked = p3d_emu.k_Mpc_masked[(p3d_emu.mu_masked>mu_min)&(p3d_emu.mu_masked<mu_max)]

# %%
plt.figure(figsize=(13,5))
plt.errorbar(x = kmasked,y=frac_diff_masked, yerr=err_cov_diff_masked, color='goldenrod', label = r'$0.31<\mu<0.38$' )
plt.axhline(y=0, ls='--', color='black')
plt.ylim(-20,20)
plt.xticks(kmasked,fontsize=16)
plt.yticks(fontsize=16)

plt.xlabel('$k$ [1/Mpc]', fontsize=22)
plt.ylabel('Percent error', fontsize=22)
plt.xscale('log')

plt.legend(fontsize=18)

# %% [markdown]
# ## PIT DIAGRAM

# %%
from scipy.stats import norm

# %%
pit_list=[]
test_sim =  Archive3D.get_testing_data(sim_label, force_recompute_plin=False)
for iz, z in enumerate(np.arange(2,2.6,0.25)):

    test_sim_z = [d for d in test_sim if d['z']==z and d['val_scaling'] ==val_scaling]
    k_Mpc =test_sim_z[0]["k_Mpc"]
    k_mask = (k_Mpc < 4) & (k_Mpc > 0)
    k_Mpc = k_Mpc[k_mask]

    test_params = {param: test_sim_z[0][param] for param in p3d_emu.emuparams}
    p3d_true = test_sim_z[0]['p3d_Mpc'][p3d_emu.k_mask]
    
    out = p3d_emu.predict_P3D_Mpc(
        sim_label=sim_label,
        z=z, 
        emu_params=test_params
    )
    err_cov = np.sqrt(np.diagonal(out['p3d_cov']))
    
    mu, sigma = out['p3d'], err_cov
    
    for ii in range(len(p3d_true)):
        grid = np.linspace(-5*p3d_true[ii], 5*p3d_true[ii],10000)
        pdf_k = norm.pdf(grid, mu[ii], sigma[ii])
        cdf_k = norm.cdf(p3d_true[ii],mu[ii],sigma[ii])
        pit = norm.cdf(p3d_true[ii],mu[ii], sigma[ii]).sum()
        pit_list.append(pit)


# %%
plt.hist(pit_list, bins=15)
plt.xlabel('PIT')
plt.ylabel('Counts')

# %%

# %%
