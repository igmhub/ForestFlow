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
#     display_name: forestflow
#     language: python
#     name: forestflow
# ---

# %% [markdown]
# # NOTEBOOK TO REPRODUCE THE LEAVE-REDSHIFT-OUT TEST OF FORESTFLOW 

# %%
import numpy as np
import os
import sys
import pandas as pd
import scipy.stats as stats

# %%
from ForestFlow.model_p3d_arinyo import ArinyoModel
from ForestFlow import model_p3d_arinyo
from ForestFlow.archive import GadgetArchive3D
from ForestFlow.P3D_cINN import P3DEmulator
from ForestFlow.likelihood import Likelihood


# %%

import matplotlib.pyplot as plt
import matplotlib
font = {'size'   : 22}
matplotlib.rc('font', **font)
plt.rc('text', usetex=False)
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15 )



# %% [markdown]
#
# ## DEFINE FUNCTIONS

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
def sigma68(data): return 0.5*(pd.DataFrame(data).quantile(q = 0.84, axis = 0) - pd.DataFrame(data).quantile(q = 0.16, axis = 0)).values



# %%
def plot_p3d_LzO(fractional_errors):
    # Extract data from Archive3D
    k_Mpc = Archive3D.training_data[0]["k3d_Mpc"]
    mu = Archive3D.training_data[0]["mu3d"]

    # Apply a mask to select relevant k values
    k_mask = (k_Mpc < 4) & (k_Mpc > 0)
    k_Mpc = k_Mpc[k_mask]
    mu = mu[k_mask]

    # Create subplots with shared y-axis and x-axis
    fig, axs = plt.subplots(len(z_test), 1, figsize=(6, 8), sharey=True, sharex=True)
    
    # Define mu bins
    mu_lims = [[0, 0.06], [0.31, 0.38], [0.62, 0.69], [0.94, 1]]
    
    # Define colors for different mu bins
    colors = ['navy', 'crimson', 'forestgreen', 'goldenrod']
    
    # Loop through redshifts
    for ii, z in enumerate(z_test):
        axs[ii].set_title(f'$z={z}$', fontsize=14)
        axs[ii].axhline(y=-10, ls='--', color='black')
        axs[ii].axhline(y=10, ls='--', color='black')

        # Loop through mu bins
        for mi in range(int(len(mu_lims))):
            mu_mask = (mu >= mu_lims[mi][0]) & (mu <= mu_lims[mi][1])
            k_masked = k_Mpc[mu_mask]

            # Calculate fractional error statistics
            frac_err = np.nanmedian(fractional_errors[:, ii, :], 0)
            frac_err_err = sigma68(fractional_errors[:, ii, :])

            frac_err_masked = frac_err[mu_mask]
            frac_err_err_masked = frac_err_err[mu_mask]

            color = colors[mi]

            # Add a line plot with shaded error region to the current subplot
            axs[ii].plot(k_masked, frac_err_masked, label=f'${mu_lims[mi][0]}\leq \mu \leq {mu_lims[mi][1]}$', color=color)
            axs[ii].fill_between(
                k_masked,
                frac_err_masked - frac_err_err_masked,
                frac_err_masked + frac_err_err_masked,
                color=color,
                alpha=0.2,
            )
            axs[ii].tick_params(axis='both', which='major', labelsize=16)

    # Customize subplot appearance
    for xx, ax in enumerate(axs):
        if xx == len(axs) // 2:  # Centered y-label
            ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.set_ylim(-10, 10)

    axs[len(axs) - 1].set_xlabel(r"$k$ [1/Mpc]", fontsize=16)
    
    axs[0].legend(fontsize=12)

    # Adjust spacing between subplots
    plt.tight_layout()
    fig.text(0, 0.5, r'Error $P_{\rm 3D}$ [%]', va='center', rotation='vertical', fontsize=16)

    
    # Save the plot
    #plt.savefig(savename, bbox_inches='tight')



# %%
def plot_p1d_LzO(fractional_errors):
    # Create subplots with shared y-axis
    fig, axs = plt.subplots(len(z_test), 1, figsize=(6, 8), sharey=True)

    
    # Loop through redshifts
    for ii, z in enumerate(z_test):
        axs[ii].set_title(f'$z={z}$', fontsize=16)
        axs[ii].axhline(y=-1, ls='--', color='black')
        axs[ii].axhline(y=1, ls='--', color='black')

        # Calculate fractional error statistics
        frac_err = np.nanmedian(fractional_errors[:, ii, :], 0)
        frac_err_err = sigma68(fractional_errors[:, ii, :])

        # Mask for k values greater than 0
        k_plot = k_Mpc[(k_Mpc > 0)]

        # Add a line plot with shaded error region to the current subplot
        axs[ii].plot(k1d_sim, frac_err, color='crimson')
        axs[ii].fill_between(
            k1d_sim,
            frac_err - frac_err_err,
            frac_err + frac_err_err,
            color='crimson',
            alpha=0.2,
        )

        axs[ii].tick_params(axis='both', which='major', labelsize=18)

    # Customize subplot appearance
    for xx, ax in enumerate(axs):
        if xx == len(axs) // 2:  # Centered y-label
            ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.set_ylim(-5, 5)

    axs[len(axs) - 1].set_xlabel(r"$k$ [1/Mpc]", fontsize=16)
    axs[0].legend(fontsize=12)

    # Adjust spacing between subplots
    plt.tight_layout()
    fig.text(0, 0.5, r'Error $P_{\rm 1D}$ [%]', va='center', rotation='vertical', fontsize=16)

    
    # Save the plot
    #plt.savefig(savename, bbox_inches='tight')



# %% [markdown]
# # LOAD DATA

# %%
# %%time
folder_interp = path_program+"/data/plin_interp/"
folder_lya_data = path_program +  "/data/best_arinyo/"

Archive3D = GadgetArchive3D(
    base_folder=path_program, 
    folder_data=folder_lya_data, 
    force_recompute_plin=False,
    average='both'
)
print(len(Archive3D.training_data))


# %%
Nrealizations=100
Nsim=30

test_sim = central = Archive3D.get_testing_data(
        'mpg_central', 
        force_recompute_plin=True
        )
z_grid = [d['z'] for d in test_sim]
Nz=len(z_grid)


k_Mpc = Archive3D.training_data[0]["k3d_Mpc"]
mu = Archive3D.training_data[0]["mu3d"]

k_mask = (k_Mpc < 4) & (k_Mpc > 0)

k_Mpc = k_Mpc[k_mask]
mu = mu[k_mask]

# %% [markdown]
# ## LEAVE REDSHIFT OUT TEST

# %% [markdown]
# #### Define redshifts to test
#

# %%
z_test = [2.5,3.5]

# %%
p3ds_pred = np.zeros(shape=(Nsim,len(z_test),148))
p1ds_pred = np.zeros(shape=(Nsim,len(z_test),53))

p3ds_arinyo = np.zeros(shape=(Nsim,len(z_test),148))
p1ds_arinyo= np.zeros(shape=(Nsim, len(z_test), 53))

p1ds_sims = np.zeros(shape=(Nsim, len(z_test), 53))
p3ds_sims = np.zeros(shape=(Nsim, len(z_test), 148))


for iz, zdrop in enumerate(z_test):
    print(f'Dropping redshift {z_test}')
    

    training_data= [d for d in Archive3D.training_data if d['z']!=zdrop]
    
    p3d_emu = P3DEmulator(
        training_data,
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
        folder_chains='/data/desi/scratch/jchavesm/p3d_fits_new/',
        model_path=f'../data/emulator_models/mpg_dropz{zdrop}.pt'
    )
    
    for s in range(Nsim):
        
        #load arinyo module
        flag = f'Plin_interp_sim{s}.npy'
        file_plin_inter = folder_interp + flag
        pk_interp = np.load(file_plin_inter, allow_pickle=True).all()
        model_Arinyo = model_p3d_arinyo.ArinyoModel(camb_pk_interp=pk_interp)

        #define test sim
        dict_sim = [d for d in Archive3D.training_data if d['z']==zdrop and d['sim_label']==f'mpg_{s}' and d['val_scaling'] ==1]
    
        #p1d from sim
        like = Likelihood(dict_sim[0], Archive3D.rel_err_p3d, Archive3D.rel_err_p1d )
        k1d_mask = like.like.ind_fit1d.copy() 
        p1d_sim = like.like.data["p1d"][k1d_mask]
    
        #p3d from sim
        p3d_sim = dict_sim[0]['p3d_Mpc'][p3d_emu.k_mask]
        p3d_sim = np.array(p3d_sim)
        
        p1ds_sims[s,iz]=p1d_sim
        p3ds_sims[s,iz]=p3d_sim
    
            
        #load BF Arinyo and estimated the p3d and p1d from BF arinyo parameters
        BF_arinyo = dict_sim[0]['Arinyo_minin']        
        
        p3d_arinyo = model_Arinyo.P3D_Mpc(zdrop,k_Mpc, mu,BF_arinyo )
        p3ds_arinyo[s,iz] = p3d_arinyo
        
        p1d_arinyo = like.like.get_model_1d(parameters=BF_arinyo)
        p1d_arinyo = p1d_arinyo[k1d_mask]
        p1ds_arinyo[s,iz] = p1d_arinyo
        
        
        
        #predict p3d and p1d from predicted arinyo parameters        
        p3d_pred_median = p3d_emu.predict_P3D_Mpc(
                                sim_label=f'mpg_{s}',
                                z=zdrop,
                                test_sim=dict_sim,    
                                return_cov=False)
        
        p1d_pred_median = p3d_emu.predict_P1D_Mpc(
                        sim_label=f'mpg_{s}',
                        z=zdrop,
                        test_sim=dict_sim,    
                        return_cov=False)
        
        p3ds_pred[s,iz]=p3d_pred_median
        p1ds_pred[s,iz]=p1d_pred_median
        
    print('Mean fractional error P3D pred to Arinyo', ((p3ds_pred[:,iz]/p3ds_arinyo[:,iz] - 1)*100).mean())
    print('Std fractional error P3D pre to Arinyo', ((p3ds_pred[:,iz]/p3ds_arinyo[:,iz] - 1)*100).std())
    
    print('Mean fractional error P3D Arinyo model', ((p3ds_arinyo[:,iz]/p3ds_sims[:,iz] - 1)*100).mean())
    print('Std fractional error P3D Arinyo model', ((p3ds_arinyo[:,iz]/p3ds_sims[:,iz] - 1)*100).std())
    
    print('Mean fractional error P3D pred to sim', ((p3ds_pred[:,iz]/p3ds_sims[:,iz] - 1)*100).mean())    
    print('Std fractional error P3D pred to sim', ((p3ds_pred[:,iz]/p3ds_sims[:,iz] - 1)*100).std())
    
    print('Mean fractional error P1D pred to Arinyo', ((p1ds_pred[:,iz]/p1ds_arinyo[:,iz] - 1)*100).mean())
    print('Std fractional error P1D pred to Arinyo', ((p1ds_pred[:,iz]/p1ds_arinyo[:,iz] - 1)*100).std())
    
    
    print('Mean fractional error P1D Arinyo model', ((p1ds_arinyo[:,iz]/p1ds_sims[:,iz] - 1)*100).mean())
    print('Std fractional error P1D Arinyo model', ((p1ds_arinyo[:,iz]/p1ds_sims[:,iz] - 1)*100).std())
    
    print('Mean fractional error P1D pred to sim', ((p1ds_pred[:,iz]/p1ds_sims[:,iz] - 1)*100).mean())
    print('Std fractional error P1D pred to sim', ((p1ds_pred[:,iz]/p1ds_sims[:,iz] - 1)*100).std())
        


# %% [markdown]
# ## PLOTTING

# %%
fractional_errors_arinyo =  (p3ds_pred / p3ds_arinyo-1)*100
fractional_errors_sims = (p3ds_pred / p3ds_sims-1)*100
fractional_errors_bench = (p3ds_arinyo / p3ds_sims-1)*100

# %%
plot_p3d_LzO(fractional_errors_arinyo)

# %%
fractional_errors_arinyo_p1d = (p1ds_pred / p1ds_arinyo -1)*100
fractional_errors_sims_p1d = (p1ds_pred / p1ds_sims -1)*100
fractional_errors_bench_p1d = (p1ds_arinyo / p1ds_sims -1)*100

# %%
#any simulation to get k1d_sim
dict_sim = [d for d in Archive3D.training_data if d['z']==3 and d['sim_label']==f'mpg_4' and d['val_scaling'] ==1]
like = Likelihood(dict_sim[0], Archive3D.rel_err_p3d, Archive3D.rel_err_p1d )
k1d_mask = like.like.ind_fit1d.copy() 
k1d_sim = like.like.data["k1d"][k1d_mask]

# %%
plot_p1d_LzO(fractional_errors_sims_p1d)

# %%
