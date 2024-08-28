# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: pcross
#     language: python
#     name: python3
# ---

import numpy as np
import matplotlib.pyplot as plt
import os
from forestflow import pcross
from lace.cosmo import camb_cosmo
from forestflow.model_p3d_arinyo import ArinyoModel
from forestflow.archive import GadgetArchive3D
from forestflow.P3D_cINN import P3DEmulator
import forestflow
from forestflow.rebin_p3d import get_p3d_modes
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# # Compare $P_\times$ measurements to emulator

path_program = os.path.dirname(forestflow.__path__[0]) + '/'
path_program

folder_lya_data = path_program + "/data/best_arinyo/"
print("Loading GadgetArchive3d; may take several minutes.")
Archive3D = GadgetArchive3D(
    base_folder=path_program[:-1],
    folder_data=folder_lya_data,
    force_recompute_plin=False,
    average="both",
)

# +
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

# -

meas_path = "../../data/px_sim_meas/"

separation_bins = np.concatenate((np.logspace(np.log10(.05),np.log10(2),5),np.linspace(6,15,4))) # the desired spacing for r_perp
kpar = np.fft.rfftfreq(1350, .05)*2*np.pi # frequency in Mpc^-1


def weighted_Px(kpar_est, z, p3d_func, sep_bin, ndiv=10, **pp):
    rrange = np.linspace(sep_bin[0], sep_bin[1], ndiv)
    rperp_pred,Pxper_r = pcross.Px_Mpc_detailed(kpar_est,
    p3d_func,
    z,
    rperp_choice=rrange,
    P3D_mode='pol',
    min_kperp=10**-3,
    max_kperp=10**2.9,
    nkperp=2**12,
    **{"pp": pp})
    
    Px_pred_smooth = np.average(Pxper_r, weights=rrange, axis=1)
    
    # repeat for exact kpar
    rperp_pred,Pxper_r2 = pcross.Px_Mpc_detailed(kpar[1:65],
    p3d_func,
    z,
    rperp_choice=rrange,
    P3D_mode='pol',
    min_kperp=10**-3,
    max_kperp=10**2.9,
    nkperp=2**12,
    **{"pp": pp})

    Px_pred_same_kpar = np.average(Pxper_r2, weights=rrange, axis=1)
    return Px_pred_smooth, Px_pred_same_kpar


# +
Px_pred = [[],[],[],[],[],[],[]]
Px_pred_plus = [[],[],[],[],[],[],[]]
Px_pred_minus = [[],[],[],[],[],[],[]]
Px_pred_same_kpar = [[],[],[],[],[],[],[]]
Px_pred_same_kpar_plus = [[],[],[],[],[],[],[]]
Px_pred_same_kpar_minus = [[],[],[],[],[],[],[]]
Px_pred_bestfit = [[],[],[],[],[],[],[]]
Px_pred_same_kpar_bestfit = [[],[],[],[],[],[],[]]

Px_pred_all = [[],[],[],[],[],[],[]]
Px_pred_samekpar_all = [[],[],[],[],[],[],[]]

z_test = np.array([3]) 

info_power = {
    "sim_label": "mpg_central",
    "z": z_test,
}

sim_label = info_power["sim_label"]
test_sim = Archive3D.get_testing_data(
    sim_label, force_recompute_plin=False
)
test_sim_z = [d for d in test_sim if d["z"] == info_power["z"]]
emu_params = test_sim_z[0]
print("Evaluating emulator.")
out = emulator.evaluate(
    emu_params=emu_params,
    info_power=info_power,
    natural_params=False,
    Nrealizations=100,
    return_all_realizations=True

)
print("Done.")
cosmo = camb_cosmo.get_cosmology_from_dictionary(test_sim[0]["cosmo_params"])
camb_results = camb_cosmo.get_camb_results(cosmo, zs=z_test, camb_kmax_Mpc=1000) # set default cosmo
arinyo = ArinyoModel(cosmo=cosmo, camb_results=camb_results, zs=z_test, camb_kmax_Mpc=1000) # set model

kpar_est = np.logspace(np.log10(kpar[1]), np.log10(kpar[66]),40)
nkpar = len(kpar_est)
min_kperp=10**-3,
nkperp=2**11
max_kperp=10**2.9
kperps = np.logspace(
    np.log10(min_kperp), np.log10(max_kperp), nkperp
)
kperp2d = np.tile(kperps[:, np.newaxis], nkpar)  # mu grid for P3D
kpar2d = np.tile(kpar_est[:, np.newaxis], nkperp).T
k2d = np.sqrt(kperp2d**2 + kpar2d**2)
mu2d = kpar2d / k2d

# get the best-fit parameters
arinyo_bestfit = test_sim_z[0]['Arinyo_min'] # best-fitting Arinyo params
coeffs_arinyo_plus = out['coeffs_Arinyo'].copy()
for key in coeffs_arinyo_plus.keys():
    coeffs_arinyo_plus[key] = coeffs_arinyo_plus[key] + out['coeffs_Arinyo_std'][key]
coeffs_arinyo_minus = out['coeffs_Arinyo'].copy()
for key in coeffs_arinyo_minus.keys():
    coeffs_arinyo_minus[key] = coeffs_arinyo_minus[key] - out['coeffs_Arinyo_std'][key]

for s in range(1, len(separation_bins)-1):
    mean_Px_smth, mean_Px_exact = weighted_Px(kpar_est, z_test[0], arinyo.P3D_Mpc, [separation_bins[s], separation_bins[s+1]], **out['coeffs_Arinyo'], ndiv=30)
    mean_Px_smth_plus, mean_Px_exact_plus = weighted_Px(kpar_est, z_test[0], arinyo.P3D_Mpc, [separation_bins[s], separation_bins[s+1]], **coeffs_arinyo_plus, ndiv=30)
    mean_Px_smth_minus, mean_Px_exact_minus = weighted_Px(kpar_est, z_test[0], arinyo.P3D_Mpc, [separation_bins[s], separation_bins[s+1]], **coeffs_arinyo_minus, ndiv=30)
    print("Now running all realizations.")
    for i in range(len(out['coeffs_Arinyo_all'])):
        if i%10==0:
            print(f"{i/10} pct done.")
        coeffs_i = out['coeffs_Arinyo_all'][i]
        mean_Px_smth_i, mean_Px_exact_i = weighted_Px(kpar_est, z_test[0], arinyo.P3D_Mpc, [separation_bins[s], separation_bins[s+1]], **coeffs_i, ndiv=30)
        Px_pred_all[s-1].append(mean_Px_smth_i)
        Px_pred_samekpar_all[s-1].append(mean_Px_exact_i)
    print(len(Px_pred_samekpar_all[s-1]))
    mean_Px_smth_bestfit, mean_Px_exact_bestfit = weighted_Px(kpar_est, z_test[0], arinyo.P3D_Mpc, [separation_bins[s], separation_bins[s+1]], **arinyo_bestfit, ndiv=30)
    
    Px_pred[s-1].extend(mean_Px_smth)
    Px_pred_plus[s-1].extend(mean_Px_smth_plus)
    Px_pred_minus[s-1].extend(mean_Px_smth_minus)
    Px_pred_same_kpar[s-1].extend(mean_Px_exact)
    Px_pred_same_kpar_plus[s-1].extend(mean_Px_exact_plus)
    Px_pred_same_kpar_minus[s-1].extend(mean_Px_exact_minus)
    Px_pred_bestfit[s-1].extend(mean_Px_smth_bestfit)
    Px_pred_same_kpar_bestfit[s-1].extend(mean_Px_exact_bestfit)


# +
Px_pred = np.asarray(Px_pred)
Px_pred_plus = np.asarray(Px_pred_plus)
Px_pred_minus = np.asarray(Px_pred_minus)
Px_pred_same_kpar = np.asarray(Px_pred_same_kpar)

Px_pred_same_kpar_plus = np.asarray(Px_pred_same_kpar_plus)
Px_pred_same_kpar_minus = np.asarray(Px_pred_same_kpar_minus)
Px_pred_bestfit = np.asarray(Px_pred_bestfit)
Px_pred_same_kpar_bestfit = np.asarray(Px_pred_same_kpar_bestfit)
# -

Px_pred_all_stds = []
for sbin in Px_pred_all:
    Px_pred_all_stds.append(np.std(np.asarray(sbin), axis=0))
Px_pred_all_stds_samek = []
for sbin in Px_pred_samekpar_all:
    Px_pred_all_stds_samek.append(np.std(np.asarray(sbin), axis=0))
Px_pred_all_stds = np.asarray(Px_pred_all_stds)
Px_pred_all_stds_samek = np.asarray(Px_pred_all_stds_samek)

# +
fs = 13
plt.rcParams.update({'font.size': 16})
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=[7,9], gridspec_kw={'height_ratios': [4, 1,1]})
for s in range(1,len(separation_bins)-1):
    print("{:.2f}_{:.2f}".format(separation_bins[s],separation_bins[s+1]))
    Px_info = np.load(meas_path+"Px_skewers_{:.2f}_{:.2f}_allax_allphase.npz".format(separation_bins[s],separation_bins[s+1]))
    Px_thisbin_avg = Px_info['Px']
    Px_std = Px_info['std']

    if s==4:
        label = ''
        
    else:
        label = ''
        
    if s<=4:
        
        # ax[0].scatter(kpar[1:65], kpar[1:65]*Px_thisbin_avg.T[1:], label=r"$r_\perp={:.2f}-{:.2f}$ Mpc".format(separation_bins[s],separation_bins[s+1]), marker='o', s=10, color=colors[s])
        ax[0].scatter(kpar[1:65], kpar[1:65]*Px_thisbin_avg.T[1:], marker='o', s=20, color=colors[s])
        ax[0].plot(kpar_est, kpar_est*(Px_pred[s-1]), label=label, color=colors[s])
        ax[0].fill_between(kpar_est, kpar_est*(Px_pred[s-1])-(Px_pred_all_stds[s-1]),kpar_est*(Px_pred[s-1])+(Px_pred_all_stds[s-1]), alpha=.2, color=colors[s])
        ax[0].plot(kpar_est, kpar_est*(Px_pred_bestfit[s-1]), color=colors[s], linestyle='dashed')
        cond = (Px_pred_same_kpar[s-1][:64]) > (np.amax((Px_pred_same_kpar[s-1][:64]))/100.)
        cond2 = Px_pred[s-1]>np.amax(Px_pred[s-1])/100.
        fracerr = ((Px_thisbin_avg[1:])[cond]-(Px_pred_same_kpar[s-1][:64])[cond])/(Px_pred_same_kpar[s-1][:64])[cond]
        fracerr_bestfit = (Px_thisbin_avg.T[1:][cond]-(Px_pred_same_kpar_bestfit[s-1][:64])[cond])/(Px_pred_same_kpar_bestfit[s-1][:64])[cond]
        fracerr_emubest = ((Px_pred[s-1][cond2])-(Px_pred_bestfit[s-1][cond2]))/(Px_pred_bestfit[s-1][cond2])
        fracerr_emubest_plus = (((Px_pred[s-1][cond2])+(Px_pred_all_stds[s-1][cond2]))-(Px_pred_bestfit[s-1][cond2]))/(Px_pred_bestfit[s-1][cond2])
        fracerr_emubest_minus = (((Px_pred[s-1][cond2])-(Px_pred_all_stds[s-1][cond2]))-(Px_pred_bestfit[s-1][cond2]))/(Px_pred_bestfit[s-1][cond2])
        print("best-fit first crosses 10% at: ", kpar[np.where(np.abs(fracerr_bestfit)>0.1)[0]])
        print("Emulator first crosses 5% at: ", kpar_est[np.where(np.abs(fracerr_emubest)>0.05)[0]])
        print("Emu performance", np.average(fracerr_emubest))
        ax[1].plot(kpar[1:65][cond], fracerr_bestfit, color=colors[s])
        ax[2].plot(kpar_est[cond2], fracerr_emubest, color=colors[s])
        ax[2].fill_between(kpar_est[cond2], fracerr_emubest_minus, fracerr_emubest_plus, color=colors[s], alpha=.2)
        print(kpar[1:65][cond][abs(fracerr)>0.1], "over")
        print(np.amax(abs(fracerr)))
ax[1].axhline(y=0.1, color="black", ls="--", alpha=0.8)
ax[1].axhline(y=-0.1, color="black", ls="--", alpha=0.8)
ax[2].axhline(y=0.1, color="black", ls="--", alpha=0.8)
ax[2].axhline(y=-0.1, color="black", ls="--", alpha=0.8)
ax[1].axhline(y=0, color='black', ls="dotted")
ax[2].axhline(y=0, color='black', ls="dotted")

black_line = mlines.Line2D([], [], color='black', label='ForestFlow')
dashed_line = mlines.Line2D([], [], color='black', label='Fit', linestyle='dashed')
scatter_point = mlines.Line2D([], [], color='black', marker='o', label='Simulation', linestyle='None')
# get existing legend handles
handles, labels = ax[0].get_legend_handles_labels()
# add black line to handles
handles.append(black_line)
handles.append(dashed_line)
handles.append(scatter_point)
# add legend
leg1 = ax[0].legend(handles=handles, loc='upper right')
ax[0].add_artist(leg1)
# add another legend with patches of colors

ax[0].tick_params(axis="both", which="major", labelsize=18)
ax[1].tick_params(axis="both", which="major", labelsize=18)
patch_handles = []
for i in range(1, len(separation_bins)-4):
    patch_handles.append(mpatches.Patch(color=colors[i], label=r"$r_\perp={:.2f}-{:.2f}$ Mpc".format(separation_bins[i],separation_bins[i+1])))
ax[0].legend(handles=patch_handles, loc='upper left')

ax[0].set_xscale('log')
ax[0].set_ylabel(r"$k_\parallel P_\times$", fontsize=20)
ax[0].set_xlim([0.09,6])
ax[0].set_ylim([-.01,0.28])
ax[2].set_xlabel(r"$k_\parallel$ [Mpc$^{-1}]$", fontsize=20)
# ax[0].set_title(fr"$P_\times$, central simulation, $z={z_test[0]}$", fontsize=20)
ax[1].set_ylim([-.22,.22])
ax[2].set_ylim([-.22,.22])
plt.subplots_adjust(hspace=.1)
ax[1].set_ylabel(r"$P_{\times}^{\mathrm{sim}}/P_{\times}^{\mathrm{fit}}-1$", fontsize=20)
ax[2].set_ylabel(r"$P_{\times}^{\mathrm{emu}}/P_{\times}^{\mathrm{fit}}-1$", fontsize=20)

plt.tight_layout()
plt.savefig("Pcross_central_snap6_kPx_allbin_emupred_first4_withfracerr.pdf")

# +
out = {}

s = 1
for key in ["orange", "green", "red", "purple"]:

    Px_info = np.load(meas_path+"Px_skewers_{:.2f}_{:.2f}_allax_allphase.npz".format(separation_bins[s],separation_bins[s+1]))
    Px_thisbin_avg = Px_info['Px']
    
    out["top_" + key + "_dashed_x"] = kpar_est
    out["top_" + key + "_dashed_y"] = kpar_est*(Px_pred_bestfit[s-1])
    
    out["top_" + key + "_solid_x"] = kpar_est
    out["top_" + key + "_solid_y"] = kpar_est*(Px_pred[s-1])
    
    out["top_" + key + "_points_x"] = kpar[1:65]
    out["top_" + key + "_points_y"] = kpar[1:65]*Px_thisbin_avg.T[1:]
    
    cond = (Px_pred_same_kpar[s-1][:64]) > (np.amax((Px_pred_same_kpar[s-1][:64]))/100.)
    cond2 = Px_pred[s-1]>np.amax(Px_pred[s-1])/100.
        
    fracerr_bestfit = (Px_thisbin_avg.T[1:][cond]-(Px_pred_same_kpar_bestfit[s-1][:64])[cond])/(Px_pred_same_kpar_bestfit[s-1][:64])[cond]
    out["middle_" + key + "_solid_x"] = kpar[1:65][cond]
    out["middle_" + key + "_solid_y"] = fracerr_bestfit

    fracerr_emubest = ((Px_pred[s-1][cond2])-(Px_pred_bestfit[s-1][cond2]))/(Px_pred_bestfit[s-1][cond2])    
    out["bottom_" + key + "_solid_x"] = kpar_est[cond2]
    out["bottom_" + key + "_solid_y"] = fracerr_emubest
    
    s+=1
import forestflow
path_forestflow= os.path.dirname(forestflow.__path__[0]) + "/"
folder = path_forestflow + "data/figures_machine_readable/"
np.save(folder + "fig8", out)
# -


