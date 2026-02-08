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
# # Central simulation at z=3, l10 test simulations, referee

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
# training_type = "Arinyo_minz"

# if (training_type == "Arinyo_min_q1"):
#     nparams = 7
#     model_path = path_program+"/data/emulator_models/mpg_q1/mpg_hypercube.pt"
# elif(training_type == "Arinyo_min"):
#     nparams = 8
#     # model_path = path_program+"/data/emulator_models/mpg_q1_q2/mpg_hypercube.pt"
#     model_path=path_program+"/data/emulator_models/mpg_joint.pt"
# elif(training_type == "Arinyo_minz"):
#     nparams = 8
#     # model_path = path_program+"/data/emulator_models/mpg_q1_q2/mpg_hypercube.pt"
#     model_path=path_program+"/data/emulator_models/mpg_jointz.pt"



# load_old 
# training_type = "Arinyo_min"
# model_path=path_program+"/data/emulator_models/mpg_hypercube.pt"

# emulator = P3DEmulator(
#     Archive3D.training_data,
#     Archive3D.emu_params,
#     nepochs=300,
#     lr=0.001,  # 0.005
#     batch_size=20,
#     step_size=200,
#     gamma=0.1,
#     weight_decay=0,
#     adamw=True,
#     nLayers_inn=12,  # 15
#     Archive=Archive3D,
#     Nrealizations=10000,
#     training_type=training_type,
#     model_path=model_path,
#     # save_path=model_path,
# )

emulator = P3DEmulator(
    model_path=path_program+"/data/emulator_models/new_emu.pt",
)

# %% [markdown]
# #### General stuff

# %%
Nsim = 30
zs = np.flip(np.arange(2, 4.6, 0.25))
Nz = zs.shape[0]

n_mubins = 4
kmax_3d_fit = 5
kmax_1d_fit = 4
kmax_3d_plot = kmax_3d_fit + 1
kmax_1d_plot = kmax_1d_fit + 1

sim = Archive3D.training_data[0]

k3d_Mpc = sim['k3d_Mpc']
mu3d = sim['mu3d']
p3d_Mpc = sim['p3d_Mpc']
kmu_modes = get_p3d_modes(kmax_3d_plot)

mask_3d = k3d_Mpc[:, 0] <= kmax_3d_plot

mask_1d = (sim['k_Mpc'] <= kmax_1d_plot) & (sim['k_Mpc'] > 0)
k1d_Mpc = sim['k_Mpc'][mask_1d]
p1d_Mpc = sim['p1d_Mpc'][mask_1d]

# %%
# np.sum(k3d_Mpc <= kmax_3d_fit)
# np.sum((sim['k_Mpc'] <= kmax_1d_fit) & (sim['k_Mpc'] > 0))

# %% [markdown]
# ### Central simulation

# %%
# %%time

zcen = 3

info_power = {
    "sim_label": "mpg_central",
    "k3d_Mpc": k3d_Mpc[mask_3d, :],
    "mu": mu3d[mask_3d, :],
    "kmu_modes": kmu_modes,
    "k1d_Mpc": k1d_Mpc,
    "return_p3d": True,
    "return_p1d": True,
    # "return_cov": True,
    "z": zcen,
}

sim_label = info_power["sim_label"]
test_sim = Archive3D.get_testing_data(
    sim_label, force_recompute_plin=False
)
test_sim_z = [d for d in test_sim if d["z"] == info_power["z"]]
emu_params = test_sim_z[0]

input_emu = {}
for par in emulator.emu_input_names:
    input_emu[par] = emu_params[par]


out = emulator.evaluate(
    emu_params=input_emu,
    info_power=info_power,
    # Nrealizations=2
    Nrealizations=3000
)

# %% [markdown]
# #### Rebin data

# %%
_ = p3d_rebin_mu(out["k_Mpc"], out["mu"], test_sim_z[0]["p3d_Mpc"][mask_3d], kmu_modes, n_mubins=n_mubins)
knew, munew, rebin_p3d_sim, mu_bins = _

_ = p3d_rebin_mu(out["k_Mpc"], out["mu"], out["p3d"], kmu_modes, n_mubins=n_mubins)
knew, munew, rebin_p3d_emu, mu_bins = _

# _ = p3d_rebin_mu(out["k_Mpc"], out["mu"], out["p3d_std"], kmu_modes, n_mubins=n_mubins)
# knew, munew, rebin_p3d_std_emu, mu_bins = _

_ = p3d_rebin_mu(out["k_Mpc"], out["mu"], out["Plin"], kmu_modes, n_mubins=n_mubins)
knew, munew, rebin_plin, mu_bins = _

# %% [markdown]
# Precision

# %%
_ = np.isfinite(knew) & (knew > 0.5) & (knew < 5)
rat = rebin_p3d_sim[_]/rebin_p3d_emu[_]- 1
y = np.percentile(rat, [50, 16, 84])
print(y[0]*100, 0.5*(y[2]-y[1])*100, np.std(rat)*100)

# %%
norm_p1d = out["k1d_Mpc"]/np.pi
p1d_emu = norm_p1d * out["p1d"]
# p1d_std_emu = norm_p1d * out["p1d_std"]
p1d_sim = norm_p1d * test_sim_z[0]["p1d_Mpc"][mask_1d]

# %%
_ = np.isfinite(k1d_Mpc) & (k1d_Mpc < 4) & (k1d_Mpc > 0)
rat = p1d_sim[_]/p1d_emu[_] - 1
y = np.percentile(rat, [50, 16, 84])
print(y[0]*100, 0.5*(y[2]-y[1])*100, np.std(rat)*100)

# %%
rebin_p3d_std_emu = rebin_p3d_emu * 0.001

# %%
# folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"
folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures_new/"
plot_p3d_snap(
    folder, 
    knew, 
    munew,
    rebin_p3d_sim/rebin_plin,
    rebin_p3d_emu/rebin_plin,
    rebin_p3d_std_emu/rebin_plin,
    mu_bins,
    kmax_3d_fit=kmax_3d_fit
)

# %%
p1d_std_emu = p1d_emu * 0.001
plot_p1d_snap(
    folder, 
    out["k1d_Mpc"], 
    p1d_sim,
    p1d_emu,
    p1d_std_emu,
    kmax_1d=kmax_1d_plot,
    kmax_1d_fit=kmax_1d_fit,
)

# %% [markdown]
# ### Save data for zenodo

# %%
conv = {}
conv["blue"] = 0
conv["orange"] = 1
conv["green"] = 2
conv["red"] = 3
outs = {}

for key in conv.keys():
    ii = conv[key]
    
    outs["p3d_top_" + key + "_dotted_x"] = knew[:, ii]
    outs["p3d_top_" + key + "_dotted_y"] = rebin_p3d_sim[:, ii]/rebin_plin[:, ii]
    
    outs["p3d_top_" + key + "_solid_x"] = knew[:, ii]
    outs["p3d_top_" + key + "_solid_y"] = rebin_p3d_emu[:, ii]/rebin_plin[:, ii]

    outs["p3d_bottom_" + key + "_x"] = knew[:, ii]
    outs["p3d_bottom_" + key + "_y"] = rebin_p3d_emu[:, ii]/rebin_p3d_sim[:, ii]

outs["p1d_top_blue_x"] = out["k1d_Mpc"]
outs["p1d_top_blue_y"] = p1d_sim

outs["p1d_top_orange_x"] = out["k1d_Mpc"]
outs["p1d_top_orange_y"] = p1d_emu

outs["p1d_bottom_x"] = out["k1d_Mpc"]
outs["p1d_bottom_y"] = p1d_emu/p1d_sim


# %%
import forestflow
path_forestflow= os.path.dirname(forestflow.__path__[0]) + "/"
folder = path_forestflow + "data/figures_machine_readable/"
np.save(folder + "fig4", outs)

# %%
res = np.load(folder + "fig4.npy", allow_pickle=True).item()
res.keys()

# %%

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
    # "mpg_running",
    "mpg_reio",
]

# sim_labels = [
#     "mpg_central"
# ]

# %%
from forestflow.utils import transform_arinyo_params

# %%
arr_p3d_sim = np.zeros((len(sim_labels), Nz, np.sum(mask_3d), n_mubins))
arr_p3d_emu = np.zeros((len(sim_labels), Nz, np.sum(mask_3d), n_mubins))
arr_p1d_sim = np.zeros((len(sim_labels), Nz, np.sum(mask_1d)))
arr_p1d_emu = np.zeros((len(sim_labels), Nz, np.sum(mask_1d)))
params_sim = np.zeros((len(sim_labels), Nz, 3))
params_emu = np.zeros((len(sim_labels), Nz, 3))

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

        input_pars = {}
        for par in emulator.emu_input_names:
            input_pars[par] = test_sim_z[0][par]
        
        out = emulator.evaluate(
            emu_params=input_pars,
            info_power=info_power,
            return_bias_eta=True,
            Nrealizations=3000
        )
        
        _ = p3d_rebin_mu(out["k_Mpc"], out["mu"], test_sim_z[0]["p3d_Mpc"][mask_3d], kmu_modes, n_mubins=n_mubins)
        knew, munew, arr_p3d_sim[isim, iz], mu_bins = _
        
        _ = p3d_rebin_mu(out["k_Mpc"], out["mu"], out["p3d"], kmu_modes, n_mubins=n_mubins)
        knew, munew, arr_p3d_emu[isim, iz], mu_bins = _
        
        arr_p1d_emu[isim, iz] = out["p1d"]
        arr_p1d_sim[isim, iz] = test_sim_z[0]["p1d_Mpc"][mask_1d]

        params_sim[isim, iz, 0] = test_sim_z[0]["Arinyo_min"]["bias"]
        params_sim[isim, iz, 1] = test_sim_z[0]["Arinyo_min"]["bias"] * test_sim_z[0]["Arinyo_min"]["beta"] / out['coeffs_Arinyo']["f_p"]
        params_sim[isim, iz, 2] = test_sim_z[0]["Arinyo_min"]["beta"]
        # _ = new_params = transform_arinyo_params(
        #     test_sim_z[0]["Arinyo_min"], 
        #     test_sim_z[0]["f_p"]
        # )
        # params_sim[isim, iz, 1] = _["bias_eta"]

        params_emu[isim, iz, 0] = out["coeffs_Arinyo"]["bias"]        
        params_emu[isim, iz, 1] = out["coeffs_Arinyo"]["bias_eta"]
        params_emu[isim, iz, 2] = out["coeffs_Arinyo"]["beta"]        
        # _ = new_params = transform_arinyo_params(
        #     out["coeffs_Arinyo"], 
        #     test_sim_z[0]["f_p"]
        # )
        # params_emu[isim, iz, 1] = _["bias_eta"]

# %%
# folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"
folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures_new/"
np.savez(
    # folder + "temporal_central", 
    folder + "temporal_all", 
    arr_p3d_sim=arr_p3d_sim, 
    arr_p3d_emu=arr_p3d_emu, 
    arr_p1d_sim=arr_p1d_sim, 
    arr_p1d_emu=arr_p1d_emu,
    params_sim=params_sim,
    params_emu=params_emu
)

# %% [markdown]
# #### The following only for the central simulation!!!

# %%
# folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"
folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures_new/"
# fil = np.load(folder + "temporal_central.npz")
fil = np.load(folder + "temporal_all.npz")
arr_p3d_sim=fil["arr_p3d_sim"]
arr_p3d_emu=fil["arr_p3d_emu"]
arr_p1d_sim=fil["arr_p1d_sim"]
arr_p1d_emu=fil["arr_p1d_emu"]
params_sim=fil["params_sim"]
params_emu=fil["params_emu"]

# %%
import matplotlib.cm as cm
# Generate a colormap from 'rainbow'
cmap = cm.get_cmap('rainbow', arr_p3d_sim.shape[1])

# %%
fig, ax = plt.subplots(5, 1, sharex=True, figsize=(8, 10))
mu_use = [0, 1, 2, 3]
for i0 in mu_use:
    for iz in range(arr_p3d_sim.shape[1]):
        y = arr_p3d_emu[0, iz, :, i0]/arr_p3d_sim[0, iz, :, i0]-1
        if (i0 == 0) and (iz < 4):
            label = "z="+str(zs[iz])
        elif (i0 == 1) and (iz >= 4) and (iz < 8):
            label = "z="+str(zs[iz])
        elif (i0 == 2) and (iz >= 8) and (iz < 12):
            label = "z="+str(zs[iz])
        else:
            label = ""
        _ = np.isfinite(knew[:, i0])
        ax[i0].plot(knew[_, i0], y[_], color=cmap(iz), label=label)

i0 = 4
for iz in range(arr_p3d_sim.shape[1]):
    y = arr_p1d_emu[0, iz, :]/arr_p1d_sim[0, iz, :]-1
    ax[i0].plot(out["k1d_Mpc"], y, color=cmap(iz))


ftsize=20

ax[0].set_xscale("log")
for ii in range(len(mu_use)):
    ax[ii].axhline(color="k", linestyle=":")
    ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)
    ax[ii].set_ylim(-0.25, 0.25)
    ax[ii].axhline(0.1, color="k", linestyle="--")
    ax[ii].axhline(-0.1, color="k", linestyle="--")
    ax[ii].axvline(5, color="k", linestyle="--")
    ax[ii].set_ylabel(r"$P_\mathrm{3D}^\mathrm{emu}/P_\mathrm{3D}^\mathrm{sim}-1$", fontsize=ftsize)
    if ii < 3:
        ax[ii].legend(loc="lower right", ncols=4, fontsize=ftsize-4)

_x = 1.6
_y = 0.15
ax[0].text(_x, _y, r"$0.0\leq\mu<0.25$", fontsize=ftsize)
ax[1].text(_x, _y, r"$0.25\leq\mu<0.5$", fontsize=ftsize)
ax[2].text(_x, _y, r"$0.5\leq\mu<0.75$", fontsize=ftsize)
ax[3].text(_x, _y, r"$0.75\leq\mu<1.0$", fontsize=ftsize)
ax[3].set_xlabel(r"$k[\mathrm{Mpc}^{-1}]$", fontsize=ftsize)

ii = 4
ax[ii].set_ylim(-0.06, 0.06)
ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)
ax[ii].axhline(0.01, color="k", linestyle="--")
ax[ii].axhline(-0.01, color="k", linestyle="--")
ax[ii].axvline(4, color="k", linestyle="--")
ax[ii].set_ylabel(r"$P_\mathrm{1D}^\mathrm{emu}/P_\mathrm{1D}^\mathrm{sim}-1$", fontsize=ftsize)
ax[ii].set_xlabel(r"$k_\parallel[\mathrm{Mpc}^{-1}]$", fontsize=ftsize)


# folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"
folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures_new/"
plt.tight_layout()
plt.savefig(folder + "central_z.pdf")
plt.savefig(folder + "central_z.png")

# %% [markdown]
# ### Save data for zenodo

# %%
outs = {}


for i0 in mu_use:
    for iz in range(arr_p3d_sim.shape[1]):
        y = arr_p3d_emu[0, iz, :, i0]/arr_p3d_sim[0, iz, :, i0]-1
        label = "z="+str(zs[iz])
        _ = np.isfinite(knew[:, i0])

        xsave = knew[_, i0]
        ysave = y[_]
        outs["panel"+str(i0)+"_"+label+"_x"] = xsave
        outs["panel"+str(i0)+"_"+label+"_y"] = ysave

i0 = 4
for iz in range(arr_p3d_sim.shape[1]):
    xsave = out["k1d_Mpc"]
    ysave = arr_p1d_emu[0, iz, :]/arr_p1d_sim[0, iz, :]-1

    outs["panel"+str(i0)+"_"+label+"_x"] = xsave
    outs["panel"+str(i0)+"_"+label+"_y"] = ysave



# %%
import forestflow
path_forestflow= os.path.dirname(forestflow.__path__[0]) + "/"
folder = path_forestflow + "data/figures_machine_readable/"
np.save(folder + "fig5", outs)

# %%
res = np.load(folder + "fig5.npy", allow_pickle=True).item()
res.keys()

# %%

# %%

# %%
for ii in range(2):
    rat = params_emu[:, ii] / params_sim[:, ii] - 1
    y = np.percentile(rat, [50, 16, 84])
    print(y[0]*100, 0.5*(y[2] - y[1])*100, np.std(rat)*100)

# %%
kaiser_emu = np.zeros((params_emu.shape[1], 2))
kaiser_sim = np.zeros((params_emu.shape[1], 2))
kaiser_emu[:, 0] = params_emu[0, :, 0]**2
kaiser_emu[:, 1] = params_emu[0, :, 0]**2*(1+params_emu[0, :, 2])**2
kaiser_sim[:, 0] = params_sim[0, :, 0]**2
kaiser_sim[:, 1] = params_sim[0, :, 0]**2*(1+params_sim[0, :, 2])**2

for ii in range(2):
    rat = kaiser_emu[:, ii] / kaiser_sim[:, ii] - 1
    y = np.percentile(rat, [50, 16, 84])
    print(y[0]*100, 0.5*(y[2] - y[1])*100, np.std(rat)*100)

# %%
_ = np.isfinite(knew) & (knew > 0.3) & (knew < 5)
rat = arr_p3d_emu[0, :, _]/arr_p3d_sim[0, :, _] - 1
y = np.percentile(rat, [50, 16, 84])
print(y[0]*100, 0.5*(y[2]-y[1])*100, np.std(rat)*100)

# %%
_ = np.isfinite(k1d_Mpc) & (k1d_Mpc < 4) & (k1d_Mpc > 0)
rat = arr_p1d_emu[0, :, _]/arr_p1d_sim[0, :, _] - 1
y = np.percentile(rat, [50, 16, 84])
print(y[0]*100, 0.5*(y[2]-y[1])*100, np.std(rat)*100)

# %% [markdown]
# ### Now plot all

# %%
folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"
# fil = np.load(folder + "temporal_central.npz")
fil = np.load(folder + "temporal_all.npz")
arr_p3d_sim=fil["arr_p3d_sim"]
arr_p3d_emu=fil["arr_p3d_emu"]
arr_p1d_sim=fil["arr_p1d_sim"]
arr_p1d_emu=fil["arr_p1d_emu"]
params_sim=fil["params_sim"]
params_emu=fil["params_emu"]

# %%
rat_p3d = arr_p3d_emu/arr_p3d_sim - 1
rat_p1d = arr_p1d_emu/arr_p1d_sim - 1

# %%
# folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"
folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures_new/"

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
        kmax_3d_fit=kmax_3d_fit
    )

# %%
savename = folder + "test_cosmo/test_cosmo_P1D"
for ext in [".png", ".pdf"]:
    plot_p1d_test_sims(
        sim_labels,
        out["k1d_Mpc"],
        rat_p1d,
        savename=savename+ext,
        fontsize=20,
        kmax_1d_fit=kmax_1d_fit
    );

# %% [markdown]
# ### Save data for zenodo

# %%
conv = {}
conv["blue"] = 0
conv["orange"] = 1
conv["green"] = 2
conv["red"] = 3
outs = {}

med_rat_p3d = np.median(rat_p3d, axis=1)
med_rat_p1d = np.median(rat_p1d, axis=1)

for jj in range(med_rat_p3d.shape[0]):
    for key in conv.keys():
        ii = conv[key]
        
        outs["p3d_panel" + str(jj) + "_" + key + "_x"] = knew[:, ii]
        outs["p3d_panel" + str(jj) + "_" + key + "_y"] = med_rat_p3d[jj, :, ii]
    
    outs["p1d_panel" + str(jj) + "_x"] = out["k1d_Mpc"]
    outs["p1d_panel" + str(jj) + "_y"] = med_rat_p1d[jj]


# %%
import forestflow
path_forestflow= os.path.dirname(forestflow.__path__[0]) + "/"
folder = path_forestflow + "data/figures_machine_readable/"
np.save(folder + "fig7", outs)

# %%
res = np.load(folder + "fig7.npy", allow_pickle=True).item()
res.keys()

# %%
