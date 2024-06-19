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
# # Redshift evolution of Arinyo parameters

# %%
# %load_ext autoreload
# %autoreload 2

import sys
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

from forestflow.archive import GadgetArchive3D
from forestflow.plots_v0 import plot_test_p3d
from forestflow.P3D_cINN import P3DEmulator
from forestflow.model_p3d_arinyo import ArinyoModel
from forestflow import model_p3d_arinyo
from forestflow.utils import transform_arinyo_params, params_numpy2dict


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
# folder_interp = path_program+"/data/plin_interp/"

Archive3D = GadgetArchive3D(
    base_folder=path_program[:-1],
    folder_data=folder_lya_data,
    force_recompute_plin=False,
    average="both",
)
print(len(Archive3D.training_data))


# %% [markdown]
# ## LOAD EMULATOR

# %%
training_type = "Arinyo_min"
model_path=path_program + "/data/emulator_models/mpg_hypercube.pt"

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
    # Nrealizations=10000,
    Nrealizations=1000,
    training_type=training_type,
    model_path=model_path,
    # save_path=model_path,
)


# %% [markdown]
# ## LOAD SIMULATIONS

# %%
sim_label = "mpg_central"
central = Archive3D.get_testing_data(
    sim_label, force_recompute_plin=False
)

sim_label = "mpg_seed"
seed = Archive3D.get_testing_data(
    sim_label, force_recompute_plin=False
)

# get average of both
list_merge = []
zlist = []
par_merge = ["mF", "T0", "gamma", "sigT_Mpc", "kF_Mpc"]
for ii in range(len(central)):
    _cen = central[ii]
    _seed = seed[ii]
    zlist.append(_cen["z"])

    tar = _cen.copy()
    for par in par_merge:
        tar[par] = 0.5 * (_cen[par] + _seed[par])
        
    tar["p1d_Mpc"] = (_cen["mF"]**2 * _cen["p1d_Mpc"] + _seed["mF"]**2 * _seed["p1d_Mpc"]) / tar["mF"]**2 / 2
    tar["p3d_Mpc"] = (_cen["mF"]**2 * _cen["p3d_Mpc"] + _seed["mF"]**2 * _seed["p3d_Mpc"]) / tar["mF"]**2 / 2

    # print(cen["mF"], seed["mF"], tar["mF"])
    list_merge.append(tar)

# %%
from forestflow.utils import params_numpy2dict_minimizerz

def paramz_to_paramind(z, paramz):
    paramind = []
    for ii in range(len(z)):
        param = {}
        for key in paramz:
            param[key] = 10 ** np.poly1d(paramz[key])(z[ii])
        paramind.append(param)
    return paramind

file = path_program + "/data/best_arinyo/minimizer/fit_sim_label_combo_kmax3d_5_kmax1d_4.npz"
data = np.load(file, allow_pickle=True)
# best_params = paramz_to_paramind(zlist, data["best_params"].item())

for ii in range(len(list_merge)):
    list_merge[ii]["Arinyo_min"] = data["best_params"][ii]

# %% [markdown]
# ### Plot ratios sims

# %%
zs = 3
central_z = [d for d in central if d["z"] == zs][0]
seed_z = [d for d in seed if d["z"] == zs][0]
booth_z = [d for d in list_merge if d["z"] == zs][0]

# %%
from forestflow.rebin_p3d import p3d_allkmu, get_p3d_modes, p3d_rebin_mu
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

n_mubins = 4
kmax = 4
kmax_fit = 3

k3d_Mpc = central_z['k3d_Mpc']
mu3d = central_z['mu3d']

kmu_modes = get_p3d_modes(kmax)

mask_3d = k3d_Mpc[:, 0] <= kmax

mask_1d = central_z['k_Mpc'] < kmax

k1d_Mpc = central_z['k_Mpc'][mask_1d]
p1d_central = central_z['p1d_Mpc'][mask_1d]
p1d_seed = seed_z['p1d_Mpc'][mask_1d]
p1d_both = booth_z['p1d_Mpc'][mask_1d]

# rebin

_ = p3d_rebin_mu(k3d_Mpc[mask_3d], mu3d[mask_3d], central_z['p3d_Mpc'][mask_3d], kmu_modes, n_mubins=n_mubins)
knew, munew, p3d_central, mu_bins = _

_ = p3d_rebin_mu(k3d_Mpc[mask_3d], mu3d[mask_3d], seed_z['p3d_Mpc'][mask_3d], kmu_modes, n_mubins=n_mubins)
knew, munew, p3d_seed, mu_bins = _

_ = p3d_rebin_mu(k3d_Mpc[mask_3d], mu3d[mask_3d], booth_z['p3d_Mpc'][mask_3d], kmu_modes, n_mubins=n_mubins, return_modes=True)
knew, munew, p3d_both, mu_bins, n_modes = _

# %%
folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"
# for ii in range(len(list_central)):

ftsize = 20
fig, arx = plt.subplots(2, 1, figsize=(8, 6))
lw = 2
ndiff = 2
corr_factor = np.sqrt(ndiff)
ax = arx[0]

for ii in range(4):
    col = "C"+str(ii)

    _ = np.isfinite(knew[:, ii])
    ax.plot(knew[_, ii], p3d_central[_, ii]/p3d_both[_, ii]-1, col+"-", lw=lw, alpha=0.7)
    ax.plot(knew[_, ii], p3d_seed[_, ii]/p3d_both[_, ii]-1, col+"--", lw=lw, alpha=0.7)
    ax.plot(knew[_, ii], np.sqrt(2/n_modes[_,ii])/corr_factor, col+":")
    ax.plot(knew[_, ii], -np.sqrt(2/n_modes[_,ii])/corr_factor, col+":")
    # plt.plot(x[mask], cen["p1d_Mpc"][mask]/tar["p1d_Mpc"][mask])
    # plt.plot(x[mask], seed["p1d_Mpc"][mask]/tar["p1d_Mpc"][mask])
ax.set_ylim(-0.1, 0.3)
ax.set_xscale("log")
ax.set_ylabel("Residual", fontsize=ftsize)
ax.set_xlabel(r"$k$ [Mpc$^{-1}$]", fontsize=ftsize)

ax.tick_params(axis="both", which="major", labelsize=ftsize)
ax.axhline(y=0, color="k", ls=":", lw=2)

hand = []
for i in range(4):
    if i != 3:
        lab = str(mu_bins[i]) + r"$\leq\mu<$" + str(mu_bins[i + 1])
    else:
        lab = str(mu_bins[i]) + r"$\leq\mu\leq$" + str(mu_bins[i + 1])
    col = "C" + str(i)
    hand.append(mpatches.Patch(color=col, label=lab))
legend1 = ax.legend(
    fontsize=ftsize - 2, loc="upper right", handles=hand, ncols=1
)

line1 = Line2D(
    [0], [0], label="Gaussian error", color="k", ls=":", linewidth=2
)
line2 = Line2D([0], [0], label="Central", color="k", ls="-", linewidth=2)
line3 = Line2D([0], [0], label="Seed", color="k", ls="--", linewidth=2)
hand = [line1, line2, line3]
ax.legend(fontsize=ftsize - 2, loc="upper right", handles=hand, ncols=1)
# ax.add_artist(legend1)


ax = arx[1]


ax.plot(k1d_Mpc, p1d_central/p1d_both-1, "C0-", lw=lw, alpha=0.7)
ax.plot(k1d_Mpc, p1d_seed/p1d_both-1, "C1--", lw=lw, alpha=0.7)
# ax.plot(k1d_Mpc, k1d_Mpc[:]*0+np.sqrt(2/500**2), "C1--", lw=lw, alpha=0.7)

ax.set_xscale("log")
ax.set_ylabel("Residual", fontsize=ftsize)
ax.set_xlabel(r"$k_\parallel$ [Mpc$^{-1}$]", fontsize=ftsize)

ax.tick_params(axis="both", which="major", labelsize=ftsize)
ax.axhline(y=0, color="k", ls=":", lw=2)

# line1 = Line2D(
#     [0], [0], label="Gaussian error", color="C2", ls=":", linewidth=2
# )
line2 = Line2D([0], [0], label="Central", color="C0", ls="-", linewidth=2)
line3 = Line2D([0], [0], label="Seed", color="C1", ls="--", linewidth=2)
hand = [line2, line3]
ax.legend(fontsize=ftsize - 2, loc="upper left", handles=hand, ncols=1)
# ax.add_artist(legend1)


plt.tight_layout()
plt.savefig(folder+"cvariance.pdf")

# %%

# %%

# %% [markdown]
# ### Continue

# %%
Arinyo_coeffs_central = np.array(
    [list(central[i][training_type].values()) for i in range(len(central))]
)

Arinyo_coeffs_seed = np.array(
    [list(seed[i][training_type].values()) for i in range(len(seed))]
)

Arinyo_coeffs_merge = np.array(
    [list(list_merge[i][training_type].values()) for i in range(len(list_merge))]
)


# %%
Arinyo_central = []
Arinyo_seed = []
Arinyo_merge = []
for ii in range(Arinyo_coeffs_central.shape[0]):
    dict_params = params_numpy2dict(Arinyo_coeffs_central[ii])
    new_params = transform_arinyo_params(dict_params, central[ii]["f_p"])
    Arinyo_central.append(new_params)
    
    dict_params = params_numpy2dict(Arinyo_coeffs_seed[ii])
    new_params = transform_arinyo_params(dict_params, seed[ii]["f_p"])
    Arinyo_seed.append(new_params)

    dict_params = params_numpy2dict(Arinyo_coeffs_merge[ii])
    new_params = transform_arinyo_params(dict_params, list_merge[ii]["f_p"])
    Arinyo_merge.append(new_params)

# %% [markdown]
# ## LOOP OVER REDSHIFTS PREDICTING THE ARINYO PARAMETERS

# %%
z_central = [d["z"] for d in central]
z_central

# %%
Arinyo_emu = []
Arinyo_emu_std = []

for iz, z in enumerate(z_central):
    test_sim_z = [d for d in central if d["z"] == z]
    out = p3d_emu.evaluate(
        emu_params=test_sim_z[0],
        natural_params=True,
        Nrealizations=10000,
    )
    Arinyo_emu.append(out["coeffs_Arinyo"])
    Arinyo_emu_std.append(out["coeffs_Arinyo_std"])


# %% [markdown]
# ## PLOT

# %%
from forestflow.plots.params_z import plot_arinyo_z, plot_forestflow_z

# %%
folder_fig = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"

# %%
bias_cen = np.zeros((len(Arinyo_central), 2))
bias_seed = np.zeros((len(Arinyo_central), 2))
for ii in range(len(Arinyo_central)):
    bias_cen[ii, 0] = Arinyo_central[ii]["bias"]
    bias_cen[ii, 1] = Arinyo_central[ii]["bias_eta"]
    bias_seed[ii, 0] = Arinyo_seed[ii]["bias"]
    bias_seed[ii, 1] = Arinyo_seed[ii]["bias_eta"]

# %%
for ii in range(2):
    plt.plot(z_central, bias_cen[:, ii]/bias_seed[:, ii]-1)

# %%
# for ii in range(len(Arinyo_emu)):
#     print(ii, Arinyo_emu[ii]["kv"])
#     Arinyo_emu[ii]["kv"] = Arinyo_emu[ii]["kv"]**Arinyo_emu[ii]["av"]
#     print(Arinyo_emu[ii]["kv"])

# %%
plot_arinyo_z(z_central, Arinyo_central, Arinyo_seed, Arinyo_merge, folder_fig=folder_fig, ftsize=20)

# %%
plot_forestflow_z(z_central, Arinyo_central, Arinyo_emu, Arinyo_emu_std, folder_fig=folder_fig, ftsize=20)

# %%
Arinyo_central[0].keys()

# %%
kaiser_cen = np.zeros((len(Arinyo_central), 2))
kaiser_seed = np.zeros((len(Arinyo_central), 2))
kaiser_merge = np.zeros((len(Arinyo_central), 2))
for iz in range(len(Arinyo_central)):
    kaiser_cen[iz, 0] = (Arinyo_central[iz]["bias"])**2
    kaiser_cen[iz, 1] = (Arinyo_central[iz]["bias"] + Arinyo_central[iz]["bias_eta"])**2
    kaiser_seed[iz, 0] = (Arinyo_seed[iz]["bias"])**2
    kaiser_seed[iz, 1] = (Arinyo_seed[iz]["bias"] + Arinyo_seed[iz]["bias_eta"])**2
    kaiser_merge[iz, 0] = (Arinyo_merge[iz]["bias"])**2
    kaiser_merge[iz, 1] = (Arinyo_merge[iz]["bias"] + Arinyo_merge[iz]["bias_eta"])**2

# %%
for ii in range(2):
    # y = np.concatenate([kaiser_cen[:,ii]/kaiser_merge[:,ii] - 1, kaiser_seed[:,ii]/kaiser_merge[:,ii] - 1])
    y = (kaiser_cen[:,ii]/kaiser_seed[:,ii] - 1)
    s_pred = np.percentile(y, [16, 50, 84])
    std = 0.5 * (s_pred[2] - s_pred[0]) * 100
    print(ii, s_pred[1]* 100, std)


# %%
0 1.513956969208241 0.9816090063136884
1 0.7647390340765448 1.8861762107728053
