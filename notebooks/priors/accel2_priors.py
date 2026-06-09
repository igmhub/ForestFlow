# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: lace
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Validation with ACCEL2

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

from forestflow.priors_paper.load import load_map_igm_p3d
from forestflow.priors_paper.set_samples import load_k_mu_accel2
from forestflow.priors_paper.all_plots import plot_p3d_validation, plot_bias_beta_zev_val

np.__version__

# %%
knew3d, munew3d, data_accel2 = load_k_mu_accel2()
data = load_map_igm_p3d(lab_sample="accel2")

# %%
# old model, not needed
# folder = "/home/jchaves/Proyectos/projects/lya/ForestFlow/notebooks/priors/int_data_figs/"
# data = np.load(folder + "arinyo_from_p1d_accel2.npy", allow_pickle=True).item()

# %% [markdown]
# Quick check forestflow P1D agrees with data

# %%
z_use = 2.6
ind_accel = np.argmin(np.abs(data_accel2["z"] - z_use))
ind_data = np.argmin(np.abs(data["zs"] - z_use))

k1d_Mpc = data_accel2["k1d_Mpc"]
ind = np.argwhere(k1d_Mpc < 4)[:,0]
k1d_Mpc = k1d_Mpc[ind]
plt.plot(k1d_Mpc, k1d_Mpc * data_accel2["p1d_Mpc"][ind_accel, ind], label="accel2")
for ii in range(100):
    plt.plot(k1d_Mpc, k1d_Mpc * data["forest_out"]["p1d"][ii, ind_data, ind], "C1", label="Arinyo", alpha=0.1)
plt.plot(k1d_Mpc, k1d_Mpc * np.median(data["forest_out"]["p1d"][:, ind_data, ind], axis=0), "C2", label="Arinyo")
# plt.ylim(1e-3, 0.6)
# plt.xscale("log")

# %%
data["forest_out"]["p1d"].shape

# %%
plot_p3d_validation(knew3d, munew3d, data, data_accel2)

# %%

# %%
plot_bias_beta_zev_val(data)

# %%
25/0.675

# %%

# %% [markdown]
# OLD figure

# %%
fig, ax = plt.subplots(len(emulator.Arinyo_params), 1, sharex=True, figsize=(8, 16))

print("par", "mean", "std", "min", "max")

for ii, par in enumerate(emulator.Arinyo_params):
    if par == "bias":
        sing = -1
    else:
        sing = 1
    percen = np.percentile(sing * out_ari[par], [16, 84], axis=0)
    ax[ii].fill_between(zs, percen[0], percen[1], label="ACCEL2 DR1-like mock")
    percen = np.percentile(sing * out_ari[par], [5, 95], axis=0)
    cen = np.mean(sing * out_ari[par][:, 1])
    std = np.std(sing * out_ari[par][:, 1])

    # ax[ii].errorbar(
    #     params_accel2["z"],
    #     params_accel2[par]["value"],
    #     params_accel2[par]["error"],
    #     color="C1",
    #     label="ACCEL2"
    # )

    print(
        par,
        np.round(cen, 3),
        np.round(std, 3),
        np.round(np.min(percen[0, 1]), 3),
        np.round(np.max(percen[1, 1]), 3),
    )
    ax[ii].set_ylabel(par)
    # print(par, np.mean(out_ari[par])
ax[-1].set_xlabel(r"$z$")
plt.tight_layout()
plt.savefig("Arinyo_with_z_accel2.pdf")
plt.savefig("Arinyo_with_z_accel2.png")

# %% [markdown]
# desi

# %%
par mean std min max
bias -0.124 0.007 -0.135 -0.113
beta 1.417 0.044 1.346 1.49
q1 0.282 0.055 0.193 0.369
kvav 0.554 0.049 0.485 0.639
av 0.426 0.048 0.353 0.51
bv 1.674 0.023 1.642 1.716
kp 10.817 0.388 10.349 11.529
q2 0.27 0.059 0.182 0.375

# %% [markdown]
# accel2

# %%
bias -0.109 0.009 -0.122 -0.094
beta 1.669 0.079 1.548 1.805
q1 0.281 0.046 0.205 0.357
kvav 0.422 0.046 0.342 0.496
av 0.272 0.059 0.169 0.355
bv 1.656 0.03 1.608 1.708
kp 11.833 0.531 11.066 12.709
q2 0.12 0.042 0.062 0.198

# %%

# %% [markdown]
# # Check compatibility forestflow and cup1d

# %%
from forestflow.priors_paper.set_samples import set_input_process_p1d_chain
pip, chain, d2star, nstar, zs, zeff = set_input_process_p1d_chain("accel2")

folder = "/home/jchaves/Proyectos/projects/lya/data/accel2/chains/chain_1/"
ln_prob = np.load(folder + "lnprob.npy").reshape(-1)


chain = np.load(folder + "chain.npy").reshape(-1, 53)

ind0 = np.argmax(ln_prob)

# %% [markdown]
# Plot cup1d and data

# %%
p1d = pip.fitter.like.get_p1d_kms(values=chain[ind0])[0]

k_kms = pip.fitter.like.data.k_kms

knew3d, munew3d, data_accel2 = load_k_mu_accel2()


ind = data_accel2["k1d_Mpc"] < 4

plt.plot(data_accel2["k1d_Mpc"][ind], data_accel2["k1d_Mpc"][ind] * data_accel2["p1d_Mpc"][1, ind], label="accel2")
plt.plot(data_accel2["k1d_Mpc"][ind], data_accel2["k1d_Mpc"][ind] * P1D_Mpc_forest[ind], label="forest")


plt.plot(
    k_kms[2] * pip.fitter.like.theory.fid_cosmo["M_of_zs"][2],
    k_kms[2] * p1d[2],
    label="accel2 emulator",
)
plt.yscale("log")
plt.xscale("log")

# %% [markdown]
# Input forestflow

# %%
zs = 2.6
chain_params = pip.fitter.like.parameters_from_sampling_point(chain[95703])
mF = pip.fitter.like.theory.model_igm.models[
    "F_model"
].get_mean_flux(zs, like_params=chain_params)

gamma = pip.fitter.like.theory.model_igm.models[
    "T_model"
].get_gamma(zs, like_params=chain_params)

sigT_Mpc = pip.fitter.like.theory.model_igm.models["T_model"].get_sigT_kms(
    zs, like_params=chain_params
)/ pip.fitter.like.theory.fid_cosmo["M_of_zs"][2]

kF_Mpc = pip.fitter.like.theory.model_igm.models["P_model"].get_kF_kms(
    zs, like_params=chain_params
) * pip.fitter.like.theory.fid_cosmo["M_of_zs"][2]

As = chain_params[0].value_from_cube(chain[95703, 0])
ns = chain_params[1].value_from_cube(chain[95703, 1])


# %%

from lace.cosmo import cosmology, rescale_cosmology

class_planck = cosmology.Cosmology(cosmo_label="Planck18")
class_new = rescale_cosmology.RescaledCosmology(
    fid_cosmo=class_planck,
    new_params_dict={"As": As, "ns": ns},
)

# %%
linP_params = class_new.get_linP_Mpc_params(2.6, kp_Mpc=0.7)
Delta2_p = linP_params["Delta2_p"]
n_p = linP_params["n_p"]

# %%
import forestflow
from forestflow.P3D_cINN import P3DEmulator
emulator = P3DEmulator(
    model_path=os.path.join(
        os.path.dirname(forestflow.__path__[0]),
        "data",
        "emulator_models",
        "forest_mpg",
    )
)

# %%
input_emu = {}
input_emu["mF"] = mF
input_emu["gamma"] = gamma
input_emu["sigT_Mpc"] = sigT_Mpc
input_emu["kF_Mpc"] = kF_Mpc
input_emu["Delta2_p"] = Delta2_p
input_emu["n_p"] = n_p

# %%
par_ari = emulator.predict_Arinyos(emu_params=input_emu)

# %%
from forestflow.model_p3d_arinyo import ArinyoModel

# %%
new_cosmo = {
    "H0": 67.66,
    "mnu": 0,
    "omch2": 0.119,
    "ombh2": 0.0224,
    "omk": 0,
    "As": As,
    "ns": ns,
    "nrun": 0.0,
    "pivot_scalar": 0.05,
    "w": -1.0,
}
fid_cosmo = cosmology.Cosmology(cosmo_params_dict=new_cosmo)
model_Arinyo = ArinyoModel(fid_cosmo)

# %%
P1D_Mpc_forest = model_Arinyo.P1D_Mpc(
    2.6,
    data_accel2["k1d_Mpc"],
    par_ari,
)

# %%
P1D_Mpc_forest.shape

# %%
pip.fitter.like.plot_p1d(values=chain[95703],residuals=True, plot_panels=True)

# %%
