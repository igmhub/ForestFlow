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
# # Get priors on Arinyo from DR1 fit
#
# - Get chain data from DESI DR1 fit
# - Evaluate ForestFlow to get Arinyo, P1D, and P3D for each point of the chain
# - Compare ForestFlow P1D with lace-mpg P1D
# - Get priors on the Arinyo parameters

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

from forestflow.priors_paper import all_plots, load

np.__version__

# %% [markdown]
# ## P1D chain -> ForestFlow -> P3D

# %% [markdown]
# #### Load data

# %%
dict_mapping = load.load_map_igm_p3d()

# %% [markdown]
# Table with output of P1D chain

# %%
all_plots.table_cosmo_igm(dict_mapping)

# %%
bao_data = load.load_BAO_data()

# %%
emulator = P3DEmulator(
    model_path=os.path.join(
        os.path.dirname(forestflow.__path__[0]), "data", "emulator_models", "forest_mpg"
    )
)

# %%
from forestflow.priors import get_arinyo_priors

z = 3.0
priors = get_arinyo_priors(z)
print(priors.keys())
priors["mean"]

# %%
dict_mapping.keys()

# %%
dict_mapping["forest_out"].keys()

# %%

# %%
bao_data["dr1_hsnr"].keys()

# %%
bao_data["dr1_hsnr"]["bias_delta"].shape

# %%
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 12))
ftsize = 20
zmax = 5

bao_plot = {}
for ii, key in enumerate(["dr1_hsnr", "dr2_hsnr"]):
    bao_plot[key] = {}
    bao_plot[key]["zeff"] = 2.33
    bao_plot[key]["label"] = "BAO DR" + str(ii+1)
    for lab in ["bias_delta", "bias_eta", "beta"]:
        bao_plot[key][lab] = {}
        bao_plot[key][lab]["mean"] = bao_data[key][lab].mean()
        bao_plot[key][lab]["std"] = bao_data[key][lab].std()


for ii, lab in enumerate(["bias_delta", "bias_eta", "beta"]):

    param = dict_mapping["forest_out"][lab]

    mean = np.mean(param, axis=0)
    percen = np.percentile(param, [16, 84], axis=0)

    _ = dict_mapping["zs"] < zmax
    ax[ii].fill_between(
        dict_mapping["zs"][_], percen[0][_], percen[1][_], alpha=0.5, label="P1D DR1"
    )

    for jj, key in enumerate(bao_plot):
        dumm = np.zeros(2)
        ax[ii].errorbar(
            dumm + bao_plot[key]["zeff"],
            dumm + bao_plot[key][lab]["mean"],
            dumm + bao_plot[key][lab]["std"],
            fmt=".",
            label=bao_plot[key]["label"],
            color="C" + str(jj + 1),
        )


ax[0].legend(fontsize=ftsize)

ax[0].set_ylabel(r"$b_\delta$", fontsize=ftsize)
ax[1].set_ylabel(r"$b_\eta$", fontsize=ftsize)
ax[2].set_ylabel(r"$\beta$", fontsize=ftsize)
ax[2].set_xlabel(r"$z$", fontsize=ftsize)

for ii in range(3):
    ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)

plt.tight_layout()
# plt.savefig("bias_beta_BAOvsP1D.png")
# plt.savefig("bias_beta_BAOvsP1D.pdf")

# %% [markdown]
# ### Get priors

# %%
out_ari["bias"].shape

# %%
fig, ax = plt.subplots(len(emulator.Arinyo_params), 1, sharex=True, figsize=(8, 16))

print("par", "mean", "std", "min", "max")

for ii, par in enumerate(emulator.Arinyo_params):
    if par == "bias":
        sing = -1
    else:
        sing = 1
    percen = np.percentile(sing * out_ari[par], [16, 84], axis=0)
    ax[ii].fill_between(zs, percen[0], percen[1])
    percen = np.percentile(sing * out_ari[par], [5, 95], axis=0)
    cen = np.mean(sing * out_ari[par][:, 1])
    std = np.std(sing * out_ari[par][:, 1])
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
plt.savefig("Arinyo_with_z.pdf")
plt.savefig("Arinyo_with_z.png")

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
