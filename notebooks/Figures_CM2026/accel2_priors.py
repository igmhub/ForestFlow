# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
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

from forestflow.priors.paper.load import load_map_igm_p3d
from forestflow.priors.paper.set_samples import load_k_mu_accel2
from forestflow.priors.paper.plots import plot_p3d_validation, plot_bias_beta_zev_val

# %%
knew3d, munew3d, data_accel2 = load_k_mu_accel2()
data = load_map_igm_p3d(lab_sample="accel2", emu_type = "legacy")

# %%
data.keys()

# %%
# for key in data.keys():
#     try:
#         print(key, data[key].keys())
#     except AttributeError:
#         print(data[key])

# %% [markdown]
# Quick check forestflow P1D agrees with data

# %%
z_use = 2.6
ind_accel = np.argmin(np.abs(data_accel2["z"] - z_use))
ind_data = np.argmin(np.abs(data["zs"] - z_use))

k1d_Mpc = data_accel2["k1d_Mpc"]
ind = np.argwhere(k1d_Mpc < 4)[:,0]
k1d_Mpc = k1d_Mpc[ind]
plt.plot(k1d_Mpc, k1d_Mpc * data_accel2["p1d_Mpc"][ind_accel, ind], "k", label="accel2")
for ii in range(100):
    plt.plot(k1d_Mpc, k1d_Mpc * data["forest_out"]["p1d"][ii, ind_data, ind], "C1", alpha=0.1)
plt.plot(k1d_Mpc, k1d_Mpc * np.median(data["forest_out"]["p1d"][:, ind_data, ind], axis=0), "C2", label="ForestFlow, median")
plt.xlabel(r"$k_\parallel$ [Mpc$^{-1}$]")
plt.ylabel(r"$k_\parallel P_\mathrm{1D}$")
plt.title(f"z={z_use}")
plt.legend()

# %%
plot_p3d_validation(knew3d, munew3d, data, data_accel2)

# %%
plot_bias_beta_zev_val(data)

# %% [markdown]
# Arinyo parameters at z=2.33 with the legacy emulator (mpg_hypercube.pt cINN), for
# comparison (columns: mean, std, 5th and 95th percentiles).
#
# DESI DR1 P1D chain:
# ```
# bias -0.124 0.007 -0.135 -0.113
# beta 1.417 0.044 1.346 1.49
# q1 0.282 0.055 0.193 0.369
# kvav 0.554 0.049 0.485 0.639
# av 0.426 0.048 0.353 0.51
# bv 1.674 0.023 1.642 1.716
# kp 10.817 0.388 10.349 11.529
# q2 0.27 0.059 0.182 0.375
# ```
#
# ACCEL2 DR1-like mock:
# ```
# bias -0.109 0.009 -0.122 -0.094
# beta 1.669 0.079 1.548 1.805
# q1 0.281 0.046 0.205 0.357
# kvav 0.422 0.046 0.342 0.496
# av 0.272 0.059 0.169 0.355
# bv 1.656 0.03 1.608 1.708
# kp 11.833 0.531 11.066 12.709
# q2 0.12 0.042 0.062 0.198
# ```

# %% [markdown]
# Same table with the current emulator

# %%
for lab_sample in ["desi", "accel2"]:
    dd = data if lab_sample == "accel2" else load_map_igm_p3d(lab_sample="desi")
    iz = np.argmin(np.abs(dd["zs"] - 2.33))
    print(lab_sample)
    for par in ["bias", "beta", "q1", "kvav", "av", "bv", "kp", "q2"]:
        sign = -1 if par == "bias" else 1
        vals = sign * np.abs(dd["forest_out"][par][:, iz])
        p5, p95 = np.nanpercentile(vals, [5, 95])
        print(par, np.round(np.nanmean(vals), 3), np.round(np.nanstd(vals), 3), np.round(p5, 3), np.round(p95, 3))
    print()

# %%
# systematic error of the model: difference between the model (Arinyo parameters
# mapped from the Accel2 P1D) and the Arinyo parameters fitted directly to the
# Accel2 P3D, as in plot_bias_beta_zev_val. Relative difference of the median,
# |model/sim - 1|, at the simulation redshifts covered by the model
from forestflow.priors.paper.plots import ACCEL2_ARINYO_FITS

# simulation label: model label
sys_pars = {
    "bias_delta": "bias_delta",
    "beta": "beta",
    "q1": "q1",
    "q2": "q2",
    "k_v": "kvav",
    "a_v": "av",
    "b_v": "bv",
    "k_p": "kp",
}

z_sim = ACCEL2_ARINYO_FITS["z"]
ind_sim = np.argwhere(z_sim >= data["zs"].min() - 1e-3)[:, 0]
iz_model = [np.argmin(np.abs(data["zs"] - z)) for z in z_sim[ind_sim]]
assert np.allclose(data["zs"][iz_model], z_sim[ind_sim]), "redshift grids differ"

sys_shift = {"z": z_sim[ind_sim]}
sys_err_sim = {"z": z_sim[ind_sim]}
for par_sim, par in sys_pars.items():
    if par == "kvav":
        # the simulation fits quote k_v, the model kvav = k_v**av
        val = data["forest_out"]["kvav"] ** (1 / data["forest_out"]["av"])
    else:
        val = data["forest_out"][par]
    med_model = np.nanmedian(val[:, iz_model], axis=0)
    val_sim = ACCEL2_ARINYO_FITS[par_sim]["value"][ind_sim]
    err_sim = ACCEL2_ARINYO_FITS[par_sim]["error"][ind_sim]
    sys_shift[par_sim] = np.abs(med_model / val_sim - 1)
    # relative error of the simulation fit, to compare with the shift
    sys_err_sim[par_sim] = np.abs(err_sim / val_sim)
    print(par_sim, np.round(sys_shift[par_sim], 3), np.round(sys_err_sim[par_sim], 3))

# %% [markdown]
# New GP
#
# ```
# Arinyo mapping from emulator: forest_mpg_gp
# desi
# bias -0.118 0.006 -0.128 -0.11
# beta 1.599 0.054 1.509 1.685
# q1 0.384 0.105 0.196 0.54
# kvav 0.575 0.049 0.499 0.658
# av 0.352 0.049 0.271 0.433
# bv 1.555 0.018 1.531 1.589
# kp 10.169 0.28 9.803 10.683
# q2 0.275 0.091 0.144 0.434
#
# accel2
# bias -0.103 0.007 -0.113 -0.091
# beta 1.83 0.105 1.67 2.018
# q1 0.362 0.095 0.206 0.524
# kvav 0.46 0.045 0.388 0.536
# av 0.235 0.052 0.15 0.322
# bv 1.532 0.025 1.492 1.574
# kp 11.203 0.403 10.617 11.922
# q2 0.124 0.066 0.023 0.239
#
# ```

# %% [markdown]
#

# %% [markdown]
# Systematic shift as a function of redshift, with the relative error of the
# simulation fits and a linear fit, to check whether it can be extrapolated

# %%
fig, ax = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
ax = ax.reshape(-1)
zfit = np.linspace(2.0, 4.4, 50)
for ii, par_sim in enumerate(sys_pars):
    fit = np.polyfit(sys_shift["z"], sys_shift[par_sim], 1)
    ax[ii].errorbar(
        sys_shift["z"], sys_shift[par_sim], sys_err_sim[par_sim], fmt="o", label="Model vs sim"
    )
    ax[ii].plot(zfit, np.polyval(fit, zfit), "C1-", label="Linear fit")
    ax[ii].axhline(0, color="k", ls=":", lw=1)
    ax[ii].set_title(par_sim)
for ii in range(4, 8):
    ax[ii].set_xlabel(r"$z$")
ax[0].set_ylabel(r"$|x_\mathrm{model}/x_\mathrm{sim} - 1|$")
ax[4].set_ylabel(r"$|x_\mathrm{model}/x_\mathrm{sim} - 1|$")
ax[0].legend()
plt.tight_layout()

# %%
np.save("int_data_figs/legacy_emu/accel2_priors_sys_shift.npy", sys_shift)

# %%
