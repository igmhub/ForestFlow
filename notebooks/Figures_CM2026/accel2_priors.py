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
data = load_map_igm_p3d(lab_sample="accel2")

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
        p5, p95 = np.percentile(vals, [5, 95])
        print(par, np.round(vals.mean(), 3), np.round(vals.std(), 3), np.round(p5, 3), np.round(p95, 3))
    print()
