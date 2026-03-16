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

from getdist import plots
from lace.cosmo import cosmology

from forestflow.priors_paper import load, set_getdist, all_plots, importance

np.__version__

# %% [markdown]
# beta = biaseta * f(z) / biasdelta
#
# biaseta = beta * biasdelta / f(z)
#
# biaseta * f(z) * sig8(z) = beta * biasdelta * sig8(z)

# %% [markdown]
# ### Setup fiducial cosmology

# %%
zeff = 2.33
class_planck = cosmology.Cosmology(cosmo_label="Planck18_noBAO")
planck_sig8 = class_planck.get_sigma8(zeff)
planck_sig8_z0 = class_planck.get_sigma8(0)
planck_f = class_planck.get_growth_rate(zeff)
planck_fsig8 = planck_f * planck_sig8
print(planck_sig8_z0, planck_sig8, planck_fsig8)

# %% [markdown]
# ### Read chains from the DESI DR2 Lya BAO analysis

# %%
BAO = load.load_BAO_data(planck_sig8, planck_f)

P1D = load.load_p1d_data(planck_f)

# %%
print("P1D", "dr1_hsnr", "dr2_hsnr")
for key in ["bias_delta_sig_8_z", "bias_eta_f_sig_8_z"]:
    print()
    print(
        key,
        np.round(np.mean(P1D[key]), 4),
        np.round(np.mean(BAO["dr1_hsnr"][key]), 4),
        np.round(np.mean(BAO["dr2_hsnr"][key]), 4),
    )
    print(
        "err",
        np.round(np.std(P1D[key]), 4),
        np.round(np.std(BAO["dr1_hsnr"][key]), 4),
        np.round(np.std(BAO["dr2_hsnr"][key]), 4),
    )

# %%
samples = set_getdist.set_getdist_samples(BAO, P1D)
samples.keys()

# %% [markdown]
# Fit for importance sampling

# %%
fits = {}
for label in ['dr1', 'dr2', "dr1_hsnr", "dr2_hsnr"]:
    fits[label] = importance.fit_gaussian(samples[label])

# %%
labs_comb = {
    "dr1": "p1d_dr1_low",
    "dr2": "p1d_dr2_low",
    "dr1_hsnr": "p1d_dr1",
    "dr2_hsnr": "p1d_dr2",
}
label_sample = {
    "dr1": "P1D + BAO DR1 low SNR",
    "dr2": "P1D + BAO DR2 low SNR",
    "dr1_hsnr": "P1D + BAO DR1",
    "dr2_hsnr": "P1D + BAO DR2",
}


for label in fits.keys():
    print(labs_comb[label], label)
    samples[labs_comb[label]] = importance.combine(
        samples["p1d"], fits[label], label_sample[label]
    )

# %% [markdown]
# ### Plots

# %% [markdown]
# Data to combine

# %%
all_plots.plot_bsig8_betafsigma8(samples)

# %% [markdown]
# BAO biases

# %%
all_plots.plot_bao_biases(samples)

# %% [markdown]
# Fit to P1D data using Gaussian approximation to the likelihood. We can compute the mean and covariance from the samples, then plot the corresponding Gaussian contours on top of the GetDist plot.

# %%
samples.keys()

# %%
all_plots.plot_comb_bdsig8_befsig8(samples)

# %%
all_plots.plot_sig8(samples)

# %%
all_plots.plot_sig8z(samples)

# %%
all_plots.plot_fsig8z(samples)

# %%
all_plots.plot_sig8z233(samples)

# %%
all_plots.plot_compressed(samples)

# %%
all_plots.plot_bdelta_beta_beta(samples)

# %%
all_plots.plot_P3D_small_params(samples)

# %%

# %%
