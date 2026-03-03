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

from cup1d.likelihood.pipeline import Pipeline
from lace.cosmo import camb_cosmo, fit_linP
import forestflow
from forestflow.P3D_cINN import P3DEmulator
# path of the repo
path_repo = os.path.dirname(forestflow.__path__[0])

np.__version__

# %%
from getdist import plots, loadMCSamples, MCSamples

# %% [markdown]
# beta = biaseta * f(z) / biasdelta
#
# biaseta = beta * biasdelta / f(z)
#
# biaseta * f(z) * sig8(z) = beta * biasdelta * sig8(z)

# %% [markdown]
# ### Setup fiducial cosmology from DESI DR2 Lya BAO and compute sigma_8

# %%
import camb

pars = camb.CAMBparams()
# set background cosmology
pars.set_cosmology(H0=67.36, ombh2=0.02237, omch2=0.12, mnu=0.06)
# set primordial power
pars.InitPower.set_params(As=2.1e-9, ns=0.9649)
# compute sig_8 at z=2.33 and at z=0
pars.set_matter_power(redshifts=[2.33, 0.0])
results = camb.get_results(pars)
sig_8, sig_8_z0 = np.array(results.get_sigma8())
print(sig_8, sig_8_z0)
# same for f sig_8
f_sig_8, f_sig_8_z0 = results.get_fsigma8()
print(f_sig_8, f_sig_8_z0)

# %% [markdown]
# ### Read chains from the DESI DR2 Lya BAO analysis

# %%

chains_dir = "/home/jchaves/Proyectos/projects/lya/data/lya_bao/dr1/"
chains_file = chains_dir + "/lyaxlya_lyaxlyb_lyaxqso_lybxqso-baseline_combined"
dr1_samples = loadMCSamples(chains_file)

chains_dir = "/home/jchaves/Proyectos/projects/lya/data/lya_bao/dr2/"
chains_file = chains_dir + "/lyaxlya_lyaxlyb_lyaxqso_lybxqso-final_base"
dr2_samples = loadMCSamples(chains_file)

BAO = {}

BAO["dr1"] = {}
BAO["dr1"]["bias_delta_sig_8_z"] = dr1_samples.getParams().bias_LYA * sig_8
BAO["dr1"]["bias_eta_f_sig_8_z"] = dr1_samples.getParams().beta_LYA * dr1_samples.getParams().bias_LYA * sig_8

BAO["dr2"] = {}
BAO["dr2"]["bias_delta_sig_8_z"] = dr2_samples.getParams().bias_LYA * sig_8
BAO["dr2"]["bias_eta_f_sig_8_z"] = dr2_samples.getParams().beta_LYA * dr2_samples.getParams().bias_LYA * sig_8

# %%
dict_out_all = np.load("arinyo_from_desi_p1d.npy", allow_pickle=True).item()
dict_out_all.keys()

dict_out_all["forest_out"]["bias"] = -np.abs(dict_out_all["forest_out"]["bias"])

P1D = {}
P1D["bias_delta_sig_8_z"] = (
    dict_out_all["emu_params"]["sig_8"] * dict_out_all["forest_out"]["bias"][:, 1]
)
P1D["bias_eta_f_sig_8_z"] = (
    dict_out_all["emu_params"]["sig_8"]
    * dict_out_all["forest_out"]["bias"][:, 1]
    * dict_out_all["forest_out"]["beta"][:, 1]
)

# %%
ftsize = 18

# --- prepare samples ---
bao_dr1 = np.vstack(
    [BAO["dr1"]["bias_delta_sig_8_z"], BAO["dr1"]["bias_eta_f_sig_8_z"]]
).T
bao_dr2 = np.vstack(
    [BAO["dr2"]["bias_delta_sig_8_z"], BAO["dr2"]["bias_eta_f_sig_8_z"]]
).T
p1d = np.vstack(
    [P1D["bias_delta_sig_8_z"], P1D["bias_eta_f_sig_8_z"]]
).T

names  = ["b_delta_sigma8", "b_eta_f_sigma8"]
# labels = [r"$b_\delta \sigma_8(z)$", r"$b_{\eta} f \sigma_8(z)$"]
labels = [r"b_\delta \sigma_8(z)", r"b_{\eta} f \sigma_8(z)"]

# --- define MCSamples ---
s_bao_dr1 = MCSamples(
    samples=bao_dr1,
    names=names,
    labels=labels,
    weights=dr1_samples.weights,
    label="BAO DR1",
)

s_bao_dr2 = MCSamples(
    samples=bao_dr2,
    names=names,
    labels=labels,
    weights=dr2_samples.weights,
    label="BAO DR2",
)

s_p1d = MCSamples(
    samples=p1d,
    names=names,
    labels=labels,
    label="P1D DR1",
)

# # optional smoothing comparable to corner(smooth=True)
for s in [s_bao_dr1, s_bao_dr2, s_p1d]:
    s.updateSettings({'smooth_scale_2D': 0.5})

# --- plotting ---
g = plots.get_subplot_plotter(width_inch=10)
g.settings.lab_fontsize = ftsize
g.settings.axes_fontsize = ftsize
g.settings.legend_fontsize = ftsize
g.settings.num_plot_contours = 2  # 68%, 95%

g.triangle_plot(
    [s_bao_dr1, s_bao_dr2, s_p1d],
    # filled=True,
    contour_colors=["C0", "C1", "C2"],
)


# %% [markdown]
# beta = biaseta * f(z) / biasdelta
#
# biaseta = beta * biasdelta / f(z)
#
# biaseta * f(z) * sig8(z) = beta * biasdelta * sig8(z)

# %%
ftsize = 18

bias = np.mean(dict_out_all["forest_out"]["bias"][:, 1])
beta = np.mean(dict_out_all["forest_out"]["beta"][:, 1])
f_p1d = 0.9678318902805135
bias_eta = beta * bias / f_p1d

# --- prepare samples ---
bao_dr1 = np.vstack(
    [
        BAO["dr1"]["bias_delta_sig_8_z"] / bias,
        BAO["dr1"]["bias_eta_f_sig_8_z"] / bias_eta,
    ]
).T
bao_dr2 = np.vstack(
    [
        BAO["dr2"]["bias_delta_sig_8_z"] / bias,
        BAO["dr2"]["bias_eta_f_sig_8_z"] / bias_eta,
    ]
).T
p1d = np.vstack(
    [P1D["bias_delta_sig_8_z"] / bias, P1D["bias_eta_f_sig_8_z"] / bias_eta]
).T

names = ["sigma8", "f_sigma8"]
# labels = [r"$b_\delta \sigma_8(z)$", r"$b_{\eta} f \sigma_8(z)$"]
labels = [r"\sigma_8(z)", r"f \sigma_8(z)"]

# --- define MCSamples ---
s_bao_dr1 = MCSamples(
    samples=bao_dr1,
    names=names,
    labels=labels,
    weights=dr1_samples.weights,
    label="BAO DR1",
)

s_bao_dr2 = MCSamples(
    samples=bao_dr2,
    names=names,
    labels=labels,
    weights=dr2_samples.weights,
    label="BAO DR2",
)

s_p1d = MCSamples(
    samples=p1d,
    names=names,
    labels=labels,
    label="P1D DR1",
)

# # optional smoothing comparable to corner(smooth=True)
for s in [s_bao_dr1, s_bao_dr2, s_p1d]:
    s.updateSettings({"smooth_scale_2D": 0.5})

# --- plotting ---
g = plots.get_subplot_plotter(width_inch=10)
g.settings.lab_fontsize = ftsize
g.settings.axes_fontsize = ftsize
g.settings.legend_fontsize = ftsize
g.settings.num_plot_contours = 2  # 68%, 95%

g.triangle_plot(
    [s_bao_dr1, s_bao_dr2, s_p1d],
    # filled=True,
    contour_colors=["C0", "C1", "C2"],
)

# %%

# %%
