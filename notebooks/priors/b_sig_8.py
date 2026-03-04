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
f_p1d = 0.9678318902805135

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
P1D["sig_8"] = dict_out_all["emu_params"]["sig_8"]
P1D["fsig_8"] = P1D["sig_8"] * f_p1d

for par in dict_out_all["forest_out"].keys():
    if par == "bias":
        lab = "bias_delta"
    else:
        lab = par

    P1D[lab] = dict_out_all["forest_out"][par][:, 1]

P1D["bias_eta"] = P1D["bias_delta"] * P1D["beta"] /f_p1d

P1D["Delta2star"] = dict_out_all["emu_params"]["Delta2star"]
P1D["nstar"] = dict_out_all["emu_params"]["nstar"]

# %% [markdown]
# I'd start by doing two importance sampling runs: P1D + DR1 BAO and P1D + DR2 BAO .
#
# Then I'd make a triangle plot for these two join runs and the three runs above, showing the following parameters: sig8, fsig8, b_delta sig8 and b_eta f sig8
#
# Some runs will miss parameters, but that's ok. For instance:
#
# - BAO runs will not have constraints on sig8 and fsig8
# - P1D runs will have fully degenerate contours for sig8 and fsig8 (f should not be a free parameter in P1D chains, right?)

# %%
P1D.keys()

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
    [
        P1D["bias_delta_sig_8_z"],
        P1D["bias_eta_f_sig_8_z"],
        P1D["sig_8"],
        P1D["fsig_8"],
        P1D["bias_delta"],
        P1D["beta"],
        P1D["q1"],
        P1D["kvav"],
        P1D["av"],
        P1D["bv"],
        P1D["kp"],
        P1D["q2"],
        P1D["bias_eta"],
        P1D["Delta2star"],
        P1D["nstar"],
    ]
).T

names = ["b_delta_sigma8", "b_eta_f_sigma8"]
# labels = [r"$b_\delta \sigma_8(z)$", r"$b_{\eta} f \sigma_8(z)$"]
labels = [r"b_\delta \sigma_8(z)", r"b_{\eta} f \sigma_8(z)"]

# --- define MCSamples ---
s_bao_dr1 = MCSamples(
    samples=bao_dr1.copy(),
    names=names,
    labels=labels,
    weights=dr1_samples.weights.copy(),
    label="BAO DR1",
)

s_bao_dr2 = MCSamples(
    samples=bao_dr2.copy(),
    names=names,
    labels=labels,
    weights=dr2_samples.weights.copy(),
    label="BAO DR2",
)


names = [
    "b_delta_sigma8",
    "b_eta_f_sigma8",
    "sigma8",
    "fsigma8",
    "bias_delta",
    "beta",
    "q1",
    "kvav",
    "av",
    "bv",
    "kp",
    "q2",
    "bias_eta",
    "Delta2star",
    "nstar",
]
# labels = [r"$b_\delta \sigma_8(z)$", r"$b_{\eta} f \sigma_8(z)$"]
labels = [
    r"b_\delta \sigma_8(z)",
    r"b_{\eta} f \sigma_8(z)",
    r"\sigma_8(z)",
    r"f \sigma_8(z)",
    r"b_\delta(z)",
    r"\beta(z)",
    r"q_1",
    r"k_{vav}",
    r"a_v",
    r"b_v",
    r"k_p",
    r"q_2",
    r"b_\eta(z)",
    r"\Delta^2_*$",
    r"n_*",
]

s_p1d = MCSamples(
    samples=p1d.copy(),
    names=names,
    labels=labels,
    label="P1D DR1",
)

rw1_p1d = MCSamples(
    samples=p1d.copy(),
    names=names,
    labels=labels,
    label="P1D DR1 weighted BAO DR1",
)

rw2_p1d = MCSamples(
    samples=p1d.copy(),
    names=names,
    labels=labels,
    label="P1D DR1 weighted BAO DR2",
)

# %%

# # optional smoothing comparable to corner(smooth=True)
# for s in [s_bao_dr1, s_bao_dr2, s_p1d]:
#     s.updateSettings({'smooth_scale_2D': 0.5})

# --- plotting ---
g = plots.get_subplot_plotter(width_inch=10)
g.settings.lab_fontsize = ftsize
g.settings.axes_fontsize = ftsize
g.settings.legend_fontsize = ftsize
g.settings.num_plot_contours = 2  # 68%, 95%

g.triangle_plot(
    [s_p1d, s_bao_dr1, s_bao_dr2],
    params=["b_delta_sigma8", "b_eta_f_sigma8"],
    # filled=True,
    contour_colors=["C0", "C1", "C2"],
)


# %% [markdown]
# Fit to P1D data using Gaussian approximation to the likelihood. We can compute the mean and covariance from the samples, then plot the corresponding Gaussian contours on top of the GetDist plot.

# %%
from getdist import plots

fits  = {}

for samples in [s_bao_dr1, s_bao_dr2]:

    # assume `samples` is an MCSamples object
    p1, p2 = 'b_delta_sigma8', 'b_eta_f_sigma8'

    # --- Gaussian approximation from samples ---
    params = samples.getParams()
    x = getattr(params, p1)
    y = getattr(params, p2)
    w = samples.weights.copy()
    w = w / np.sum(w)  # normalize weights

    data = np.vstack([x, y]).T
    mean = np.average(data, axis=0, weights=w)
    cov = np.cov(data, rowvar=False, aweights=w)

    x_val, y_val = mean
    x_err = np.sqrt(cov[0, 0])
    y_err = np.sqrt(cov[1, 1])
    r = cov[0, 1] / (x_err * y_err)

    fits[samples.label] = {
        'x_val': x_val,
        'y_val': y_val,
        'x_err': x_err,
        'y_err': y_err,
        'r': r,
    }

    # eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # 68% contour scaling for 2D Gaussian
    # chi2_2(0.68) ≈ 2.30
    scale = np.sqrt(2.30)

    theta = np.linspace(0, 2*np.pi, 400)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    ellipse1 = (eigvecs @ np.diag(np.sqrt(eigvals)) @ circle) * scale
    ellipse1[0] += mean[0]
    ellipse1[1] += mean[1]

    scale = np.sqrt(5.99)  # 95% contour scaling for 2D Gaussian
    ellipse2 = (eigvecs @ np.diag(np.sqrt(eigvals)) @ circle) * scale
    ellipse2[0] += mean[0]
    ellipse2[1] += mean[1]  

    # --- GetDist plot ---
    # g = plots.get_subplot_plotter()
    g = plots.get_subplot_plotter(width_inch=10)
    g.plot_2d(samples, p1, p2, filled=True)

    ax = g.subplots[0,0]
    ax.plot(ellipse1[0], ellipse1[1], color='k', lw=2, label='Gaussian (68%)')
    ax.plot(ellipse2[0], ellipse2[1], color='k', lw=2, ls='--', label='Gaussian (95%)')
    ax.legend()


# %%
def gaussian_chi2(x, y, x_val, y_val, x_err, y_err, r):
    """Given central values and errors for Delta_L^2 and n_eff, and its
    cross-correlation coefficient r, compute Gaussian delta chi^2 at
    points (neff,DL2).
    """
    chi2 = (
        (y - y_val) ** 2 / y_err**2
        + (x - x_val) ** 2 / x_err**2
        - 2 * r * (x - x_val) * (y - y_val) / y_err / x_err
    ) / (1 - r * r)
    return chi2


# %%
fits.keys()

# %%
labels = ["BAO DR1", "BAO DR2"]

for ii, samples in enumerate([rw1_p1d, rw2_p1d]):

    label = labels[ii]

    # assume `samples` is an MCSamples object
    p1, p2 = 'b_delta_sigma8', 'b_eta_f_sigma8'

    # --- Gaussian approximation from samples ---
    params = samples.getParams()
    x = getattr(params, p1)
    y = getattr(params, p2)

    logw = 0.5 * gaussian_chi2(x, y, fits[label]['x_val'], fits[label]['y_val'], fits[label]['x_err'], fits[label]['y_err'], fits[label]['r'])

    samples.reweightAddingLogLikes(logw)


# %%
# --- plotting ---
g = plots.get_subplot_plotter(width_inch=12)
g.settings.lab_fontsize = ftsize
g.settings.axes_fontsize = ftsize
g.settings.legend_fontsize = ftsize
g.settings.num_plot_contours = 2  # 68%, 95%

g.triangle_plot(
    [s_p1d, s_bao_dr1, s_bao_dr2, rw1_p1d, rw2_p1d],
    filled=[False, False, False, True, True],
    params=["b_delta_sigma8", "b_eta_f_sigma8"],
    contour_colors=["C0", "C1", "C2", "C3", "C4"],
)

# %%
# --- plotting ---
g = plots.get_subplot_plotter(width_inch=12)
g.settings.lab_fontsize = ftsize
g.settings.axes_fontsize = ftsize
g.settings.legend_fontsize = ftsize
g.settings.num_plot_contours = 2  # 68%, 95%

g.triangle_plot(
    [s_p1d, rw1_p1d, rw2_p1d],
    filled=[False, False, False, True, True],
    params=["sigma8", "fsigma8"],
    contour_colors=["C0", "C1", "C2", "C3", "C4"],
)

# %%

for sampler in [s_p1d, rw1_p1d, rw2_p1d]:
    sampler.updateSettings({'smooth_scale_2D': 0.8, 'smooth_scale_1D': 0.9})


# --- plotting ---
g = plots.get_subplot_plotter(width_inch=12)
g.settings.lab_fontsize = ftsize
g.settings.axes_fontsize = ftsize
g.settings.legend_fontsize = ftsize
g.settings.num_plot_contours = 2  # 68%, 95%


g.triangle_plot(
    [s_p1d, rw1_p1d, rw2_p1d],
    filled=[False, False, False, True, True],
    params=["Delta2star", "nstar"],
    contour_colors=["C0", "C1", "C2", "C3", "C4"],
)

# %%
for sampler in [s_p1d, rw1_p1d, rw2_p1d]:
    sampler.updateSettings({'smooth_scale_2D': 0.8, 'smooth_scale_1D': 0.6})


# --- plotting ---
g = plots.get_subplot_plotter(width_inch=12)
g.settings.lab_fontsize = ftsize
g.settings.axes_fontsize = ftsize
g.settings.legend_fontsize = ftsize
g.settings.num_plot_contours = 2  # 68%, 95%


g.triangle_plot(
    [s_p1d, rw1_p1d, rw2_p1d],
    filled=[False, False, False, True, True],
    params=["bias_delta", "bias_eta"],
    contour_colors=["C0", "C1", "C2", "C3", "C4"],
)

# %%
for sampler in [s_p1d, rw1_p1d, rw2_p1d]:
    sampler.updateSettings({"smooth_scale_2D": 0.8, "smooth_scale_1D": 0.6})


# --- plotting ---
g = plots.get_subplot_plotter(width_inch=12)
g.settings.lab_fontsize = ftsize
g.settings.axes_fontsize = ftsize
g.settings.legend_fontsize = ftsize
g.settings.num_plot_contours = 2  # 68%, 95%


g.triangle_plot(
    [s_p1d, rw1_p1d, rw2_p1d],
    filled=[False, False, False, True, True],
    params=["bias_delta", "bias_eta", "beta"],
    contour_colors=["C0", "C1", "C2", "C3", "C4"],
)

# %%
for sampler in [s_p1d, rw1_p1d, rw2_p1d]:
    sampler.updateSettings({'smooth_scale_2D': 0.8, 'smooth_scale_1D': 0.6})


# --- plotting ---
g = plots.get_subplot_plotter(width_inch=12)
g.settings.lab_fontsize = ftsize
g.settings.axes_fontsize = ftsize
g.settings.legend_fontsize = ftsize
g.settings.num_plot_contours = 2  # 68%, 95%


g.triangle_plot(
    [s_p1d, rw1_p1d, rw2_p1d],
    filled=[False, False, False, True, True],
    params=["q1", "q2", "kvav", "av", "bv", "kp"],
    contour_colors=["C0", "C1", "C2", "C3", "C4"],
)

# %% [markdown]
# beta = biaseta * f(z) / biasdelta
#
# biaseta = beta * biasdelta / f(z)
#
# biaseta * f(z) * sig8(z) = beta * biasdelta * sig8(z)

# %%

# %%
