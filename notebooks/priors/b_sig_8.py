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
from vega import FitResults

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
BAO["dr1"]["bias_delta"] = dr1_samples.getParams().bias_LYA
BAO["dr1"]["beta"] = dr1_samples.getParams().beta_LYA
BAO["dr1"]["bias_hcd"] = dr1_samples.getParams().bias_hcd

BAO["dr2"] = {}
BAO["dr2"]["bias_delta_sig_8_z"] = dr2_samples.getParams().bias_LYA * sig_8
BAO["dr2"]["bias_eta_f_sig_8_z"] = dr2_samples.getParams().beta_LYA * dr2_samples.getParams().bias_LYA * sig_8
BAO["dr2"]["bias_delta"] = dr2_samples.getParams().bias_LYA
BAO["dr2"]["beta"] = dr2_samples.getParams().beta_LYA
BAO["dr2"]["bias_hcd"] = dr2_samples.getParams().bias_hcd


# %%
basedir = '/home/jchaves/Proyectos/projects/lya/data/lya_bao/fits_andreu/'
dr2snr_samples = FitResults(basedir + 'fit_output_mid_prior.fits')

# parameters dictionary (15 parameters)
params = dr2snr_samples.params

# covariance matrix (15x15)
cov = dr2snr_samples.cov

# mean vector (ordered consistently with the covariance)
names = list(params.keys())
mean = np.array([params[k] for k in names])

# number of realizations
n = 10000

# draw samples
samples = np.random.multivariate_normal(mean, cov, size=n)

# dictionary of arrays (each array has length n)
samples_dict = {name: samples[:, i] for i, name in enumerate(names)}

BAO["dr2_hsnr"] = {}
BAO["dr2_hsnr"]["bias_delta_sig_8_z"] = samples_dict["bias_LYA"] * sig_8
BAO["dr2_hsnr"]["bias_eta_f_sig_8_z"] = samples_dict["beta_LYA"] * samples_dict["bias_LYA"] * sig_8
BAO["dr2_hsnr"]["bias_delta"] = samples_dict["bias_LYA"]
BAO["dr2_hsnr"]["beta"] = samples_dict["beta_LYA"]
BAO["dr2_hsnr"]["bias_hcd"] = samples_dict["bias_hcd"]


dr1snr_samples = FitResults(basedir + 'fit_output_dr1_mid_prior.fits')

# parameters dictionary (15 parameters)
params = dr1snr_samples.params

# covariance matrix (15x15)
cov = dr1snr_samples.cov

# mean vector (ordered consistently with the covariance)
names = list(params.keys())
mean = np.array([params[k] for k in names])

# number of realizations
n = 10000

# draw samples
samples = np.random.multivariate_normal(mean, cov, size=n)

# dictionary of arrays (each array has length n)
samples_dict = {name: samples[:, i] for i, name in enumerate(names)}

BAO["dr1_hsnr"] = {}
BAO["dr1_hsnr"]["bias_delta_sig_8_z"] = samples_dict["bias_LYA"] * sig_8
BAO["dr1_hsnr"]["bias_eta_f_sig_8_z"] = samples_dict["beta_LYA"] * samples_dict["bias_LYA"] * sig_8
BAO["dr1_hsnr"]["bias_delta"] = samples_dict["bias_LYA"]
BAO["dr1_hsnr"]["beta"] = samples_dict["beta_LYA"]
BAO["dr1_hsnr"]["bias_hcd"] = samples_dict["bias_hcd"]

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
P1D["sig_8_z0"] = dict_out_all["emu_params"]["sig_8_z0"]
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
ftsize = 18

# --- prepare samples ---
bao_dr1 = np.vstack(
    [
        BAO["dr1"]["bias_delta_sig_8_z"],
        BAO["dr1"]["bias_eta_f_sig_8_z"],
        BAO["dr1"]["bias_delta"],
        BAO["dr1"]["beta"],
        BAO["dr1"]["bias_hcd"],
    ]
).T

bao_dr2 = np.vstack(
    [
        BAO["dr2"]["bias_delta_sig_8_z"],
        BAO["dr2"]["bias_eta_f_sig_8_z"],
        BAO["dr2"]["bias_delta"],
        BAO["dr2"]["beta"],
        BAO["dr2"]["bias_hcd"],
    ]
).T

bao_dr1_hsnr = np.vstack(
    [
        BAO["dr1_hsnr"]["bias_delta_sig_8_z"],
        BAO["dr1_hsnr"]["bias_eta_f_sig_8_z"],
        BAO["dr1_hsnr"]["bias_delta"],
        BAO["dr1_hsnr"]["beta"],
        BAO["dr1_hsnr"]["bias_hcd"],
    ]
).T

bao_dr2_hsnr = np.vstack(
    [
        BAO["dr2_hsnr"]["bias_delta_sig_8_z"],
        BAO["dr2_hsnr"]["bias_eta_f_sig_8_z"],
        BAO["dr2_hsnr"]["bias_delta"],
        BAO["dr2_hsnr"]["beta"],
        BAO["dr2_hsnr"]["bias_hcd"],
    ]
).T

p1d = np.vstack(
    [
        P1D["bias_delta_sig_8_z"],
        P1D["bias_eta_f_sig_8_z"],
        P1D["sig_8"],
        P1D["sig_8_z0"],
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

names = ["b_delta_sigma8", "b_eta_f_sigma8", "bias_delta", "beta", "bias_hcd"]
labels = [r"b_\delta \sigma_8", r"b_{\eta} f \sigma_8", r"b_\delta", r"\beta", r"b_\mathrm{HCD}"]

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

s_bao_dr1_hsnr = MCSamples(
    samples=bao_dr1_hsnr.copy(),
    names=names,
    labels=labels,
    label="BAO DR1 SNR",
)

s_bao_dr2_hsnr = MCSamples(
    samples=bao_dr2_hsnr.copy(),
    names=names,
    labels=labels,
    label="BAO DR2 SNR",
)

names = [
    "b_delta_sigma8",
    "b_eta_f_sigma8",
    "sigma8",
    "sigma8_z0",
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
labels = [
    r"b_\delta \sigma_8",
    r"b_\eta f \sigma_8",
    r"\sigma_8(z=2.33)",
    r"\sigma_8(z=0)",
    r"f \sigma_8",
    r"b_\delta",
    r"\beta",
    r"q_1",
    r"k_\mathrm{vav}",
    r"a_\mathrm{v}",
    r"b_\mathrm{v}",
    r"k_\mathrm{p}",
    r"q_2",
    r"b_\eta",
    r"\Delta^2_\star",
    r"n_\star",
]

s_p1d = MCSamples(
    samples=p1d.copy(),
    names=names,
    labels=labels,
    label="P1D",
)

rw1_p1d = MCSamples(
    samples=p1d.copy(),
    names=names,
    labels=labels,
    label="P1D + BAO DR1",
)

rw2_p1d = MCSamples(
    samples=p1d.copy(),
    names=names,
    labels=labels,
    label="P1D + BAO DR2",
)

# %%

# # optional smoothing comparable to corner(smooth=True)
# for s in [s_bao_dr1, s_bao_dr2, s_p1d]:
#     s.updateSettings({'smooth_scale_2D': 0.5})

# --- plotting ---
g = plots.get_subplot_plotter(width_inch=8)
g.settings.lab_fontsize = ftsize
g.settings.axes_fontsize = ftsize
g.settings.legend_fontsize = ftsize
g.settings.num_plot_contours = 2  # 68%, 95%

g.triangle_plot(
    [s_p1d, s_bao_dr1, s_bao_dr2, s_bao_dr1_hsnr, s_bao_dr2_hsnr],
    params=["b_delta_sigma8", "b_eta_f_sigma8"],
    filled=[True, False, False, True, True,],
    contour_colors=["C0", "C1", "C2", "C3", "C4"],
    contour_ls=[
        "-",
        "--",
        "-.",
        ":",
        ":",
    ],
    contour_lws=[3, 3, 3, 3, 3],
)


plt.tight_layout()
plt.savefig("figs/nocomb_bdsig8_befsig8.pdf")
plt.savefig("figs/nocomb_bdsig8_befsig8.png")

# %%
# --- plotting ---
ftsize = 20
g = plots.get_subplot_plotter(width_inch=8)
g.settings.lab_fontsize = ftsize
g.settings.axes_fontsize = ftsize
g.settings.legend_fontsize = ftsize
g.settings.num_plot_contours = 2  # 68%, 95%

g.triangle_plot(
    [s_bao_dr1, s_bao_dr2, s_bao_dr1_hsnr, s_bao_dr2_hsnr],
    params=["bias_delta", "beta", "bias_hcd"],
    filled=[False, False, True, True,],
    contour_colors=["C0", "C1", "C9", "C3"],
    contour_ls=[
        "-",
        "--",
        ":",
        ":",
    ],
    contour_lws=[2, 2, 3, 3],
)

plt.tight_layout()
plt.savefig("figs/bao_biases.pdf")
plt.savefig("figs/bao_biases.png")

# %% [markdown]
# Fit to P1D data using Gaussian approximation to the likelihood. We can compute the mean and covariance from the samples, then plot the corresponding Gaussian contours on top of the GetDist plot.

# %%
from getdist import plots

fits  = {}

for samples in [s_bao_dr1_hsnr, s_bao_dr2_hsnr]:

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
labels = ["BAO DR1 SNR", "BAO DR2 SNR"]

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

# %%
# --- plotting ---
ftsize = 20
g = plots.get_subplot_plotter(width_inch=8)
g.settings.lab_fontsize = ftsize
g.settings.axes_fontsize = ftsize
g.settings.legend_fontsize = ftsize
g.settings.num_plot_contours = 2  # 68%, 95%

# g.triangle_plot(
#     [s_p1d, s_bao_dr1, s_bao_dr2, rw2_p1d],
#     filled=[False, False, False, True],
#     params=["b_delta_sigma8", "b_eta_f_sigma8"],
#     contour_colors=["C0", "C2", "C3", "C1"],
#     contour_ls=["-", ":", "-.", "--",],
#     contour_lws=[2.0, 2.0, 2.0, 2.],
# )



g.triangle_plot(
    [s_p1d, s_bao_dr1_hsnr, s_bao_dr2_hsnr, rw1_p1d, rw2_p1d],
    filled=[False, False, False, True, True, True],
    params=["b_delta_sigma8", "b_eta_f_sigma8"],
    contour_colors=["C0", "C9", "C3", "C1", "C2"],
    contour_ls=[
        "-",
        ":",
        ":",
        "--",
        "-.",
    ],
    contour_lws=[3.0, 3.0, 3.0, 3.0, 3.0, 2],
)

plt.tight_layout()
plt.savefig("figs/comb_bdsig8_befsig8.pdf")
plt.savefig("figs/comb_bdsig8_befsig8.png")

# %%

# %%

# DESY6 Table IV https://arxiv.org/pdf/2601.14559
# DES 3x2pt LCDM
mu_des = 0.751
sigma_des = 0.035
# DES all LCDM
mu_des_all = 0.771
sigma_des_all = 0.020

# CMB-SPA https://arxiv.org/abs/2506.20707v1 LCDM
mu_cmb = 0.8137
sigma_cmb = 0.0038

# DESI BAO + full shape LCDM https://arxiv.org/abs/2602.18761
mu_desi = 0.822
sigma_desi = 0.034

# --- GetDist chains ---
chains = [s_p1d, rw1_p1d, rw2_p1d]
chain_labels = [r"DESI P1D", r"DESI P1D & Ly$\alpha$ BAO DR1", r"DESI P1D & Ly$\alpha$ BAO DR2"]

mus = []
sigmas = []

for s in chains:
    m = s.getMeans()[s.index["sigma8_z0"]]
    cov = s.getCov()
    i = s.index["sigma8_z0"]
    sig = np.sqrt(cov[i, i])
    mus.append(m)
    sigmas.append(sig)

# --- external constraints ---
labels = chain_labels + [r"DESI BAO & FS", "DES", "CMB-SPA"]
mu = np.array(mus + [mu_desi, mu_des_all, mu_cmb])
sigma = np.array(sigmas + [sigma_desi, sigma_des_all, sigma_cmb])

# reverse order
labels = labels[::-1]
mu = mu[::-1]
sigma = sigma[::-1]


colors = [f"C{i}" for i in range(len(mu))]
colors = colors[::-1]
y = np.arange(len(labels))

ftsize = 18
fig, ax = plt.subplots(figsize=(8,6))

for i in range(len(mu)):
    ax.errorbar(
        mu[i],
        y[i],
        xerr=sigma[i],
        fmt="o",
        lw=2,
        capsize=4,
        color=colors[i],
    )

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=ftsize)
ax.set_xlabel(r"$\sigma_8$", fontsize=ftsize+2)
ax.tick_params(labelsize=ftsize)
# ax.grid(axis="x", alpha=0.3)

plt.tight_layout()

plt.tight_layout()
plt.savefig("figs/sig8.pdf")
plt.savefig("figs/sig8.png")

# %%
# --- plotting ---
ftsize = 20
g = plots.get_subplot_plotter(width_inch=8)
g.settings.lab_fontsize = ftsize
g.settings.axes_fontsize = ftsize
g.settings.legend_fontsize = ftsize
g.settings.num_plot_contours = 2  # 68%, 95%


g.triangle_plot(
    [s_p1d, rw1_p1d, rw2_p1d],
    filled=[False, True, True, True],
    params=["sigma8"],
    contour_colors=["C0", "C1", "C2", "C3"],
    contour_ls=["-", "--", "-.", ":",],
    contour_lws=[3.0, 3.0, 3.0, 2.],
)

plt.tight_layout()
plt.savefig("figs/sig8z233.pdf")  
plt.savefig("figs/sig8z233.png")  

# %%

# for sampler in [s_p1d, rw2_p1d]:
#     sampler.updateSettings({'smooth_scale_2D': 0.5})

# --- plotting ---
ftsize = 20
g = plots.get_subplot_plotter(width_inch=8)
g.settings.lab_fontsize = ftsize
g.settings.axes_fontsize = ftsize
g.settings.legend_fontsize = ftsize
g.settings.num_plot_contours = 2  # 68%, 95%

g.triangle_plot(
    [s_p1d, rw1_p1d, rw2_p1d],
    filled=[False, True, True, True],
    params=["Delta2star", "nstar"],
    contour_colors=["C0", "C1", "C2", "C3"],
    contour_ls=["-", "--", "-.", ":",],
    contour_lws=[3.0, 3.0, 3.0, 2.],
)

# g.triangle_plot(
#     [s_p1d, rw2_p1d],
#     filled=[False, False, False, True],
#     params=["Delta2star", "nstar"],
#     contour_colors=["C0", "C1", "C2", "C3"],
#     contour_ls=["-", "--", "-.", ":",],
#     contour_lws=[2.0, 2.0, 2.0, 2.],
# )

plt.tight_layout()
plt.savefig("figs/delta2star_nstar.pdf")  
plt.savefig("figs/delta2star_nstar.png")

# %%
# for sampler in [s_p1d, rw1_p1d, rw2_p1d]:
#     sampler.updateSettings({'smooth_scale_2D': 0.8, 'smooth_scale_1D': 0.6})


# --- plotting ---
ftsize = 20
g = plots.get_subplot_plotter(width_inch=8)
g.settings.lab_fontsize = ftsize
g.settings.axes_fontsize = ftsize
g.settings.legend_fontsize = ftsize
g.settings.num_plot_contours = 2  # 68%, 95%

# g.triangle_plot(
#     [s_p1d, rw2_p1d],
#     filled=[False, False, False, True],
#     params=["bias_delta", "bias_eta"],
#     contour_colors=["C0", "C1", "C2", "C3"],
#     contour_ls=["-", "--", "-.", ":",],
#     contour_lws=[2.0, 2.0, 2.0, 2.],
# )

g.triangle_plot(
    [s_p1d, rw1_p1d, rw2_p1d],
    filled=[False, True, True, True],
    params=["bias_delta", "bias_eta"],
    contour_colors=["C0", "C1", "C2", "C3"],
    contour_ls=["-", "--", "-.", ":",],
    contour_lws=[3.0, 3.0, 3.0, 2.],
)


plt.tight_layout()
plt.savefig("figs/bdelta_beta.pdf")  
plt.savefig("figs/bdelta_beta.png")

# %%
# --- plotting ---
ftsize = 20
g = plots.get_subplot_plotter(width_inch=8)
g.settings.lab_fontsize = ftsize
g.settings.axes_fontsize = ftsize
g.settings.legend_fontsize = ftsize
g.settings.num_plot_contours = 2  # 68%, 95%

g.triangle_plot(
    [s_p1d, rw1_p1d, rw2_p1d],
    filled=[False, True, True, True],
    params=["bias_delta", "bias_eta", "beta"],
    contour_colors=["C0", "C1", "C2", "C3"],
    contour_ls=["-", "--", "-.", ":",],
    contour_lws=[3.0, 3.0, 3.0, 2.],
)


# g.triangle_plot(
#     [s_p1d, rw2_p1d],
#     filled=[False, False, False, True],
#     params=["bias_delta", "bias_eta", "beta"],
#     contour_colors=["C0", "C1", "C2", "C3"],
#     contour_ls=["-", "--", "-.", ":",],
#     contour_lws=[2.0, 2.0, 2.0, 2.],
# )

plt.tight_layout()
plt.savefig("figs/bdelta_beta_beta.pdf")  
plt.savefig("figs/bdelta_beta_beta.png")

# %%
# --- plotting ---
ftsize = 20
g = plots.get_subplot_plotter(width_inch=10)
g.settings.lab_fontsize = ftsize
g.settings.axes_fontsize = ftsize
g.settings.legend_fontsize = ftsize
g.settings.num_plot_contours = 2  # 68%, 95%

g.triangle_plot(
    [s_p1d, rw1_p1d, rw2_p1d],
    filled=[False, True, True, True],
    params=["q1", "q2", "kvav", "av", "bv", "kp"],
    contour_colors=["C0", "C1", "C2", "C3"],
    contour_ls=[
        "-",
        "--",
        "-.",
        ":",
    ],
    contour_lws=[3.0, 3.0, 3.0, 2.0],
)


# g.triangle_plot(
#     [s_p1d, rw2_p1d],
#     filled=[False, False, False, True],
#     params=["q1", "q2", "kvav", "av", "bv", "kp"],
#     contour_colors=["C0", "C1", "C2", "C3"],
#     contour_ls=[
#         "-",
#         "--",
#         "-.",
#         ":",
#     ],
#     contour_lws=[2.0, 2.0, 2.0, 2.0],
# )

plt.tight_layout()
plt.savefig("figs/corner_arinyo.pdf")
plt.savefig("figs/corner_arinyo.png")

# %% [markdown]
# beta = biaseta * f(z) / biasdelta
#
# biaseta = beta * biasdelta / f(z)
#
# biaseta * f(z) * sig8(z) = beta * biasdelta * sig8(z)

# %%

# %%
