# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: lace
#     language: python
#     name: python3
# ---

# +
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
# -

from getdist import plots, loadMCSamples

# ### Setup fiducial cosmology from DESI DR2 Lya BAO and compute sigma_8

# +
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
# -

# ### Read chains from the DESI DR2 Lya BAO analysis

# chains_dir='/global/cfs/cdirs/desi/science/lya/y3/loa/final_results/sampler_runs/output_sampler/'
chains_dir = "/home/jchaves/Proyectos/projects/lya/data/lya_bao/output_sampler/"
chains_file = chains_dir + "/lyaxlya_lyaxlyb_lyaxqso_lybxqso-final_base"
samples = loadMCSamples(chains_file)

params = samples.getParams()

plt.hist(params.bias_LYA, bins=30)

# +
BAO = {}

BAO["bias_delta_sig_8_z"] = params.bias_LYA * sig_8
BAO["bias_eta_f_sig_8_z"] = params.beta_LYA * params.bias_LYA * sig_8

# +
dict_out_all = np.load("arinyo_from_desi_p1d.npy", allow_pickle=True).item()
dict_out_all.keys()

dict_out_all["forest_out"]["bias"] = -np.abs(dict_out_all["forest_out"]["bias"])
# -

dict_out_all["forest_out"].keys()

# beta = biaseta * f(z) / biasdelta
#
# biaseta = beta * biasdelta / f(z)
#
# biaseta * f(z) * sig8(z) = beta * biasdelta * sig8(z)

dict_out_all["emu_params"]["z"]

P1D = {}
P1D["bias_delta_sig_8_z"] = (
    dict_out_all["emu_params"]["sig_8"] * dict_out_all["forest_out"]["bias"][:, 1]
)
P1D["bias_eta_f_sig_8_z"] = (
    dict_out_all["emu_params"]["sig_8"]
    * dict_out_all["forest_out"]["bias"][:, 1]
    * dict_out_all["forest_out"]["beta"][:, 1]
)

from corner import corner

bins_list

# +

from matplotlib.lines import Line2D

# +
ftsize = 18


# prepare the two 2‑D arrays exactly as before
bao = np.vstack([BAO["bias_delta_sig_8_z"], BAO["bias_eta_f_sig_8_z"]]).T
p1d = np.vstack([P1D["bias_delta_sig_8_z"], P1D["bias_eta_f_sig_8_z"]]).T

# labels now formatted in LaTeX
labels = [r"$b_\delta \sigma_8(z)$", r"$b_{\eta} f \sigma_8(z)$"]

# --- compute common bins and range for each parameter ---
ndim = bao.shape[1]
bins = [20, 20]
range_list = []
for j in range(ndim):
    combined = np.concatenate((bao[:, j], p1d[:, j]))
    mn = np.percentile(combined, 0.5)
    mx = np.percentile(combined, 99.5)
    if mn == mx:
        mx = mn + 1e-6
    range_list.append((mn, mx))

# compute histogram peak for each dataset used for normalization
weights_bao = np.ones(bao.shape[0])/bao.shape[0]
weights_p1d = np.ones(p1d.shape[0])/p1d.shape[0]
# for j in range(ndim):
#     counts_bao, _ = np.histogram(bao[:, j], bins=bins[j], range=range_list[j])
#     counts_p1d, _ = np.histogram(p1d[:, j], bins=bins[j], range=range_list[j])
#     max_b = counts_bao.max() if counts_bao.size else 1
#     max_p = counts_p1d.max() if counts_p1d.size else 1
#     weights_bao /= float(max_b)
#     weights_p1d /= float(max_p)

fig = plt.figure(figsize=(10, 10))

# draw first dataset with its own weights
fig = corner(
    bao,
    labels=labels,
    show_titles=True,
    title_fmt=".4f",
    title_kwargs={"fontsize": ftsize},
    levels=[0.68, 0.95],
    color="C0",
    plot_datapoints=False,
    fill_contours=True,
    bins=bins,
    weights=weights_bao,
    range=range_list,
    label_kwargs={"fontsize": ftsize},
    fig=fig,
    smooth=True
)

# overlay second dataset with its weights
corner(
    p1d,
    fig=fig,
    levels=[0.68, 0.95],
    color="C1",
    plot_datapoints=False,
    fill_contours=True,
    bins=bins,
    weights=weights_p1d,
    range=range_list,
    smooth=True
)

# legend and formatting
handles = [
    Line2D([0], [0], color="C0", lw=2, label="BAO DR2"),
    Line2D([0], [0], color="C1", lw=2, label="P1D DR1"),
]
fig.legend(handles=handles, loc="upper right", fontsize=ftsize)
for ax in fig.axes:
    ax.tick_params(axis="both", labelsize=ftsize)

fig.axes[3].set_ylim(0, 0.25)

# -

range_list





g = plots.getSubplotPlotter(width_inch=6)
g.settings.axes_fontsize = 9
g.settings.legend_fontsize = 11
g.triangle_plot(
    [samples], ["bias_LYA", "beta_LYA"], legend_labels=[r"DESI DR2 LYA BAO"]
)

Nsamp, Npar = samples.samples.shape
print(Nsamp, Npar)

new_params_names = ["bias_eta_LYA", "bias_LYA_sig_8", "bias_eta_LYA_f_sig_8"]
# this will collect a dictionary for each sample in the chain
new_params_entries = []

test = samples.getParamSampleDict(0)

test

for i in range(Nsamp):
    verbose = i % 1000 == 0
    verbose = False
    if verbose:
        print("sample point", i)
    # get point from original chain
    sample = samples.getParamSampleDict(i)
    bias = sample["bias_LYA"]
    beta = sample["beta_LYA"]
    # compute derived parameters
    new_params = {}
    new_params["bias_eta_LYA"] = beta * bias / f_sig_8 * sig_8
    new_params["bias_LYA_sig_8"] = bias * sig_8
    # beta * bias * sig_8 = b_eta f sig_8
    new_params["bias_eta_LYA_f_sig_8"] = beta * bias * sig_8
    # add them to the list of entries
    new_params_entries.append({k: new_params[k] for k in new_params_names})
    if verbose:
        print("new params", new_params_entries[-1])

# setup numpy arrays with new parameters
b_eta = np.array([new_params_entries[i]["bias_eta_LYA"] for i in range(Nsamp)])
b_sig_8 = np.array([new_params_entries[i]["bias_LYA_sig_8"] for i in range(Nsamp)])
b_eta_f_sig_8 = np.array(
    [new_params_entries[i]["bias_eta_LYA_f_sig_8"] for i in range(Nsamp)]
)

# mean and uncertainties
mean_b_eta = np.mean(b_eta)
var_b_eta = np.var(b_eta)
print("b_eta = {} +/- {}".format(mean_b_eta, np.sqrt(var_b_eta)))
mean_b_sig_8 = np.mean(b_sig_8)
var_b_sig_8 = np.var(b_sig_8)
print("b_sig_8 = {} +/- {}".format(mean_b_sig_8, np.sqrt(var_b_sig_8)))
mean_b_eta_f_sig_8 = np.mean(b_eta_f_sig_8)
var_b_eta_f_sig_8 = np.var(b_eta_f_sig_8)
print(
    "b_eta_f_sig_8 = {} +/- {}".format(mean_b_eta_f_sig_8, np.sqrt(var_b_eta_f_sig_8))
)

# +
# from P1D we have:
#      b sig_8 = -0.0363 +/- 0.0018
#      b_eta_f_sig_8 = -0.0518 +/ 0.0012
# but this can't be true...
# -

np.corrcoef(b_sig_8, b_eta_f_sig_8)

# add derived parameters
samples.addDerived(b_eta, "bias_eta_LYA", label="b_\\eta")
samples.addDerived(b_sig_8, "bias_LYA_sig_8", label="b_\\alpha \\, \\sigma_8(z)")
samples.addDerived(
    b_eta_f_sig_8, "bias_eta_LYA_f_sig_8", label="b_\\eta \\, f \\, \\sigma_8(z)"
)

g = plots.getSubplotPlotter(width_inch=6)
g.settings.axes_fontsize = 9
g.settings.legend_fontsize = 11
g.triangle_plot(
    [samples],
    ["bias_LYA", "beta_LYA", "bias_eta_LYA"],
    legend_labels=[r"DESI DR2 LYA BAO"],
)

g = plots.getSubplotPlotter(width_inch=6)
g.settings.axes_fontsize = 9
g.settings.legend_fontsize = 11
g.triangle_plot(
    [samples],
    ["bias_LYA_sig_8", "bias_eta_LYA_f_sig_8"],
    legend_labels=[r"DESI DR2 LYA BAO"],
)




