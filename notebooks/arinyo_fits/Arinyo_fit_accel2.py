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
# # Fit the Arinyo model to an ACCEL2 snapshot
#
# This notebook fits the eight Arinyo parameters jointly to ACCEL2 P3D and
# P1D measurements. It follows `Arinyo_fit_astrid.py`: ACCEL2 is an external
# measurement rather than a ForestFlow archive entry, so it uses
# `ArinyoFitter.prepare_simulation_external`.
#
# The data are read with the same cup1d loader used for the ACCEL2 P3D
# validation figure (`set_samples.load_k_mu_accel2`). It reads the native
# measurements from the simulation (`p3d_from_sim`, all k and the four mu bins,
# and `p1d_from_sim`, averaged over the three directions), not the rebinned
# P3D of `p3d_rebinned`. It converts to Mpc$^{-1}$, Mpc$^3$ and Mpc with
# h=0.675 and moves k and mu from the lower bin edges to the bin centres. The
# fitter expects *fractional* uncertainties: P3D combines a 4% fitting floor
# with Gaussian finite-volume variance (reduced by a tunable factor for the
# average over the three directions), P1D uses a 1.5% weighting.
# The objective is the sum of mean squared normalized P3D and P1D residuals,
# not a statistical chi-squared.
#
# This is a **hybrid bin-averaged fit**, as for MP-Gadget: the model P3D is
# averaged over the exact lattice modes of the ACCEL2 box in cells with at most
# 256 modes, and over the continuous (k, mu) bin above. The ACCEL2 bins are
# linear in k, of width the fundamental mode and centred on its multiples,
# with four mu bins of width 0.25.

# %%
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

from lace.cosmo import cosmology
from lace.cosmo.labeled_cosmologies import get_cosmo_params_dict_from_label
from cup1d.p1ds.simulations.data_accel2 import load_data

from forestflow.model_fits import ArinyoFitter, gaussian_p3d_relative_error
from forestflow.statistics.rebin_p3d import get_P3D_k_mu_modes_from_edges
from forestflow.priors.paper.plots import ACCEL2_ARINYO_FITS

# %% [markdown]
# ## Load one ACCEL2 snapshot
#
# `l160_r25` is the fiducial ACCEL2 run (160 Mpc/h box, 6144^3 grid), the
# default of the cup1d loader and of the validation figure. Snapshots are
# available at z = 2.0, 2.6, 3.0, 3.6, 4.0 and 5.0 (z=5 was not tested in
# ACCEL2).

# %%
ACCEL2_FOLDER = "/home/jchaves/Proyectos/projects/lya/data/accel2/frontier_grid/"
SIM_LABEL = "l160_r25"
Z_FIT = 3.0
# z = 2.0, 2.6, 3.0, 3.6, 4.0 and 5.0

cosmo_params = get_cosmo_params_dict_from_label("accel2")
hh = cosmo_params["H0"] / 100

# kmax=np.inf: load all k; the fitted range is set by the cuts below
data_all = load_data(ACCEL2_FOLDER, sim_label=SIM_LABEL, hh=hh, kmax=np.inf)
iz = np.argmin(np.abs(data_all["z"] - Z_FIT))
z = data_all["z"][iz]
print(f"Loaded ACCEL2 {SIM_LABEL} at z={z:.2f}, h={hh}")

# %% [markdown]
# ## Select the fitted scales
#
# Empty cells (only the lowest-k bin with 0.25 < mu < 0.5) are set to NaN and
# kept: as in the MP-Gadget fits, the model is NaN there and the objective
# skips them, so the other cells of the same k bin are still fitted. P1D is the
# average over the three directions.

# %%
# kmin_3d_Mpc, kmax_3d_Mpc = 0.001, 5.0
# kmin_1d_Mpc, kmax_1d_Mpc = 0.001, 3.0
kmin_3d_Mpc, kmax_3d_Mpc = 0.001, 5.0
kmin_1d_Mpc, kmax_1d_Mpc = 0.001, 5.0

k3d_all = np.asarray(data_all["k3d_Mpc"], dtype=float)
mu3d_all = np.asarray(data_all["mu3d"], dtype=float)
power_all = np.asarray(data_all["p3d_Mpc"][iz], dtype=float)
valid_p3d = np.isfinite(power_all) & (power_all > 0)
power_all = np.where(valid_p3d, power_all, np.nan)

mask_3d = (k3d_all[:, 0] > kmin_3d_Mpc) & (k3d_all[:, 0] <= kmax_3d_Mpc)
k3d_Mpc = k3d_all[mask_3d]
mu3d = mu3d_all[mask_3d]
p3d_Mpc = power_all[mask_3d]
valid_sel = valid_p3d[mask_3d]

k1d_all_Mpc = np.asarray(data_all["k1d_Mpc"], dtype=float)
mask_1d = (k1d_all_Mpc > kmin_1d_Mpc) & (k1d_all_Mpc <= kmax_1d_Mpc)
k1d_Mpc = k1d_all_Mpc[mask_1d]
p1d_Mpc = np.asarray(data_all["p1d_Mpc"][iz], dtype=float)[mask_1d]

# %% [markdown]
# ## Bin geometry and Gaussian sample variance for P3D
#
# The bin edges of the selected rows follow from the uniform bin width (the
# fundamental mode, 2 pi / L) and the bin centres. The hybrid averaging needs
# them, together with the lattice modes of each cell, which
# `get_P3D_k_mu_modes_from_edges` builds for the ACCEL2 box.
#
# The mode counts used for the errors are those of this lattice, for a single
# box and line-of-sight direction. For a real Gaussian field,
# Var(P)/P² = 2/N_full, where N_full counts both +k and -k. This is an
# approximation for nonlinear flux power; it neglects mode coupling and
# off-diagonal covariance.
#
# The P3D is the average of the three line-of-sight directions. These are
# strongly correlated, but not completely, so the average is somewhat more
# precise than a single direction: the Gaussian error is divided by
# `P3D_GAUSSIAN_FACTOR` (1 for a single direction, sqrt(3) for independent
# directions). The total P3D error adds the floor `P3D_FRACTIONAL_FLOOR` in
# quadrature; P1D uses a fixed fractional error `P1D_FRACTIONAL_FLOOR`.

# %%
P3D_GAUSSIAN_FACTOR = 2.0  # Gaussian P3D error divided by this factor
P3D_FRACTIONAL_FLOOR = 0.03  # Set to zero for Gaussian-only weights.
P1D_FRACTIONAL_FLOOR = 0.01
boxsize_Mpc = 160 / hh
print("ACCEL2 box side [Mpc]:", boxsize_Mpc)

dk_Mpc = k3d_all[1, 0] - k3d_all[0, 0]
k3d_edges = np.append(k3d_Mpc[:, 0] - 0.5 * dk_Mpc, k3d_Mpc[-1, 0] + 0.5 * dk_Mpc)
mu_edges = np.linspace(0, 1, mu3d.shape[1] + 1)

k_mu_geometry = get_P3D_k_mu_modes_from_edges(
    k3d_edges, mu_edges, Lbox_Mpc=boxsize_Mpc, max_discrete_modes=256
)
mode_counts_full = k_mu_geometry[1]

# check: the lattice and the data must populate the same cells, and the lattice
# counts should be proportional to the counts stored in the data files
if not np.array_equal(mode_counts_full > 0, valid_sel):
    raise ValueError("The lattice and the data populate different cells.")
count_ratio = data_all["nmod3d"][mask_3d][valid_sel] / mode_counts_full[valid_sel]
print("file / lattice mode counts: min", count_ratio.min(), "max", count_ratio.max())

gaussian_fractional_p3d = np.full_like(p3d_Mpc, np.nan)
gaussian_fractional_p3d[valid_sel] = (
    gaussian_p3d_relative_error(mode_counts_full[valid_sel], count_convention="full")
    / P3D_GAUSSIAN_FACTOR
)
fractional_p3d = np.sqrt(gaussian_fractional_p3d**2 + P3D_FRACTIONAL_FLOOR**2)
gaussian_std = p3d_Mpc * gaussian_fractional_p3d

# %%
fig, ax = plt.subplots(figsize=(7, 4))
for index in range(p3d_Mpc.shape[1]):
    sel = valid_sel[:, index]
    k_Mpc = k3d_Mpc[sel, index]
    label = rf"$\mu$={np.mean(mu3d[:, index]):.2f}"
    ax.errorbar(
        k_Mpc,
        k_Mpc**2 * p3d_Mpc[sel, index],
        yerr=k_Mpc**2 * gaussian_std[sel, index],
        label=label,
    )
ax.set(xscale="log", yscale="log", xlabel=r"$k$ [Mpc$^{-1}$]", ylabel=r"$k^2 P_{3D}$")
ax.legend(ncol=2, fontsize=8)

# %% [markdown]
# ## Prepare and run the fit
#
# The initial point is the ACCEL2 article fit at this redshift (with
# bias_eta = beta * bias / f and kvav = k_v^a_v), clipped to the bounds. The
# bounds are wider than for Astrid to cover the ACCEL2 redshift range.

# %%
f_growth = cosmology.Cosmology(cosmo_params).get_growth_rate(z)

iz_sim = np.argmin(np.abs(ACCEL2_ARINYO_FITS["z"] - z))
sim_fit = {
    key: ACCEL2_ARINYO_FITS[key]["value"][iz_sim]
    for key in ACCEL2_ARINYO_FITS
    if key != "z"
}
print(f"ACCEL2 article fit at z={ACCEL2_ARINYO_FITS['z'][iz_sim]}:", sim_fit)

bounds = [
    (-0.6, -0.05),
    (-1.0, -0.05),
    (0.05, 3.0),
    (-1.5, 1.0),
    (0.05, 2.0),
    (-0.5, 1.0),
    (1.0, 2.4),
    (4.0, 30.0),
]
initial_parameters = {
    "bias": sim_fit["bias_delta"],
    "bias_eta": sim_fit["beta"] * sim_fit["bias_delta"] / f_growth,
    "q1": sim_fit["q1"],
    "q2": sim_fit["q2"],
    "kvav": sim_fit["k_v"] ** sim_fit["a_v"],
    "av": sim_fit["a_v"],
    "bv": sim_fit["b_v"],
    "kp": sim_fit["k_p"],
}
for (low, high), key in zip(bounds, initial_parameters):
    initial_parameters[key] = float(np.clip(initial_parameters[key], low, high))

fitter = ArinyoFitter(bounds=bounds)
fitter.prepare_simulation_external(
    z=z,
    cosmo_params=cosmo_params,
    k3d_Mpc=k3d_Mpc,
    mu3d=mu3d,
    p3d_Mpc=p3d_Mpc,
    std_p3d=fractional_p3d,
    k1d_Mpc=k1d_Mpc,
    p1d_Mpc=p1d_Mpc,
    std_p1d=np.full_like(p1d_Mpc, P1D_FRACTIONAL_FLOOR),
    ini_params=initial_parameters,
    k3d_edges=k3d_edges,
    mu_edges=mu_edges,
    k_mu_geometry=k_mu_geometry,
)
# repeated bounded Nelder-Mead in unit-cube coordinates, as in cup1d, with an
# absolute tolerance of 0.1 on the objective
result = fitter.fit_nelder_mead(chi2_tol=0.1)
best_parameters = fitter.params_to_dict(result.x)
print(f"success={result.success}; chi2={result.fun:.4f}")
best_parameters

# %% [markdown]
# ## Parameters near the bounds and comparison with the ACCEL2 article fit
#
# Parameters within 2% of the bound range from a bound are listed, since the
# fit result there is set by the bound. The comparison table converts to the
# article parametrization: beta = bias_eta * f / bias and k_v = kvav^(1/av).

# %%
for (low, high), key in zip(bounds, best_parameters):
    val = best_parameters[key]
    margin = 0.02 * (high - low)
    if (val - low < margin) or (high - val < margin):
        print(f"near bound: {key} = {val:.4f}, bounds ({low}, {high})")

best_article = {
    "bias_delta": best_parameters["bias"],
    "beta": best_parameters["bias_eta"] * f_growth / best_parameters["bias"],
    "q1": best_parameters["q1"],
    "q2": best_parameters["q2"],
    "k_v": best_parameters["kvav"] ** (1 / best_parameters["av"]),
    "a_v": best_parameters["av"],
    "b_v": best_parameters["bv"],
    "k_p": best_parameters["kp"],
}
print("par  this fit  ACCEL2 article  (error)")
for key, val in best_article.items():
    err = ACCEL2_ARINYO_FITS[key]["error"][iz_sim]
    print(f"{key:10s} {val:9.4f} {sim_fit[key]:9.4f}  ({err:.4f})")

# %%
k3d 5, k1d 3
par  this fit  ACCEL2 article  (error)
bias_delta   -0.2131   -0.2090  (0.0045)
beta          1.3729    1.4200  (0.0774)
q1            0.7961    1.0800  (0.1710)
q2            0.0903   -0.1560  (0.1920)
k_v           0.4585    1.0100  (0.3240)
a_v           0.3768    0.4450  (0.1000)
b_v           1.7696    1.7400  (0.1090)
k_p          11.5547   21.0000  (6.8900)

k3d 6, k1d 6
par  this fit  ACCEL2 article  (error)
bias_delta   -0.2170   -0.2090  (0.0045)
beta          1.3433    1.4200  (0.0774)
q1            0.6487    1.0800  (0.1710)
q2            0.2143   -0.1560  (0.1920)
k_v           0.4019    1.0100  (0.3240)
a_v           0.3622    0.4450  (0.1000)
b_v           1.8053    1.7400  (0.1090)
k_p          10.9597   21.0000  (6.8900)

# %% [markdown]
# ## Compare the fitted model and data
#
# The residual bands show the effective fractional uncertainties used in the
# objective. They are diagnostics of this fit, not inferred parameter errors.

# %%
p3d_fit, p1d_fit = fitter.predict(result.x)
p3d_data = fitter.data.p3d
p1d_data = fitter.data.p1d

fig, axes = plt.subplots(2, figsize=(8, 7), sharex=False)
for index in range(0, p3d_data.shape[1], 1):
    color = f"C{index}"
    k_Mpc = fitter.data.k3d[:, index]
    factor = k_Mpc**3 / (2 * np.pi**2)
    axes[0].plot(k_Mpc, factor * p3d_data[:, index], color=color)
    axes[0].plot(k_Mpc, factor * p3d_fit[:, index], "--", color=color)
axes[0].set(xscale="log", yscale="log", ylabel=r"$k^3 P_{3D}/(2\pi^2)$")

axes[1].plot(fitter.data.k1d, fitter.data.k1d * p1d_data / np.pi, label="ACCEL2")
axes[1].plot(fitter.data.k1d, fitter.data.k1d * p1d_fit / np.pi, "--", label="Arinyo fit")
axes[1].set(xscale="log", xlabel=r"$k_\parallel$ [Mpc$^{-1}$]", ylabel=r"$k_\parallel P_{1D}/\pi$")
axes[1].legend()
fig.tight_layout()

# %% [markdown]
# ## P3D relative to the linear matter power
#
# Both the measured flux power and the best-fitting Arinyo prediction are
# divided by the same linear matter spectrum, evaluated at the bin centres.
# Points represent data and dashed curves the best-fitting model; matching
# colors identify mu bins.

# %%
fig, ax = plt.subplots(figsize=(9, 6))
colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, p3d_data.shape[1]))
for index in range(0, p3d_data.shape[1]):
    k_Mpc = fitter.data.k3d[:, index]
    valid = (
        np.isfinite(k_Mpc)
        & (k_Mpc > 0)
        & np.isfinite(p3d_data[:, index])
        & np.isfinite(p3d_fit[:, index])
    )
    if not np.any(valid):
        continue
    plin_Mpc = fitter.data.power_model.linear.get_linP_Mpc(
        fitter.data.linear,
        fitter.data.z,
        k_Mpc[valid],
    )
    mu_label = np.mean(fitter.data.mu3d[valid, index])
    ax.plot(
        k_Mpc[valid],
        p3d_data[valid, index] / plin_Mpc,
        ".:",
        color=colors[index],
        label=rf"$\mu\simeq {mu_label:.2f}$",
    )
    ax.plot(
        k_Mpc[valid],
        p3d_fit[valid, index] / plin_Mpc,
        "--",
        color=colors[index],
    )
ax.set(
    xscale="log",
    yscale="log",
    xlabel=r"$k$ [Mpc$^{-1}$]",
    ylabel=r"$P_{3D}/P_{\mathrm{lin}}$",
    title="ACCEL2: data (points) and best-fitting Arinyo model (dashed)",
)
ax.legend(ncol=4, fontsize=8)
fig.tight_layout()

# %%
fig, axes = plt.subplots(2, figsize=(8, 7), sharex=False)
for index in range(0, p3d_data.shape[1], 1):
    color = f"C{index}"
    axes[0].plot(
        fitter.data.k3d[:, index],
        p3d_fit[:, index] / p3d_data[:, index] - 1.0,
        color=color,
    )
    if index == 0:
        axes[0].fill_between(
            fitter.data.k3d[:, index],
            -fitter.data.std_p3d[:, index],
            fitter.data.std_p3d[:, index],
            color=color,
            alpha=0.15,
            label=r"Total fitting error ($\mu\simeq 0$)",
        )
        gaussian_fractional = gaussian_std[:, index] / p3d_data[:, index]
        axes[0].plot(
            fitter.data.k3d[:, index],
            gaussian_fractional,
            "--",
            color=color,
            label=r"Gaussian diagonal error ($\mu\simeq 0$)",
        )
        axes[0].plot(
            fitter.data.k3d[:, index],
            -gaussian_fractional,
            "--",
            color=color,
        )
axes[0].set(xscale="log", ylabel="P3D fractional residual", ylim=(-0.2, 0.2))
axes[0].legend()

axes[1].plot(fitter.data.k1d, p1d_fit / p1d_data - 1.0)
axes[1].axhspan(-P1D_FRACTIONAL_FLOOR, P1D_FRACTIONAL_FLOOR, color="k", alpha=0.15)
axes[1].set(
    xscale="log", xlabel=r"$k_\parallel$ [Mpc$^{-1}$]", ylabel="P1D fractional residual"
)
fig.tight_layout()

# %%

# %%
