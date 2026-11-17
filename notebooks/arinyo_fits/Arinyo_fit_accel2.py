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
Z_FIT = 2.0
# z = 2.0, 2.6, 3.0, 3.6, 4.0 and 5.0

cosmo_params = get_cosmo_params_dict_from_label("accel2")
hh = cosmo_params["H0"] / 100
# one Cosmology for the whole notebook: CAMB runs once (on a z grid up to 10)
# and is reused for the growth rate and the fits at every redshift
cosmo = cosmology.Cosmology(cosmo_params)

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
kmin_3d_Mpc, kmax_3d_Mpc = 0.001, 3.5
kmin_1d_Mpc, kmax_1d_Mpc = 0.04, 1.0

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
P1D_FRACTIONAL_FLOOR = 0.015

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
f_growth = cosmo.get_growth_rate(z)

iz_sim = np.argmin(np.abs(ACCEL2_ARINYO_FITS["z"] - z))
sim_fit = {
    key: ACCEL2_ARINYO_FITS[key]["value"][iz_sim]
    for key in ACCEL2_ARINYO_FITS
    if key != "z"
}
print(f"ACCEL2 article fit at z={ACCEL2_ARINYO_FITS['z'][iz_sim]}:", sim_fit)

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
    cosmo_params=cosmo,
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
bias_delta   -0.2154   -0.2090  (0.0045)
beta          1.3571    1.4200  (0.0774)
q1            0.6939    1.0800  (0.1710)
q2            0.1864   -0.1560  (0.1920)
k_v           0.4131    1.0100  (0.3240)
a_v           0.3630    0.4450  (0.1000)
b_v           1.7853    1.7400  (0.1090)
k_p          11.1133   21.0000  (6.8900)

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


# %% [markdown]
# ## Fit all ACCEL2 redshifts
#
# `fit_accel2_redshifts` repeats the fit above at each snapshot. Several inputs do
# not change with redshift and are built once: the k and mu grids, the scale
# cuts, the bin edges, the lattice modes of each cell (the slow part), the
# Gaussian fractional errors (they depend only on the mode counts), the
# `Cosmology` (CAMB) and the fitter. At each redshift the loop checks that the
# populated cells still match the lattice. The starting point is the ACCEL2
# article fit at the nearest redshift, clipped to the bounds. z=5 has no article
# fit, so it starts from the z=4 values.
#
# The results are saved as a dictionary with `np.save`. Load them with
# `np.load(file, allow_pickle=True).item()`. `Arinyo` holds the fitted parameters
# in this fitter's parametrization and `Arinyo_article` the same fits converted
# to the article's (beta, k_v). Each entry is an array over `z`.

# %%
def fit_accel2_redshifts(
    data_all,
    cosmo,
    bounds,
    z_list=None,
    kmin_3d_Mpc=kmin_3d_Mpc,
    kmax_3d_Mpc=kmax_3d_Mpc,
    kmin_1d_Mpc=kmin_1d_Mpc,
    kmax_1d_Mpc=kmax_1d_Mpc,
    p3d_gaussian_factor=P3D_GAUSSIAN_FACTOR,
    p3d_fractional_floor=P3D_FRACTIONAL_FLOOR,
    p1d_fractional_floor=P1D_FRACTIONAL_FLOOR,
    max_discrete_modes=256,
    chi2_tol=0.1,
    bound_margin=0.02,
):
    """Fit the Arinyo model to every requested ACCEL2 snapshot.

    Parameters
    ----------
    data_all : dict
        Output of ``cup1d ... data_accel2.load_data`` (Mpc units).
    cosmo : lace.cosmo.cosmology.Cosmology
        Simulation cosmology, built once and reused at every redshift.
    bounds : list of tuple
        Flat bounds in ``ArinyoFitter.PARAM_NAMES`` order.
    z_list : array_like, optional
        Redshifts to fit; defaults to all snapshots in ``data_all``.

    Returns
    -------
    dict
        Fitted parameters (fitter and article parametrizations), objective,
        convergence flags, parameters near the bounds, and the fit settings.
    """
    param_names = ArinyoFitter.PARAM_NAMES
    z_data = np.asarray(data_all["z"], dtype=float)
    z_list = z_data if z_list is None else np.atleast_1d(np.asarray(z_list, dtype=float))
    iz_list = [int(np.argmin(np.abs(z_data - zz))) for zz in z_list]

    # ---- redshift-independent inputs: grids, cuts, edges, lattice modes, errors
    k3d_all = np.asarray(data_all["k3d_Mpc"], dtype=float)
    mu3d_all = np.asarray(data_all["mu3d"], dtype=float)
    mask_3d = (k3d_all[:, 0] > kmin_3d_Mpc) & (k3d_all[:, 0] <= kmax_3d_Mpc)
    k3d_sel = k3d_all[mask_3d]
    mu3d_sel = mu3d_all[mask_3d]

    k1d_all = np.asarray(data_all["k1d_Mpc"], dtype=float)
    mask_1d = (k1d_all > kmin_1d_Mpc) & (k1d_all <= kmax_1d_Mpc)
    k1d_sel = k1d_all[mask_1d]
    std_p1d = np.full_like(k1d_sel, p1d_fractional_floor)

    hh_sim = cosmo.get_h()
    box_Mpc = 160 / hh_sim
    dk = k3d_all[1, 0] - k3d_all[0, 0]
    k_edges = np.append(k3d_sel[:, 0] - 0.5 * dk, k3d_sel[-1, 0] + 0.5 * dk)
    mu_edges_sel = np.linspace(0, 1, mu3d_sel.shape[1] + 1)
    geometry = get_P3D_k_mu_modes_from_edges(
        k_edges, mu_edges_sel, Lbox_Mpc=box_Mpc, max_discrete_modes=max_discrete_modes
    )
    populated = geometry[1] > 0

    gaussian_frac = np.full(k3d_sel.shape, np.nan)
    gaussian_frac[populated] = (
        gaussian_p3d_relative_error(geometry[1][populated], count_convention="full")
        / p3d_gaussian_factor
    )
    frac_p3d = np.sqrt(gaussian_frac**2 + p3d_fractional_floor**2)

    fitter_z = ArinyoFitter(bounds=bounds)

    n_z = len(iz_list)
    out = {
        "schema_version": 1,
        "simulation_label": SIM_LABEL,
        "parameter_names": param_names,
        "z": z_data[iz_list],
        "f_growth": np.full(n_z, np.nan),
        "Arinyo": {key: np.full(n_z, np.nan) for key in param_names},
        "Arinyo_article": {
            key: np.full(n_z, np.nan)
            for key in ["bias_delta", "beta", "q1", "q2", "k_v", "a_v", "b_v", "k_p"]
        },
        "initial_parameters": {key: np.full(n_z, np.nan) for key in param_names},
        "chi2": np.full(n_z, np.nan),
        "success": np.zeros(n_z, dtype=bool),
        "message": np.empty(n_z, dtype=object),
        "near_bound": [],
        "bounds": list(bounds),
        "settings": {
            "kmin_3d_Mpc": kmin_3d_Mpc,
            "kmax_3d_Mpc": kmax_3d_Mpc,
            "kmin_1d_Mpc": kmin_1d_Mpc,
            "kmax_1d_Mpc": kmax_1d_Mpc,
            "p3d_gaussian_factor": p3d_gaussian_factor,
            "p3d_fractional_floor": p3d_fractional_floor,
            "p1d_fractional_floor": p1d_fractional_floor,
            "max_discrete_modes": max_discrete_modes,
            "chi2_tol": chi2_tol,
            "boxsize_Mpc": box_Mpc,
        },
    }

    for index, iz_fit in enumerate(iz_list):
        z_fit = z_data[iz_fit]

        # ---- redshift-dependent data
        power = np.asarray(data_all["p3d_Mpc"][iz_fit], dtype=float)[mask_3d]
        valid = np.isfinite(power) & (power > 0)
        if not np.array_equal(populated, valid):
            raise ValueError(f"z={z_fit}: the lattice and the data populate different cells.")
        power = np.where(valid, power, np.nan)
        p1d = np.asarray(data_all["p1d_Mpc"][iz_fit], dtype=float)[mask_1d]

        f_z = cosmo.get_growth_rate(z_fit)

        # ---- initial point: article fit at the nearest redshift, clipped
        iz_art = np.argmin(np.abs(ACCEL2_ARINYO_FITS["z"] - z_fit))
        art = {
            key: ACCEL2_ARINYO_FITS[key]["value"][iz_art]
            for key in ACCEL2_ARINYO_FITS
            if key != "z"
        }
        ini = {
            "bias": art["bias_delta"],
            "bias_eta": art["beta"] * art["bias_delta"] / f_z,
            "q1": art["q1"],
            "q2": art["q2"],
            "kvav": art["k_v"] ** art["a_v"],
            "av": art["a_v"],
            "bv": art["b_v"],
            "kp": art["k_p"],
        }
        for (low, high), key in zip(bounds, param_names):
            ini[key] = float(np.clip(ini[key], low, high))

        fitter_z.prepare_simulation_external(
            z=z_fit,
            cosmo_params=cosmo,
            k3d_Mpc=k3d_sel,
            mu3d=mu3d_sel,
            p3d_Mpc=power,
            std_p3d=frac_p3d,
            k1d_Mpc=k1d_sel,
            p1d_Mpc=p1d,
            std_p1d=std_p1d,
            ini_params=ini,
            k3d_edges=k_edges,
            mu_edges=mu_edges_sel,
            k_mu_geometry=geometry,
            max_discrete_modes=max_discrete_modes,
        )
        res = fitter_z.fit_nelder_mead(chi2_tol=chi2_tol)
        best = fitter_z.params_to_dict(res.x)

        # ---- store
        out["f_growth"][index] = f_z
        out["chi2"][index] = res.fun
        out["success"][index] = res.success
        out["message"][index] = str(getattr(res, "message", ""))
        for key in param_names:
            out["Arinyo"][key][index] = best[key]
            out["initial_parameters"][key][index] = ini[key]
        article = out["Arinyo_article"]
        article["bias_delta"][index] = best["bias"]
        article["beta"][index] = best["bias_eta"] * f_z / best["bias"]
        article["q1"][index] = best["q1"]
        article["q2"][index] = best["q2"]
        article["k_v"][index] = best["kvav"] ** (1 / best["av"])
        article["a_v"][index] = best["av"]
        article["b_v"][index] = best["bv"]
        article["k_p"][index] = best["kp"]

        near = []
        for (low, high), key in zip(bounds, param_names):
            margin = bound_margin * (high - low)
            if (best[key] - low < margin) or (high - best[key] < margin):
                near.append(key)
        out["near_bound"].append(near)

        print(f"z={z_fit:.2f}: success={res.success}; chi2={res.fun:.4f}")
        for key in near:
            low, high = bounds[param_names.index(key)]
            print(f"    near bound: {key} = {best[key]:.4f}, bounds ({low}, {high})")

    return out


# %%
from pathlib import Path

# all snapshots (z = 2.0, 2.6, 3.0, 3.6, 4.0, 5.0); pass z_list to fit a subset
accel2_fits = fit_accel2_redshifts(data_all, cosmo, bounds, z_list=[2.0, 2.6, 3.0, 3.6, 4.0])

results_file = Path(f"Arinyo_fit_accel2_{SIM_LABEL}_redshift_evolution.npy")
np.save(results_file, accel2_fits)
print(f"Saved ACCEL2 fits to {results_file.resolve()}")

# reload with:
# accel2_fits = np.load(results_file, allow_pickle=True).item()

# %%
accel2_fits

# %%
for key in accel2_fits.keys():
    print(f"{key:10s} {accel2_fits[key]}")

# %%

# %% [markdown]
# ## MCMC with emcee at the same redshift
#
# This section is self-contained: after a kernel restart, run it from here
# down without running the cells above. The next cell sets the snapshot and fit
# settings, which are copies of those above (edit both if you change them). It
# then loads the data and prepares `fitter`, and finds the Nelder-Mead best fit
# used as the starting point:
#
# 1. from `Arinyo_fit_accel2_<sim>_redshift_evolution.npy` (saved by the
#    all-redshifts fit), if it contains this redshift with the same settings;
# 2. otherwise by running the minimizer here.
#
# If this configuration's MCMC file already exists, the walkers restart from its
# last positions instead (see the test run).
#
# **Pseudo-likelihood.** We sample the same objective as the minimizer,
# `fitter.chi2`, which adds the mean squared normalized residuals of P3D and
# P1D, so both spectra get the same weight:
#
# $$-2\ln \mathcal{L} = \mathcal{N}\left[\left\langle\left(\frac{P^{\rm model}_{\rm 3D}/P^{\rm data}_{\rm 3D}-1}{\sigma_{\rm 3D}}\right)^2\right\rangle + \left\langle\left(\frac{P^{\rm model}_{\rm 1D}/P^{\rm data}_{\rm 1D}-1}{\sigma_{\rm 1D}}\right)^2\right\rangle\right],$$
#
# with flat priors within `bounds`. The maximum of the pseudo-likelihood is the
# minimizer's best fit. The errors of neither probe are well determined, so
# this is a pseudo-likelihood. The normalization $\mathcal{N}$
# (`PSEUDO_LIKE_NORM`) sets the width of the posterior but not its shape or
# maximum. By default it is the mean number of points of the two probes,
# $(N_{\rm 3D}+N_{\rm 1D})/2$. Each probe then counts as if it had that many
# points, which is the scale of a sum of squared residuals. Set it to 1 to
# sample `fitter.chi2` itself.
#
# Set `N_PROCESSES` > 1 to evaluate the walkers in parallel (fork-based pool,
# one BLAS thread per worker). Each worker is a copy of this kernel, so keep it
# modest on this machine. If the results file of this configuration already
# exists, the walkers restart from its last positions.

# %%
# ---- imports (the whole MCMC section only needs this cell and the next)
from pathlib import Path

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

# ---- settings (copies of those at the top of the notebook)
ACCEL2_FOLDER = "/home/jchaves/Proyectos/projects/lya/data/accel2/frontier_grid/"
SIM_LABEL = "l160_r25"
Z_FIT = 3.0
kmin_3d_Mpc, kmax_3d_Mpc = 0.001, 3.5
kmin_1d_Mpc, kmax_1d_Mpc = 0.04, 1.0
P3D_GAUSSIAN_FACTOR = 2.0
P3D_FRACTIONAL_FLOOR = 0.03
P1D_FRACTIONAL_FLOOR = 0.01
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

# ---- data at Z_FIT
cosmo_params = get_cosmo_params_dict_from_label("accel2")
hh = cosmo_params["H0"] / 100
cosmo = cosmology.Cosmology(cosmo_params)
data_all = load_data(ACCEL2_FOLDER, sim_label=SIM_LABEL, hh=hh, kmax=np.inf)
iz = np.argmin(np.abs(data_all["z"] - Z_FIT))
z = data_all["z"][iz]
print(f"Loaded ACCEL2 {SIM_LABEL} at z={z:.2f}")

k3d_all = np.asarray(data_all["k3d_Mpc"], dtype=float)
mask_3d = (k3d_all[:, 0] > kmin_3d_Mpc) & (k3d_all[:, 0] <= kmax_3d_Mpc)
k3d_Mpc = k3d_all[mask_3d]
mu3d = np.asarray(data_all["mu3d"], dtype=float)[mask_3d]
p3d_Mpc = np.asarray(data_all["p3d_Mpc"][iz], dtype=float)[mask_3d]
valid_sel = np.isfinite(p3d_Mpc) & (p3d_Mpc > 0)
p3d_Mpc = np.where(valid_sel, p3d_Mpc, np.nan)

k1d_all_Mpc = np.asarray(data_all["k1d_Mpc"], dtype=float)
mask_1d = (k1d_all_Mpc > kmin_1d_Mpc) & (k1d_all_Mpc <= kmax_1d_Mpc)
k1d_Mpc = k1d_all_Mpc[mask_1d]
p1d_Mpc = np.asarray(data_all["p1d_Mpc"][iz], dtype=float)[mask_1d]

# ---- bin geometry, lattice modes and fractional errors
boxsize_Mpc = 160 / hh
dk_Mpc = k3d_all[1, 0] - k3d_all[0, 0]
k3d_edges = np.append(k3d_Mpc[:, 0] - 0.5 * dk_Mpc, k3d_Mpc[-1, 0] + 0.5 * dk_Mpc)
mu_edges = np.linspace(0, 1, mu3d.shape[1] + 1)
k_mu_geometry = get_P3D_k_mu_modes_from_edges(
    k3d_edges, mu_edges, Lbox_Mpc=boxsize_Mpc, max_discrete_modes=256
)
mode_counts_full = k_mu_geometry[1]
if not np.array_equal(mode_counts_full > 0, valid_sel):
    raise ValueError("The lattice and the data populate different cells.")
gaussian_fractional_p3d = np.full_like(p3d_Mpc, np.nan)
gaussian_fractional_p3d[valid_sel] = (
    gaussian_p3d_relative_error(mode_counts_full[valid_sel], count_convention="full")
    / P3D_GAUSSIAN_FACTOR
)
fractional_p3d = np.sqrt(gaussian_fractional_p3d**2 + P3D_FRACTIONAL_FLOOR**2)

# ---- initial point: ACCEL2 article fit at the nearest redshift, clipped
f_growth = cosmo.get_growth_rate(z)
iz_sim = np.argmin(np.abs(ACCEL2_ARINYO_FITS["z"] - z))
sim_fit = {
    key: ACCEL2_ARINYO_FITS[key]["value"][iz_sim]
    for key in ACCEL2_ARINYO_FITS
    if key != "z"
}
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
    cosmo_params=cosmo,
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

# ---- Nelder-Mead best fit: saved all-redshift fit if it matches, else run it
x_best = None
fits_file = Path(f"Arinyo_fit_accel2_{SIM_LABEL}_redshift_evolution.npy")
if fits_file.is_file():
    saved = np.load(fits_file, allow_pickle=True).item()
    st = saved["settings"]
    same_settings = (
        [tuple(map(float, b)) for b in saved["bounds"]] == [tuple(map(float, b)) for b in bounds]
        and (st["kmin_3d_Mpc"], st["kmax_3d_Mpc"], st["kmin_1d_Mpc"], st["kmax_1d_Mpc"])
        == (kmin_3d_Mpc, kmax_3d_Mpc, kmin_1d_Mpc, kmax_1d_Mpc)
        and (st["p3d_gaussian_factor"], st["p3d_fractional_floor"], st["p1d_fractional_floor"])
        == (P3D_GAUSSIAN_FACTOR, P3D_FRACTIONAL_FLOOR, P1D_FRACTIONAL_FLOOR)
    )
    match_z = np.flatnonzero(np.isclose(saved["z"], z))
    if same_settings and match_z.size:
        x_best = np.array([saved["Arinyo"][key][match_z[0]] for key in ArinyoFitter.PARAM_NAMES])
        print(f"Nelder-Mead best fit read from {fits_file}")
    else:
        print(f"{fits_file} does not match these settings or z; running the minimizer")
if x_best is None:
    result = fitter.fit_nelder_mead(chi2_tol=0.1)
    x_best = np.asarray(result.x, dtype=float)
print(f"objective at the best fit: {fitter.chi2(x_best):.4f}")

# parameters near the bounds, and the best fit in the article parametrization
for (low, high), key, val in zip(bounds, ArinyoFitter.PARAM_NAMES, x_best):
    margin = 0.02 * (high - low)
    if (val - low < margin) or (high - val < margin):
        print(f"near bound: {key} = {val:.4f}, bounds ({low}, {high})")
best_parameters = fitter.params_to_dict(x_best)
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

# %%
import multiprocessing
import time
from pathlib import Path

import corner
import emcee
from threadpoolctl import threadpool_limits

PARAM_NAMES = ArinyoFitter.PARAM_NAMES
mcmc_bounds = np.asarray(bounds, dtype=float)
widths = mcmc_bounds[:, 1] - mcmc_bounds[:, 0]

n_p3d = int(np.isfinite(fitter.data.p3d).sum())
n_p1d = fitter.data.p1d.size
PSEUDO_LIKE_NORM = 0.5 * (n_p3d + n_p1d)  # 1: sample fitter.chi2 itself
print(f"P3D cells: {n_p3d}, P1D bins: {n_p1d}, PSEUDO_LIKE_NORM = {PSEUDO_LIKE_NORM}")


def log_prob_accel2(params):
    """Equal-weight P3D/P1D pseudo-log-likelihood with flat priors."""
    params = np.asarray(params, dtype=float)
    if np.any(params <= mcmc_bounds[:, 0]) or np.any(params >= mcmc_bounds[:, 1]):
        return -np.inf
    chi2_value = fitter.chi2(params)  # 1e100 when the model is not finite
    if not np.isfinite(chi2_value) or chi2_value >= 1e100:
        return -np.inf
    return -0.5 * PSEUDO_LIKE_NORM * chi2_value


def _single_thread_worker():
    threadpool_limits(limits=1)


def run_chain(p0, n_steps, n_processes=1):
    """Run an emcee EnsembleSampler from walker positions p0."""
    n_walkers, n_dim = p0.shape
    t0 = time.perf_counter()
    if n_processes > 1:
        with multiprocessing.get_context("fork").Pool(
            n_processes, initializer=_single_thread_worker
        ) as pool:
            chain_sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob_accel2, pool=pool)
            chain_sampler.run_mcmc(p0, n_steps, progress=True)
    else:
        with threadpool_limits(limits=1):
            chain_sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob_accel2)
            chain_sampler.run_mcmc(p0, n_steps, progress=True)
    print(f"Run time: {(time.perf_counter() - t0) / 60:.1f} min")
    return chain_sampler


def plot_log_prob_walkers(chain_sampler, n_burn=None, title=None):
    """Log-likelihood of every walker as a function of the step."""
    log_prob = chain_sampler.get_log_prob()  # (n_steps, n_walkers)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax in axes:
        ax.plot(log_prob, color="k", alpha=0.3, lw=0.7)
        ax.axhline(log_prob_accel2(x_best), color="C1", ls=":", label="Nelder-Mead best fit")
        if n_burn is not None:
            ax.axvline(n_burn, color="C3", ls="--", label="burn-in")
        ax.set_xlabel("step")
    axes[0].set_ylabel(r"$\ln\mathcal{L}$")
    # right panel: zoom on the last half of the chain
    last = log_prob[log_prob.shape[0] // 2 :]
    last = last[np.isfinite(last)]
    if last.size:
        spread = np.ptp(last)
        axes[1].set_ylim(last.min() - 0.5 * spread, last.max() + 0.2 * spread)
    axes[1].set_title("zoom on the second half")
    axes[0].legend(fontsize=8)
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()


print(f"ln L at the minimizer best fit: {log_prob_accel2(x_best):.2f}")

# %% [markdown]
# ### Cost estimate
#
# Time a few likelihood evaluations before launching the chain. The run makes
# about `N_WALKERS * N_STEPS` evaluations, divided by `N_PROCESSES` when the
# pool is used.

# %%
N_WALKERS = 32  # at least 2 * ndim; 4 * ndim is a common choice
N_STEPS = 3000
N_PROCESSES = 10  # > 1 for a fork-based pool

n_eval = 10
rng_time = np.random.default_rng(0)
widths = mcmc_bounds[:, 1] - mcmc_bounds[:, 0]
with threadpool_limits(limits=1):
    t0 = time.perf_counter()
    for _ in range(n_eval):
        log_prob_accel2(x_best + 1e-3 * widths * rng_time.standard_normal(len(x_best)))
    t_eval = (time.perf_counter() - t0) / n_eval
t_total = t_eval * N_WALKERS * N_STEPS / N_PROCESSES
print(f"{1e3 * t_eval:.1f} ms per evaluation (one thread)")
print(f"estimated run time: {t_total / 60:.1f} min for {N_WALKERS} walkers x {N_STEPS} steps")

# %% [markdown]
# ### Test run (100 steps)
#
# A short chain to check that the likelihood, the initialization and the
# parallel pool work, and to judge from the log-likelihood traces how long the
# burn-in is. The production run continues from where this one ends.

# %%
N_TEST_STEPS = 100

mcmc_settings = {
    "sim_label": SIM_LABEL,
    "z": float(fitter.data.z),
    "kmin_3d_Mpc": kmin_3d_Mpc,
    "kmax_3d_Mpc": kmax_3d_Mpc,
    "kmin_1d_Mpc": kmin_1d_Mpc,
    "kmax_1d_Mpc": kmax_1d_Mpc,
    "p3d_gaussian_factor": P3D_GAUSSIAN_FACTOR,
    "p3d_fractional_floor": P3D_FRACTIONAL_FLOOR,
    "p1d_fractional_floor": P1D_FRACTIONAL_FLOOR,
    "bounds": [tuple(map(float, b)) for b in bounds],
    "likelihood": "equal-weight pseudo-likelihood, -0.5 * norm * fitter.chi2",
    "pseudo_like_norm": float(PSEUDO_LIKE_NORM),
}
mcmc_file = Path(
    f"Arinyo_mcmc_accel2_{SIM_LABEL}_z{fitter.data.z:.1f}"
    f"_k3d{kmax_3d_Mpc:g}_k1d{kmax_1d_Mpc:g}.npy"
)

ndim = len(PARAM_NAMES)
rng = np.random.default_rng(1)
p0 = None
if mcmc_file.is_file():
    previous = np.load(mcmc_file, allow_pickle=True).item()
    if previous["settings"] == mcmc_settings and previous["last_positions"].shape == (N_WALKERS, ndim):
        p0 = previous["last_positions"]
        print(f"Starting from the last walker positions in {mcmc_file}")
    else:
        print(f"{mcmc_file} has different settings or walkers; starting from the minimizer")
if p0 is None:
    # small ball around the minimizer best fit, kept strictly inside the bounds
    eps = 1e-6 * widths
    centre = np.clip(x_best, mcmc_bounds[:, 0] + 1e3 * eps, mcmc_bounds[:, 1] - 1e3 * eps)
    p0 = centre + 1e-3 * widths * rng.standard_normal((N_WALKERS, ndim))
    p0 = np.clip(p0, mcmc_bounds[:, 0] + eps, mcmc_bounds[:, 1] - eps)
    print("Starting from a ball around the Nelder-Mead best fit")

initial_log_prob = np.array([log_prob_accel2(x) for x in p0])
if not np.all(np.isfinite(initial_log_prob)):
    raise ValueError("Some initial walkers have a non-finite log-likelihood")

test_sampler = run_chain(p0, N_TEST_STEPS, N_PROCESSES)
acc = test_sampler.acceptance_fraction
print(f"acceptance fraction: mean {acc.mean():.2f}, range [{acc.min():.2f}, {acc.max():.2f}]")
print(f"ln L: start max {initial_log_prob.max():.2f}, end max {test_sampler.get_log_prob()[-1].max():.2f}")
plot_log_prob_walkers(test_sampler, title=f"Test run, {N_TEST_STEPS} steps")

# %% [markdown]
# ### Production run
#
# It continues from the last positions of the test run. Choose `N_STEPS` from
# the cost estimate and the burn-in seen in the test traces.

# %%
sampler = run_chain(test_sampler.get_last_sample().coords, N_STEPS, N_PROCESSES)

# %% [markdown]
# ### Convergence, summary and saved results
#
# The burn-in is `N_BURN` steps (set by hand from the log-likelihood traces,
# or `None` for twice the largest integrated autocorrelation time), and the
# chain is thinned by half the smallest autocorrelation time. emcee recommends a chain longer than
# about 50 autocorrelation times for a reliable estimate; a warning is printed
# otherwise. The chain is saved in the fitter parametrization. The summary
# also gives beta = bias_eta f / bias and k_v = kvav^(1/av), the parameters of
# the ACCEL2 article.
#
# The parameter errors are also written to a small file next to the chain,
# `<chain file>_errors.npy`, for other figures. For each parameter it stores the
# posterior median (`value`), half the 16-84% range (`error`), the asymmetric
# errors, the mean and the standard deviation. These are given in the fitter
# parametrization (`Arinyo`) and in the article one (`Arinyo_article`, with the
# same `value`/`error` layout as `ACCEL2_ARINYO_FITS`).

# %%
acceptance = sampler.acceptance_fraction
tau = sampler.get_autocorr_time(quiet=True)
print(f"acceptance fraction: mean {acceptance.mean():.2f}, range [{acceptance.min():.2f}, {acceptance.max():.2f}]")
for key, t in zip(PARAM_NAMES, tau):
    print(f"tau {key:9s} {t:7.1f}")
if N_STEPS < 50 * np.max(tau):
    print(f"WARNING: N_STEPS={N_STEPS} < 50 tau_max = {50 * np.max(tau):.0f}; run longer")

N_BURN = 150  # steps discarded as burn-in; None: twice the largest autocorrelation time
n_burn = int(2 * np.max(tau)) if N_BURN is None else N_BURN
n_thin = max(1, int(0.5 * np.min(tau)))
samples = sampler.get_chain(discard=n_burn, thin=n_thin, flat=True)
log_post = sampler.get_log_prob(discard=n_burn, thin=n_thin, flat=True)
x_map = samples[np.argmax(log_post)]
print(f"{len(samples)} samples after burn-in {n_burn} and thinning {n_thin}")

# article parametrization
f_growth_z = cosmo.get_growth_rate(fitter.data.z)
samples_article = {
    "bias_delta": samples[:, 0],
    "beta": samples[:, 1] * f_growth_z / samples[:, 0],
    "q1": samples[:, 2],
    "q2": samples[:, 3],
    "k_v": samples[:, 4] ** (1 / samples[:, 5]),
    "a_v": samples[:, 5],
    "b_v": samples[:, 6],
    "k_p": samples[:, 7],
}

print("\npar        median   -1sigma   +1sigma   minimizer   MAP")
for ii, key in enumerate(PARAM_NAMES):
    lo, med, hi = np.percentile(samples[:, ii], [16, 50, 84])
    print(f"{key:9s} {med:8.4f} {med - lo:8.4f} {hi - med:8.4f} {x_best[ii]:10.4f} {x_map[ii]:8.4f}")

print("\npar         median   -1sigma   +1sigma   ACCEL2 article (error)")
for key, val in samples_article.items():
    lo, med, hi = np.percentile(val, [16, 50, 84])
    err = ACCEL2_ARINYO_FITS[key]["error"][iz_sim]
    print(f"{key:10s} {med:8.4f} {med - lo:8.4f} {hi - med:8.4f}   {sim_fit[key]:8.4f} ({err:.4f})")

# parameters at the flat prior bounds: MAP within 2% of the range, or a 1-99%
# posterior interval reaching that margin
print()
for ii, key in enumerate(PARAM_NAMES):
    low, high = mcmc_bounds[ii]
    margin = 0.02 * (high - low)
    p01, p99 = np.percentile(samples[:, ii], [1, 99])
    if (x_map[ii] - low < margin) or (high - x_map[ii] < margin) or (p01 - low < margin) or (high - p99 < margin):
        print(f"near bound: {key}: MAP {x_map[ii]:.4f}, 1-99% [{p01:.4f}, {p99:.4f}], bounds ({low}, {high})")

mcmc_results = {
    "schema_version": 1,
    "settings": mcmc_settings,
    "parameter_names": PARAM_NAMES,
    "chain": sampler.get_chain(),  # (n_steps, n_walkers, ndim), no burn-in removed
    "log_prob": sampler.get_log_prob(),
    "acceptance_fraction": acceptance,
    "autocorr_time": tau,
    "n_burn": n_burn,
    "n_thin": n_thin,
    "last_positions": sampler.get_last_sample().coords,
    "minimizer_best_fit": x_best,
    "map": x_map,
    "f_growth": f_growth_z,
}


def _posterior_summary(values):
    lo, med, hi = np.percentile(values, [16, 50, 84])
    return {
        "value": med,  # posterior median
        "error": 0.5 * (hi - lo),  # half of the 16-84% range
        "error_minus": med - lo,
        "error_plus": hi - med,
        "std": np.std(values),
        "mean": np.mean(values),
    }


# parameter errors: compact file for other figures. "Arinyo_article" follows
# the value/error layout of ACCEL2_ARINYO_FITS.
mcmc_errors = {
    "schema_version": 1,
    "settings": mcmc_settings,
    "z": float(fitter.data.z),
    "f_growth": f_growth_z,
    "n_samples": len(samples),
    "n_burn": n_burn,
    "n_thin": n_thin,
    "parameter_names": PARAM_NAMES,
    "Arinyo": {key: _posterior_summary(samples[:, ii]) for ii, key in enumerate(PARAM_NAMES)},
    "Arinyo_article": {key: _posterior_summary(val) for key, val in samples_article.items()},
    "minimizer_best_fit": dict(zip(PARAM_NAMES, x_best)),
    "minimizer_best_fit_article": best_article,
    "map": dict(zip(PARAM_NAMES, x_map)),
}
mcmc_results["errors"] = mcmc_errors

np.save(mcmc_file, mcmc_results)
print(f"\nSaved chain to {mcmc_file.resolve()}")

mcmc_errors_file = mcmc_file.with_name(mcmc_file.stem + "_errors.npy")
np.save(mcmc_errors_file, mcmc_errors)
print(f"Saved parameter errors to {mcmc_errors_file.resolve()}")
# reload with: np.load(mcmc_errors_file, allow_pickle=True).item()

# %% [markdown]
# ### Chains and corner plot
#
# The log-likelihood of each walker against the step, with the adopted
# burn-in, is followed by the parameter traces.
#
# The corner plot uses the article parametrization. Blue lines mark the ACCEL2
# article fit and orange lines the Nelder-Mead best fit.

# %%
plot_log_prob_walkers(sampler, n_burn=n_burn, title="Production run")

fig, axes = plt.subplots(ndim, figsize=(8, 2 * ndim), sharex=True)
chain_all = sampler.get_chain()
for ii, key in enumerate(PARAM_NAMES):
    axes[ii].plot(chain_all[:, :, ii], color="k", alpha=0.2, lw=0.5)
    axes[ii].axvline(n_burn, color="C3", ls="--")
    axes[ii].set_ylabel(key)
axes[-1].set_xlabel("step")
fig.tight_layout()

article_labels = {
    "bias_delta": r"$b_\delta$",
    "beta": r"$\beta$",
    "q1": r"$q_1$",
    "q2": r"$q_2$",
    "k_v": r"$k_v$",
    "a_v": r"$a_v$",
    "b_v": r"$b_v$",
    "k_p": r"$k_p$",
}
keys = list(samples_article)
best_article_mcmc = [best_article[key] for key in keys]
fig = corner.corner(
    np.column_stack([samples_article[key] for key in keys]),
    labels=[article_labels[key] for key in keys],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_fmt=".3f",
    truths=[sim_fit[key] for key in keys],
    truth_color="C0",
)
corner.overplot_lines(fig, best_article_mcmc, color="C1")
corner.overplot_points(fig, [best_article_mcmc], marker="s", color="C1")
fig.legend(
    handles=[
        plt.Line2D([], [], color="k", label="MCMC posterior"),
        plt.Line2D([], [], color="C0", marker="s", label="ACCEL2 article fit"),
        plt.Line2D([], [], color="C1", marker="s", label="Nelder-Mead best fit"),
    ],
    loc="upper right",
    fontsize=14,
    frameon=False,
)

# %% [markdown]
# ## P3D relative to linear theory: data, posterior and ACCEL2 article
#
# The model P3D, divided by the linear matter power at the bin centres, is
# evaluated at `N_DRAWS` points drawn at random from the post-burn-in chain. The
# band is the 16-84 percentile range over those draws and the solid line their
# median. Percentiles stay meaningful if the draws are skewed, which can happen
# at high k where k_p and the velocity terms are poorly constrained. The band shows how far
# the fit can move under the pseudo-likelihood, so its width scales with
# `PSEUDO_LIKE_NORM`. It is not an error on the simulation. Also shown are the
# data (points), the Nelder-Mead best fit (dotted) and the ACCEL2 article fit
# (dashed). The article values are converted to this parametrization
# (bias_eta = beta bias / f, kvav = k_v^a_v) without clipping to the bounds. All
# models use the same hybrid bin averaging as the fit. The lower panel shows the
# same curves relative to the data.

# %%
from matplotlib.patches import Patch

N_DRAWS = 200

rng_draws = np.random.default_rng(2)
draws = samples[rng_draws.choice(len(samples), size=min(N_DRAWS, len(samples)), replace=False)]
p3d_draws = np.array([fitter.predict(x)[0] for x in draws])  # (n_draws, n_k, n_mu)
p3d_lo, p3d_med, p3d_hi = np.percentile(p3d_draws, [16, 50, 84], axis=0)

p3d_best = fitter.predict(x_best)[0]
x_paper = fitter.params_from_dict(
    {
        "bias": sim_fit["bias_delta"],
        "bias_eta": sim_fit["beta"] * sim_fit["bias_delta"] / f_growth,
        "q1": sim_fit["q1"],
        "q2": sim_fit["q2"],
        "kvav": sim_fit["k_v"] ** sim_fit["a_v"],
        "av": sim_fit["a_v"],
        "bv": sim_fit["b_v"],
        "kp": sim_fit["k_p"],
    }
)
p3d_paper = fitter.predict(x_paper)[0]

p3d_data = fitter.data.p3d
fig, axes = plt.subplots(
    2, 1, figsize=(9, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
)
colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, p3d_data.shape[1]))
for index in range(p3d_data.shape[1]):
    k_Mpc = fitter.data.k3d[:, index]
    valid = np.isfinite(p3d_data[:, index]) & np.isfinite(p3d_med[:, index])
    if not np.any(valid):
        continue
    k_v = k_Mpc[valid]
    plin = fitter.data.power_model.linear.get_linP_Mpc(fitter.data.linear, fitter.data.z, k_v)
    color = colors[index]
    data_v = p3d_data[valid, index]
    lo_v, med_v, hi_v = p3d_lo[valid, index], p3d_med[valid, index], p3d_hi[valid, index]
    mu_label = np.mean(fitter.data.mu3d[valid, index])

    axes[0].plot(k_v, data_v / plin, ".", color=color, ms=4, label=rf"$\mu\simeq {mu_label:.2f}$")
    axes[0].fill_between(k_v, lo_v / plin, hi_v / plin, color=color, alpha=0.35, lw=0)
    axes[0].plot(k_v, med_v / plin, "-", color=color, lw=1.0)
    axes[0].plot(k_v, p3d_best[valid, index] / plin, ":", color=color, lw=1.2)
    axes[0].plot(k_v, p3d_paper[valid, index] / plin, "--", color=color, lw=1.2)

    axes[1].fill_between(k_v, lo_v / data_v - 1, hi_v / data_v - 1, color=color, alpha=0.35, lw=0)
    axes[1].plot(k_v, med_v / data_v - 1, "-", color=color, lw=1.0)
    axes[1].plot(k_v, p3d_best[valid, index] / data_v - 1, ":", color=color, lw=1.2)
    axes[1].plot(k_v, p3d_paper[valid, index] / data_v - 1, "--", color=color, lw=1.2)

style_handles = [
    plt.Line2D([], [], color="k", marker=".", ls="", label="ACCEL2 data"),
    Patch(color="k", alpha=0.35, label=f"MCMC, 16-84% ({len(draws)} draws)"),
    plt.Line2D([], [], color="k", ls="-", label="MCMC, median"),
    plt.Line2D([], [], color="k", ls=":", label="Nelder-Mead best fit"),
    plt.Line2D([], [], color="k", ls="--", label="ACCEL2 article fit"),
]
leg_mu = axes[0].legend(ncol=2, fontsize=9, loc="lower left")
axes[0].add_artist(leg_mu)
axes[0].legend(handles=style_handles, fontsize=9, loc="upper right")
axes[0].set(xscale="log", yscale="log", ylabel=r"$P_{3D}/P_{\mathrm{lin}}$", title=f"ACCEL2 {SIM_LABEL}, z={fitter.data.z:.1f}")
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set(xscale="log", xlabel=r"$k$ [Mpc$^{-1}$]", ylabel=r"model / data $-$ 1", ylim=(-0.2, 0.2))
fig.tight_layout()

# %%
