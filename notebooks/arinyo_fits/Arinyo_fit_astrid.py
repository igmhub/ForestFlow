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
# # Fit the Arinyo model to an Astrid snapshot
#
# This notebook fits the eight Arinyo parameters jointly to Astrid P3D and
# P1D measurements. Astrid is an external HDF5 measurement rather than a
# ForestFlow archive entry, so it uses `ArinyoFitter.prepare_measurements`.
# That public API evaluates P3D on the supplied `(k, mu)` grid directly; no
# hidden rebinning or notebook-local likelihood is involved.
#
# All wavenumbers below are in Mpc$^{-1}$, P3D is in Mpc$^3$, and P1D is in
# Mpc. The fitter expects *fractional* uncertainties. P3D now combines the
# previous 5% effective fitting floor with Gaussian finite-volume variance;
# P1D keeps its previous 2% weighting. This changes the Astrid fit, not MPG.
# The objective is still the sum of mean squared normalized P3D and P1D
# residuals, not a statistical chi-squared or a full joint covariance fit.
#
# Unlike `Arinyo_fit_mpg.py`, this is **not a hybrid bin-averaged fit**.
# The external HDF5 file has mode-weighted centres but no bin edges or lattice
# coordinates. Do not assign it MP-Gadget's binning. A hybrid fit requires
# the actual Astrid estimator geometry before it can be enabled safely.

# %%
# %load_ext autoreload
# %autoreload 2

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from forestflow.model_fits import ArinyoFitter, gaussian_p3d_relative_error

# %% [markdown]
# ## Load one Astrid HDF5 snapshot
#
# Set this path to the local Astrid measurement file. The file is not packaged
# with ForestFlow because it is external simulation data.

# %%
astrid_file = Path("/home/jchaves/Proyectos/projects/lya/P3d+P1d_lya_ASTRID_new.hdf5")
if not astrid_file.is_file():
    raise FileNotFoundError(f"Set `astrid_file` to the Astrid HDF5 file; not found: {astrid_file}")


def load_astrid_snapshot(filename: Path) -> dict:
    """Read the power spectra and cosmology required by the direct fitter."""
    data = {"cosmo_params": {}}
    cosmo_labels = ("ombh2", "omch2", "ns", "As", "H0")
    with h5py.File(filename, "r") as handle:
        def load_dataset(name, object_):
            if isinstance(object_, h5py.Dataset):
                data[name] = object_[()]

        handle.visititems(load_dataset)
        data["z"] = float(handle.attrs["z"])
        data["Box_size_Mpch"] = float(handle.attrs["Box_size_Mpch"])
        for name in cosmo_labels:
            data["cosmo_params"][name] = (
                float(handle.attrs["hubble"]) * 100.0
                if name == "H0"
                else float(handle.attrs[name])
            )

    data["cosmo_params"].update({"w": -1.0, "mnu": 0.0, "nrun": 0.0, "omk": 0.0})
    return data


data = load_astrid_snapshot(astrid_file)
print(f"Loaded Astrid snapshot at z={data['z']:.3f}")

# %% [markdown]
# ## Gaussian sample variance for P3D
#
# For a real Gaussian field, Var(P)/P² = 2/N_full = 1/N_independent,
# where N_full counts both +k and -k. Errors therefore decrease as N^(-1/2),
# rather than increasing with the number of modes. The Gaussian expression
# is an approximation for nonlinear flux power; it neglects mode coupling
# and off-diagonal covariance. See [the Gaussian covariance expression](https://academic.oup.com/mnras/article/466/1/780/2687799).
#
# The current Astrid file has no explicit mode-count dataset or documented
# error convention. Its `p3d_lya_std_Mpc/P3D` values match 1/sqrt(integer N).
# We explicitly assume that these encode P/sqrt(N_full), consistent with the
# sqrt(2) factor in this notebook's former measurement plot. Integer counts
# alone cannot establish the full-versus-independent convention: confirm it
# with the measurement producer. If the stored errors are already Gaussian
# standard deviations, set `STORED_STD_CONVENTION = "gaussian_sigma"` to avoid
# adding sqrt(2) twice. No extra factor for three axes or a paired simulation
# is applied. The box is 250 Mpc/h = 250/h Mpc, not 250 Mpc.

# %%
STORED_STD_CONVENTION = "power_over_sqrt_full_count"
P3D_FRACTIONAL_FLOOR = 0.05  # Set to zero for Gaussian-only weights.
power_all = np.asarray(data["p3d_lya_Mpc"], dtype=float)
stored_std = np.asarray(data["p3d_lya_std_Mpc"], dtype=float)
valid_p3d = np.isfinite(power_all) & (power_all > 0)
if np.any(valid_p3d & (~np.isfinite(stored_std) | (stored_std <= 0))):
    raise ValueError("Populated Astrid P3D bins require positive finite stored errors.")
mode_counts_full = np.full_like(power_all, np.nan)
if STORED_STD_CONVENTION == "power_over_sqrt_full_count":
    mode_counts_full[valid_p3d] = (power_all[valid_p3d] / stored_std[valid_p3d])**2
    if not np.allclose(mode_counts_full[valid_p3d], np.rint(mode_counts_full[valid_p3d])):
        raise ValueError("Stored errors are inconsistent with unweighted integer mode counts.")
elif STORED_STD_CONVENTION == "gaussian_sigma":
    mode_counts_full[valid_p3d] = 2 * (power_all[valid_p3d] / stored_std[valid_p3d])**2
else:
    raise ValueError("Unknown stored P3D error convention")
fractional_p3d = np.full_like(power_all, np.nan)
fractional_p3d[valid_p3d] = gaussian_p3d_relative_error(
    mode_counts_full[valid_p3d], fractional_floor=P3D_FRACTIONAL_FLOOR,
    count_convention="full",
)
gaussian_std = np.full_like(power_all, np.nan)
gaussian_std[valid_p3d] = power_all[valid_p3d] * gaussian_p3d_relative_error(
    mode_counts_full[valid_p3d], count_convention="full",
)
print("Astrid box side [Mpc]:", data["Box_size_Mpch"] / (data["cosmo_params"]["H0"] / 100))

# %% [markdown]
# ## Inspect the measurements and select the fitted scales
#
# The selection is applied before constructing the fitter, making the exact
# data contract visible. P3D uses the native Astrid two-dimensional `(k, mu)`
# grid and P1D uses the line-of-sight grid.

# %%
fig, ax = plt.subplots(figsize=(7, 4))
for index in range(data["p3d_lya_Mpc"].shape[1]):
    k_Mpc = data["k_Mpc"][:, index]
    mask = k_Mpc > 0.01
    label = f"$\mu$={np.nanmean(data['mu'][:, index]):.2f}"
    ax.errorbar(
        k_Mpc[mask],
        k_Mpc[mask] ** 2 * data["p3d_lya_Mpc"][mask, index],
        yerr=k_Mpc[mask] ** 2 * gaussian_std[mask, index],
        label=label,
    )
ax.set(xscale="log", yscale="log", xlabel=r"$k$ [Mpc$^{-1}$]", ylabel=r"$k^2 P_{3D}$", ylim=(3e-1, 1e1))
ax.legend(ncol=2, fontsize=8)

# %%
kmin_3d_Mpc, kmax_3d_Mpc = 0.01, 5.0
kmin_1d_Mpc, kmax_1d_Mpc = 0.1, 3.0

k3d_Mpc = np.asarray(data["k_Mpc"])
mu3d = np.asarray(data["mu"])
p3d_Mpc = np.asarray(data["p3d_lya_Mpc"])
mask_3d = (k3d_Mpc[:, 0] > kmin_3d_Mpc) & (k3d_Mpc[:, 0] <= kmax_3d_Mpc)

k1d_all_Mpc = np.asarray(data["klos_Mpc"])
mask_1d = (k1d_all_Mpc > kmin_1d_Mpc) & (k1d_all_Mpc <= kmax_1d_Mpc)
k1d_Mpc = k1d_all_Mpc[mask_1d]
p1d_Mpc = np.real(np.asarray(data["p1d_lya_Mpc"])[mask_1d])

# %% [markdown]
# ## Prepare and run the supported fit
#
# `prepare_measurements` stores the cosmology, linear-theory grid, selected
# data, and relative errors. `fit` returns SciPy's `OptimizeResult`; the
# package also retains `best_params` and `best_chi2` for plotting or saving.

# %%
initial_parameters = {
    "bias": -0.21,
    "bias_eta": -0.35,
    "q1": 0.28,
    "q2": 0.50,
    "kvav": 0.56,
    "av": 0.0,
    "bv": 1.6,
    "kp": 6.6,
}
bounds = [
    (-0.4, -0.1), (-0.7, -0.17), (0.1, 1.2), (-0.8, 0.8),
    (0.2, 0.8), (-0.5, 1.0), (1.0, 2.4), (4.0, 20.0),
]

fitter = ArinyoFitter(bounds=bounds)
fitter.prepare_measurements(
    z=data["z"],
    cosmo_params=data["cosmo_params"],
    k3d_Mpc=k3d_Mpc[mask_3d],
    mu3d=mu3d[mask_3d],
    p3d_Mpc=p3d_Mpc[mask_3d],
    std_p3d=fractional_p3d[mask_3d],
    k1d_Mpc=k1d_Mpc,
    p1d_Mpc=p1d_Mpc,
    std_p1d=np.full_like(p1d_Mpc, 0.02),
    ini_params=initial_parameters,
)
result = fitter.fit(bounds=bounds, maxiter=500)
best_parameters = fitter.params_to_dict(result.x)
print(f"success={result.success}; chi2={result.fun:.4f}")
best_parameters

# %% [markdown]
# ## Compare the fitted model and data
#
# The residual bands show the effective fractional uncertainties used in the
# objective. They are diagnostics of this fit, not inferred parameter errors.
# In the P3D fractional-residual panel, the shaded band shows the total fitting
# error (Gaussian variance plus the floor); dashed curves separately show
# the Gaussian-only diagonal error. Both appear only for the first (mu near
# zero) bin, while residual curves are shown for the other selected bins too.

# %%
p3d_fit, p1d_fit = fitter.predict(result.x)
p3d_data = fitter.data.p3d
p1d_data = fitter.data.p1d

fig, axes = plt.subplots(2, figsize=(8, 7), sharex=False)
for index in range(0, p3d_data.shape[1], 1):
    color = f"C{index // 2}"
    k_Mpc = fitter.data.k3d[:, index]
    factor = k_Mpc**3 / (2 * np.pi**2)
    axes[0].plot(k_Mpc, factor * p3d_data[:, index], color=color)
    axes[0].plot(k_Mpc, factor * p3d_fit[:, index], "--", color=color)
axes[0].set(xscale="log", yscale="log", ylabel=r"$k^3 P_{3D}/(2\pi^2)$")

axes[1].plot(fitter.data.k1d, fitter.data.k1d * p1d_data / np.pi, label="Astrid")
axes[1].plot(fitter.data.k1d, fitter.data.k1d * p1d_fit / np.pi, "--", label="Arinyo fit")
axes[1].set(xscale="log", xlabel=r"$k_\parallel$ [Mpc$^{-1}$]", ylabel=r"$k_\parallel P_{1D}/\pi$")
axes[1].legend()
fig.tight_layout()

# %% [markdown]
# ## P3D relative to the linear matter power
#
# Divide both the measured flux power and the best-fitting Arinyo prediction
# by the same linear matter spectrum from the fitted snapshot's cosmology.
# This dimensionless ratio highlights bias, angular dependence and nonlinear
# corrections. Plinear is evaluated at the supplied bin centres, consistent
# with this notebook's direct (not hybrid-averaged) model comparison.
# Points represent data and dashed curves represent the best-fitting model;
# matching colors identify mu bins. Empty measurement cells are excluded.

# %%
fig, ax = plt.subplots(figsize=(9, 6))
colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, p3d_data.shape[1]))
for index in range(0, p3d_data.shape[1],2):
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
        "-",
        color=colors[index],
    )
ax.set(
    xscale="log",
    yscale="log",
    xlabel=r"$k$ [Mpc$^{-1}$]",
    ylabel=r"$P_{3D}/P_{\mathrm{lin}}$",
    title="Astrid: data (points) and best-fitting Arinyo model (dashed)",
)
ax.legend(ncol=4, fontsize=8)
fig.tight_layout()

# %%
fig, axes = plt.subplots(2, figsize=(8, 7), sharex=False)
for index in range(0, p3d_data.shape[1], 1):
    color = f"C{index // 3}"
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
        gaussian_fractional = gaussian_std[mask_3d, index] / p3d_data[:, index]
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
axes[1].axhspan(-0.02, 0.02, color="k", alpha=0.15)
axes[1].set(
    xscale="log", xlabel=r"$k_\parallel$ [Mpc$^{-1}$]", ylabel="P1D fractional residual"
)
fig.tight_layout()

# %%
