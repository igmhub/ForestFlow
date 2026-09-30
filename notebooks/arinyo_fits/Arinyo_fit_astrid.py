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
# Mpc. The fitter expects *fractional* uncertainties. Here we deliberately
# use 5% for P3D and 2% for P1D, matching the former Astrid workflow. Replace
# them with measurement-derived fractional errors when those are available.

# %%
# %load_ext autoreload
# %autoreload 2

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from forestflow.model_fits import ArinyoFitter

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
# ## Inspect the measurements and select the fitted scales
#
# The selection is applied before constructing the fitter, making the exact
# data contract visible. P3D uses the native Astrid two-dimensional `(k, mu)`
# grid and P1D uses the line-of-sight grid.

# %%
fig, ax = plt.subplots(figsize=(7, 4))
for index in range(data["p3d_lya_Mpc"].shape[1]):
    k_Mpc = data["k_Mpc"][:, index]
    mask = k_Mpc > 0.1
    label = f"$\mu$={np.nanmean(data['mu'][:, index]):.2f}"
    ax.errorbar(
        k_Mpc[mask],
        k_Mpc[mask] ** 2 * data["p3d_lya_Mpc"][mask, index],
        yerr=k_Mpc[mask] ** 2 * data["p3d_lya_std_Mpc"][mask, index] * np.sqrt(2),
        label=label,
    )
ax.set(xscale="log", yscale="log", xlabel=r"$k$ [Mpc$^{-1}$]", ylabel=r"$k^2 P_{3D}$")
ax.legend(ncol=2, fontsize=8)

# %%
kmin_3d_Mpc, kmax_3d_Mpc = 0.1, 5.0
kmin_1d_Mpc, kmax_1d_Mpc = 0.1, 5.0

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
    std_p3d=np.full_like(p3d_Mpc[mask_3d], 0.05),
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

# %%
p3d_fit, p1d_fit = fitter.predict(result.x)
p3d_data = fitter.data.p3d
p1d_data = fitter.data.p1d

fig, axes = plt.subplots(2, figsize=(8, 7), sharex=False)
for index in range(0, p3d_data.shape[1], 2):
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

# %%
fig, axes = plt.subplots(2, figsize=(8, 7), sharex=False)
for index in range(0, p3d_data.shape[1], 3):
    color = f"C{index // 3}"
    axes[0].plot(fitter.data.k3d[:, index], p3d_fit[:, index] / p3d_data[:, index] - 1.0, color=color)
axes[0].axhspan(-0.05, 0.05, color="k", alpha=0.15)
axes[0].set(xscale="log", ylabel="P3D fractional residual")

axes[1].plot(fitter.data.k1d, p1d_fit / p1d_data - 1.0)
axes[1].axhspan(-0.02, 0.02, color="k", alpha=0.15)
axes[1].set(xscale="log", xlabel=r"$k_\parallel$ [Mpc$^{-1}$]", ylabel="P1D fractional residual")
fig.tight_layout()
