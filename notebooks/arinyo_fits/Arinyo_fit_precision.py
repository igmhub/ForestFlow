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
# # Precision of corrected Arinyo fits
#
# This notebook validates already fitted ``Cabayol23_fixp3d`` MP-Gadget
# measurements. It does not optimize any parameters. Instead, it reads the
# per-snapshot ``arinyo_fixp3d`` mappings attached by ``GadgetArchive3D`` and
# evaluates the same hybrid finite-volume P3D prediction used in the fits.
#
# It compares the central and seed testing realizations, their
# mean-flux-consistent combination, and the full corrected hypercube training
# set. The residuals below are model/data minus one.

# %%
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
import numpy as np

from forestflow.archive.gadget_archive import GadgetArchive3D
from forestflow.model_fits import ArinyoFitter
from forestflow.statistics.rebin_p3d import rebin_P3D_Mpc_mode_weighted

# %% [markdown]
# ## Load corrected fits
#
# The fit files must be in ``data/best_arinyo/cabayol23_fixp3d`` with names
# ``Arinyo_fit_<simulation_label>.npy``. In addition to the 30 hypercube files,
# this workflow expects the central, seed, and combined files:
#
# ``Arinyo_fit_mpg_central.npy``, ``Arinyo_fit_mpg_seed.npy``, and
# ``Arinyo_fit_mpg_central_seed.npy``.

# %%
archive = GadgetArchive3D(postproc="Cabayol23_fixp3d", average="both")
comparison = {
    "central": archive.get_testing_data("mpg_central"),
    "seed": archive.get_testing_data("mpg_seed"),
    "central + seed": archive.get_central_seed_average(),
}

missing = {
    label: [snapshot["z"] for snapshot in snapshots if "arinyo_fixp3d" not in snapshot]
    for label, snapshots in comparison.items()
}
missing = {label: redshifts for label, redshifts in missing.items() if redshifts}
if missing:
    raise FileNotFoundError(
        "Missing corrected Arinyo fit files for "
        f"{missing}. Run scripts/fit_p3d/fit_pflux.py with the matching label."
    )

# %% [markdown]
# ## Best-fit parameters versus redshift
#
# These are the stored, independently fitted values. They are not a joint
# redshift model or posterior constraint.

# %%
parameter_names = ArinyoFitter.PARAM_NAMES
fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
for axis, parameter in zip(axes.flat, parameter_names):
    for label, snapshots in comparison.items():
        redshift = np.asarray([snapshot["z"] for snapshot in snapshots])
        value = np.asarray(
            [snapshot["arinyo_fixp3d"][parameter] for snapshot in snapshots]
        )
        order = np.argsort(redshift)
        axis.plot(redshift[order], value[order], "o-", ms=3, alpha=0.5, label=label)
    axis.set_title(parameter)
    axis.grid(alpha=0.25)
axes[0, 0].set_ylabel("best-fit value")
axes[1, 0].set_ylabel("best-fit value")
for axis in axes[1]:
    axis.set_xlabel("redshift")
axes[0, 0].legend(fontsize=9)
fig.tight_layout()

# %% [markdown]
# ## Evaluate one snapshot for central, seed, and their combination
#
# Set ``redshift_to_plot`` to any archive redshift. P3D is rebinned from the
# native 16 mu cells into four broad bins using the discrete mode counts. This
# is a display rebinning; each P3D model prediction was first evaluated with
# the fitter's hybrid finite-volume average on the native grid.

# %%
fit_settings = dict(kmin_3d=0.01, kmax_3d=4.5, kmin_1d=0.01, kmax_1d=6.0)
redshift_to_plot = 3.0
n_mu_bins = 4


def snapshot_at_redshift(snapshots, redshift):
    """Return the stored snapshot nearest to the requested redshift."""
    return min(snapshots, key=lambda snapshot: abs(snapshot["z"] - redshift))


def evaluate_fit(snapshot, fitter):
    """Return display-rebinned P3D and native P1D model/data residuals."""
    fitter.prepare_simulation(snapshot, is_mpg=True)
    parameters = fitter.params_from_dict(snapshot["arinyo_fixp3d"])
    model_p3d, model_p1d = fitter.predict(parameters)
    k_iMpc, mu, data_p3d, mu_edges = rebin_P3D_Mpc_mode_weighted(
        fitter.data.k3d,
        fitter.data.mu3d,
        fitter.data.p3d,
        fitter._mpg_k_mu_modes,
        n_mu_bins=n_mu_bins,
    )
    _, _, fitted_p3d, _ = rebin_P3D_Mpc_mode_weighted(
        fitter.data.k3d,
        fitter.data.mu3d,
        model_p3d,
        fitter._mpg_k_mu_modes,
        n_mu_bins=n_mu_bins,
    )
    p3d_residual = fitted_p3d / data_p3d - 1.0
    p1d_residual = model_p1d / fitter.data.p1d - 1.0
    return {
        "k3d_iMpc": k_iMpc,
        "mu": mu,
        "mu_edges": mu_edges,
        "P3D_residual": p3d_residual,
        "k1d_iMpc": fitter.data.k1d.copy(),
        "P1D_residual": p1d_residual,
    }


fitter = ArinyoFitter(**fit_settings)
comparison_prediction = {
    label: evaluate_fit(snapshot_at_redshift(snapshots, redshift_to_plot), fitter)
    for label, snapshots in comparison.items()
}
actual_redshift = snapshot_at_redshift(comparison["central"], redshift_to_plot)["z"]
print(f"Plotting the archive snapshot at z={actual_redshift:.2f}")

# %%
fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex="row")
for column, (label, prediction) in enumerate(comparison_prediction.items()):
    for mu_index in range(n_mu_bins):
        valid = np.isfinite(prediction["k3d_iMpc"][:, mu_index])
        mu_label = (
            rf"${prediction['mu_edges'][mu_index]:.2f} \leq \mu "
            + (
                rf"\leq {prediction['mu_edges'][mu_index + 1]:.2f}$"
                if mu_index == n_mu_bins - 1
                else rf"< {prediction['mu_edges'][mu_index + 1]:.2f}$"
            )
        )
        axes[0, column].plot(
            prediction["k3d_iMpc"][valid, mu_index],
            prediction["P3D_residual"][valid, mu_index],
            lw=1.8,
            label=mu_label,
        )
    axes[1, column].plot(
        prediction["k1d_iMpc"], prediction["P1D_residual"], color="C4", lw=1.8
    )
    axes[0, column].set_title(label)
    axes[0, column].legend(fontsize=8)

for axis in axes.flat:
    axis.axhline(0.0, color="k", ls=":")
    axis.set_xscale("log")
    axis.grid(alpha=0.25)
for axis in axes[0]:
    axis.axvline(fit_settings["kmax_3d"], color="k", ls="--")
for axis in axes[1]:
    axis.axvline(fit_settings["kmax_1d"], color="k", ls="--")
axes[0, 0].set_ylabel(r"$P_\mathrm{3D}^\mathrm{fit}/P_\mathrm{3D}^\mathrm{data}-1$")
axes[1, 0].set_ylabel(r"$P_\mathrm{1D}^\mathrm{fit}/P_\mathrm{1D}^\mathrm{data}-1$")
for axis in axes[0]:
    axis.set_xlabel(r"$k\,[\mathrm{Mpc}^{-1}]$")
for axis in axes[1]:
    axis.set_xlabel(r"$k_\parallel\,[\mathrm{Mpc}^{-1}]$")
fig.suptitle(f"Corrected Arinyo fits at z={actual_redshift:.2f}", y=1.02)
fig.tight_layout()

# %% [markdown]
# ## Mean and standard deviation across the 30 training simulations
#
# Every available corrected hypercube snapshot is evaluated with its stored
# fit. The line is the arithmetic mean residual across all 1650 snapshots;
# the shaded region is one standard deviation, not an uncertainty on the mean.

# %%
training = archive.training_data
training_labels = {snapshot["sim_label"] for snapshot in training}
assert len(training_labels) == 30
assert all("arinyo_fixp3d" in snapshot for snapshot in training)

training_fitter = ArinyoFitter(**fit_settings)
p3d_residuals = []
p1d_residuals = []
k3d_iMpc = mu = mu_edges = k1d_iMpc = None
for index, snapshot in enumerate(training):
    if index % 100 == 0:
        print(f"{index}/{len(training)}")
    prediction = evaluate_fit(snapshot, training_fitter)
    if k3d_iMpc is None:
        k3d_iMpc = prediction["k3d_iMpc"]
        mu = prediction["mu"]
        mu_edges = prediction["mu_edges"]
        k1d_iMpc = prediction["k1d_iMpc"]
    else:
        if prediction["P3D_residual"].shape != p3d_residuals[0].shape:
            raise ValueError("Training snapshots have incompatible rebinned P3D shapes")
        if prediction["P1D_residual"].shape != p1d_residuals[0].shape:
            raise ValueError("Training snapshots have incompatible P1D shapes")
        # Reported mode-weighted centres can differ slightly among
        # realizations. Residuals are averaged by their common bin index and
        # displayed using the first snapshot's coordinates.
    p3d_residuals.append(prediction["P3D_residual"])
    p1d_residuals.append(prediction["P1D_residual"])

p3d_residuals = np.asarray(p3d_residuals)
p1d_residuals = np.asarray(p1d_residuals)
p3d_mean = np.nanmean(p3d_residuals, axis=0)
p3d_std = np.nanstd(p3d_residuals, axis=0)
p1d_mean = np.nanmean(p1d_residuals, axis=0)
p1d_std = np.nanstd(p1d_residuals, axis=0)

# %%
fig, axes = plt.subplots(2, figsize=(8, 6), sharex=False)
for mu_index in range(n_mu_bins):
    valid = np.isfinite(k3d_iMpc[:, mu_index])
    label = (
        rf"${mu_edges[mu_index]:.2f} \leq \mu "
        + (rf"\leq {mu_edges[mu_index + 1]:.2f}$" if mu_index == n_mu_bins - 1
           else rf"< {mu_edges[mu_index + 1]:.2f}$")
    )
    axes[0].plot(k3d_iMpc[valid, mu_index], p3d_mean[valid, mu_index], lw=2, label=label)
    axes[0].fill_between(
        k3d_iMpc[valid, mu_index],
        (p3d_mean - p3d_std)[valid, mu_index],
        (p3d_mean + p3d_std)[valid, mu_index],
        alpha=0.2,
    )
axes[1].plot(k1d_iMpc, p1d_mean, color="C4", lw=2)
axes[1].fill_between(k1d_iMpc, p1d_mean - p1d_std, p1d_mean + p1d_std, color="C4", alpha=0.2)

for axis, scale in zip(axes, (fit_settings["kmax_3d"], fit_settings["kmax_1d"])):
    axis.axhline(0.0, color="k", ls=":")
    axis.axvline(scale, color="k", ls="--", label="fit scale cut")
    axis.set_xscale("log")
    axis.grid(alpha=0.25)
axes[0].set(
    ylabel=r"mean residual $P_\mathrm{3D}$",
    xlabel=r"$k\,[\mathrm{Mpc}^{-1}]$",
)
axes[1].set(
    ylabel=r"mean residual $P_\mathrm{1D}$",
    xlabel=r"$k_\parallel\,[\mathrm{Mpc}^{-1}]$",
)
axes[0].legend(fontsize=9, ncol=2)
fig.tight_layout()

# %%
