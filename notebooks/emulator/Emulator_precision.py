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
# # Precision of the corrected ForestFlow emulator
#
# This notebook compares the ``forest_mpg_fix`` emulator to the independently
# fitted corrected-postprocessing Arinyo parameters, ``arinyo_fixp3d``. The
# reference is therefore the best analytic Arinyo description of each
# measurement, rather than the measurement itself. No fitting is performed.
#
# P3D and P1D ratios are emulator prediction divided by the projection of the
# saved Arinyo fit, minus one. P3D uses the same hybrid finite-volume averaging
# as the fits.

# %%
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
import numpy as np

from forestflow.archive.gadget_archive import GadgetArchive3D
from forestflow.emulator.p3d_cinn import P3DEmulator
from forestflow.model_fits import ArinyoFitter
from forestflow.statistics.rebin_p3d import rebin_P3D_Mpc_mode_weighted

# %% [markdown]
# ## Archive, reference fits, and corrected emulator
#
# ``forest_mpg_fix`` must be regenerated from the corrected fits. The archive
# attaches those same fits under ``arinyo_fixp3d`` when their files are present
# in ``data/best_arinyo/cabayol23_fixp3d``.

# %%
n_realizations = 1000
emulator = P3DEmulator(key="forest_mpg_fix", Nrealizations=n_realizations)
archive = GadgetArchive3D(postproc="Cabayol23_fixp3d")
comparison = {
    "central": archive.get_testing_data("mpg_central"),
    "seed": archive.get_testing_data("mpg_seed"),
    "central + seed": archive.get_central_seed_average(),
}
if not all(
    "arinyo_fixp3d" in snapshot
    for snapshots in comparison.values()
    for snapshot in snapshots
):
    raise FileNotFoundError("Corrected central, seed, and combined fit files are required.")

# %%
def emulator_input(snapshot):
    """Build one emulator input mapping in the bundle's documented order."""
    return {name: float(snapshot[name]) for name in emulator.input_labels}


def emulate_snapshots(snapshots, chunk_size=128):
    """Predict coefficients in cINN batches with common latent samples."""
    outputs = {name: [] for name in emulator.output_labels}
    inputs = [emulator_input(snapshot) for snapshot in snapshots]
    for start in range(0, len(inputs), chunk_size):
        chunk = inputs[start : start + chunk_size]
        prediction = emulator.evaluate(
            chunk,
            Nrealizations=n_realizations,
            seed=0,
            latent_indices=np.zeros(len(chunk), dtype=int),
        )
        for name in emulator.output_labels:
            outputs[name].extend(np.asarray(prediction[name]).reshape(-1))
    return [
        {name: float(outputs[name][index]) for name in emulator.output_labels}
        for index in range(len(snapshots))
    ]


# %% [markdown]
# ## Arinyo coefficients: central emulator versus fitted references
#
# The three testing inputs are intentionally very similar. To avoid three
# nearly coincident emulator curves, this panel evaluates the emulator only
# for ``mpg_central`` while retaining all three independently fitted reference
# curves.

# %%
central_snapshots = comparison["central"]
central_emulated = emulate_snapshots(central_snapshots)
central_redshift = np.asarray([snapshot["z"] for snapshot in central_snapshots])
central_order = np.argsort(central_redshift)

fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
for axis, name in zip(axes.flat, ArinyoFitter.PARAM_NAMES):
    for label, snapshots in comparison.items():
        redshift = np.asarray([snapshot["z"] for snapshot in snapshots])
        fitted = np.asarray([snapshot["arinyo_fixp3d"][name] for snapshot in snapshots])
        order = np.argsort(redshift)
        axis.plot(redshift[order], fitted[order], "o-", ms=3, label=f"{label}: fit")
    predicted = np.asarray([parameters[name] for parameters in central_emulated])
    axis.plot(
        central_redshift[central_order],
        predicted[central_order],
        "--",
        lw=1.6,
        color="C0",
        label="central: emulator",
    )
    axis.set_title(name)
    axis.grid(alpha=0.25)
axes[0, 0].set_ylabel("Arinyo parameter")
axes[1, 0].set_ylabel("Arinyo parameter")
for axis in axes[1]:
    axis.set_xlabel("redshift")
axes[0, 0].legend(fontsize=7)
fig.tight_layout()

# %% [markdown]
# ## One redshift: P3D and P1D emulator error
#
# Four broad mu bins make the P3D comparison readable. Choose any available
# archive redshift below.

# %%
fit_settings = dict(kmin_3d=0.01, kmax_3d=4.5, kmin_1d=0.01, kmax_1d=6.0)
redshift_to_plot = 3.0
n_mu_bins = 4


def snapshot_at_redshift(snapshots, redshift):
    return min(snapshots, key=lambda snapshot: abs(snapshot["z"] - redshift))


def projected_ratio(snapshot, emulated_parameters, fitter):
    """Return emulator/Arinyo-fit P3D and P1D ratios on fit geometry."""
    fitter.prepare_simulation(snapshot, is_mpg=True)
    fitted_p3d, fitted_p1d = fitter.predict(
        fitter.params_from_dict(snapshot["arinyo_fixp3d"])
    )
    emulator_p3d, emulator_p1d = fitter.predict(
        fitter.params_from_dict(emulated_parameters)
    )
    k_iMpc, mu, fitted_p3d, mu_edges = rebin_P3D_Mpc_mode_weighted(
        fitter.data.k3d, fitter.data.mu3d, fitted_p3d, fitter._mpg_k_mu_modes,
        n_mu_bins=n_mu_bins,
    )
    _, _, emulator_p3d, _ = rebin_P3D_Mpc_mode_weighted(
        fitter.data.k3d, fitter.data.mu3d, emulator_p3d, fitter._mpg_k_mu_modes,
        n_mu_bins=n_mu_bins,
    )
    return {
        "k3d_iMpc": k_iMpc,
        "mu_edges": mu_edges,
        "P3D_ratio": emulator_p3d / fitted_p3d - 1.0,
        "k1d_iMpc": fitter.data.k1d.copy(),
        "P1D_ratio": emulator_p1d / fitted_p1d - 1.0,
    }


fitter = ArinyoFitter(**fit_settings)
comparison_ratio = {}
for label, snapshots in comparison.items():
    snapshot = snapshot_at_redshift(snapshots, redshift_to_plot)
    emulated = emulate_snapshots([snapshot])[0]
    comparison_ratio[label] = projected_ratio(snapshot, emulated, fitter)
actual_redshift = snapshot_at_redshift(comparison["central"], redshift_to_plot)["z"]

# %%
fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex="row")
for column, (label, ratio) in enumerate(comparison_ratio.items()):
    for mu_index in range(n_mu_bins):
        valid = np.isfinite(ratio["k3d_iMpc"][:, mu_index])
        mu_label = (
            rf"${ratio['mu_edges'][mu_index]:.2f} \leq \mu "
            + (rf"\leq {ratio['mu_edges'][mu_index + 1]:.2f}$"
               if mu_index == n_mu_bins - 1 else rf"< {ratio['mu_edges'][mu_index + 1]:.2f}$")
        )
        axes[0, column].plot(ratio["k3d_iMpc"][valid, mu_index], ratio["P3D_ratio"][valid, mu_index], label=mu_label)
    axes[1, column].plot(ratio["k1d_iMpc"], ratio["P1D_ratio"], color="C4")
    axes[0, column].set_title(label)
    axes[0, column].legend(fontsize=8)
for axis, scale in zip(axes[0], [fit_settings["kmax_3d"]] * 3):
    axis.axvline(scale, color="k", ls="--")
for axis, scale in zip(axes[1], [fit_settings["kmax_1d"]] * 3):
    axis.axvline(scale, color="k", ls="--")
for axis in axes[0]:
    axis.axhspan(-0.1, 0.1, color="0.8", alpha=0.35, zorder=0)
for axis in axes[1]:
    axis.axhspan(-0.01, 0.01, color="0.8", alpha=0.35, zorder=0)
for axis in axes.flat:
    axis.axhline(0.0, color="k", ls=":")
    axis.set_xscale("log")
    axis.grid(alpha=0.25)
axes[0, 0].set_ylabel(r"$P_\mathrm{3D}^\mathrm{emu}/P_\mathrm{3D}^\mathrm{fit}-1$")
axes[1, 0].set_ylabel(r"$P_\mathrm{1D}^\mathrm{emu}/P_\mathrm{1D}^\mathrm{fit}-1$")
for axis in axes[0]: axis.set_xlabel(r"$k\,[\mathrm{Mpc}^{-1}]$")
for axis in axes[1]: axis.set_xlabel(r"$k_\parallel\,[\mathrm{Mpc}^{-1}]$")
fig.suptitle(f"Corrected emulator precision at z={actual_redshift:.2f}", y=1.02)
fig.tight_layout()

# %% [markdown]
# ## Mean and standard deviation across all corrected hypercube fits
#
# The cINN evaluation is batched before the P3D/P1D projections. The line is
# the mean emulator/fit residual over all training snapshots; shading is one
# standard deviation across snapshots, not an uncertainty on the mean.

# %%
training = archive.training_data
assert len({snapshot["sim_label"] for snapshot in training}) == 30
assert all("arinyo_fixp3d" in snapshot for snapshot in training)
training_emulated = emulate_snapshots(training)
training_redshift = np.asarray([snapshot["z"] for snapshot in training])
training_fitter = ArinyoFitter(**fit_settings)
p3d_ratios, p1d_ratios = [], []
k3d_iMpc = mu_edges = k1d_iMpc = None
for index, (snapshot, emulated) in enumerate(zip(training, training_emulated, strict=True)):
    if index % 100 == 0:
        print(f"{index}/{len(training)}")
    ratio = projected_ratio(snapshot, emulated, training_fitter)
    if k3d_iMpc is None:
        k3d_iMpc, mu_edges, k1d_iMpc = ratio["k3d_iMpc"], ratio["mu_edges"], ratio["k1d_iMpc"]
    else:
        if ratio["P3D_ratio"].shape != p3d_ratios[0].shape or ratio["P1D_ratio"].shape != p1d_ratios[0].shape:
            raise ValueError("Training snapshots have incompatible prediction shapes")
    p3d_ratios.append(ratio["P3D_ratio"])
    p1d_ratios.append(ratio["P1D_ratio"])
p3d_mean, p3d_std = np.nanmean(p3d_ratios, axis=0), np.nanstd(p3d_ratios, axis=0)
p1d_mean, p1d_std = np.nanmean(p1d_ratios, axis=0), np.nanstd(p1d_ratios, axis=0)

# %%
fig, axes = plt.subplots(2, figsize=(8, 6))
for mu_index in range(n_mu_bins):
    valid = np.isfinite(k3d_iMpc[:, mu_index])
    label = rf"${mu_edges[mu_index]:.2f} \leq \mu " + (rf"\leq {mu_edges[mu_index + 1]:.2f}$" if mu_index == n_mu_bins - 1 else rf"< {mu_edges[mu_index + 1]:.2f}$")
    axes[0].plot(k3d_iMpc[valid, mu_index], p3d_mean[valid, mu_index], lw=2, label=label)
    axes[0].fill_between(k3d_iMpc[valid, mu_index], (p3d_mean-p3d_std)[valid, mu_index], (p3d_mean+p3d_std)[valid, mu_index], alpha=0.2)
axes[1].plot(k1d_iMpc, p1d_mean, color="C4", lw=2)
axes[1].fill_between(k1d_iMpc, p1d_mean-p1d_std, p1d_mean+p1d_std, color="C4", alpha=0.2)
axes[0].axhspan(-0.1, 0.1, color="0.8", alpha=0.35, zorder=0)
axes[1].axhspan(-0.01, 0.01, color="0.8", alpha=0.35, zorder=0)
for axis, scale in zip(axes, (fit_settings["kmax_3d"], fit_settings["kmax_1d"])):
    axis.axhline(0.0, color="k", ls=":")
    axis.axvline(scale, color="k", ls="--")
    axis.set_xscale("log")
    axis.grid(alpha=0.25)
axes[0].set(ylabel=r"mean $P_\mathrm{3D}^\mathrm{emu}/P_\mathrm{3D}^\mathrm{fit}-1$", xlabel=r"$k\,[\mathrm{Mpc}^{-1}]$")
axes[1].set(ylabel=r"mean $P_\mathrm{1D}^\mathrm{emu}/P_\mathrm{1D}^\mathrm{fit}-1$", xlabel=r"$k_\parallel\,[\mathrm{Mpc}^{-1}]$")
axes[0].legend(fontsize=9, ncol=2)
fig.tight_layout()

# %%

# %% [markdown]
# ## Precision resolved by redshift
#
# The previous figure collapses all redshifts. Here each redshift has a P3D
# panel and a P1D panel. Lines and bands are the mean and standard deviation
# over the available hypercube simulations and optical-depth rescalings at that
# one redshift.

# %%
redshifts = np.sort(np.unique(training_redshift))
n_columns = 4
n_rows = int(np.ceil(len(redshifts) / n_columns))
fig, axes = plt.subplots(
    2 * n_rows,
    n_columns,
    figsize=(15, 3.8 * n_rows),
    squeeze=False,
)

for redshift_index, redshift in enumerate(redshifts):
    row, column = divmod(redshift_index, n_columns)
    p3d_axis = axes[2 * row, column]
    p1d_axis = axes[2 * row + 1, column]
    mask = np.isclose(training_redshift, redshift)
    p3d_z_mean = np.nanmean(np.asarray(p3d_ratios)[mask], axis=0)
    p3d_z_std = np.nanstd(np.asarray(p3d_ratios)[mask], axis=0)
    p1d_z_mean = np.nanmean(np.asarray(p1d_ratios)[mask], axis=0)
    p1d_z_std = np.nanstd(np.asarray(p1d_ratios)[mask], axis=0)

    p3d_axis.axhspan(-0.1, 0.1, color="0.8", alpha=0.35, zorder=0)
    for mu_index in range(n_mu_bins):
        valid = np.isfinite(k3d_iMpc[:, mu_index])
        label = (
            rf"${mu_edges[mu_index]:.2f} \leq \mu "
            + (
                rf"\leq {mu_edges[mu_index + 1]:.2f}$"
                if mu_index == n_mu_bins - 1
                else rf"< {mu_edges[mu_index + 1]:.2f}$"
            )
        )
        p3d_axis.plot(
            k3d_iMpc[valid, mu_index], p3d_z_mean[valid, mu_index], label=label
        )
        p3d_axis.fill_between(
            k3d_iMpc[valid, mu_index],
            (p3d_z_mean - p3d_z_std)[valid, mu_index],
            (p3d_z_mean + p3d_z_std)[valid, mu_index],
            alpha=0.2,
        )
    p1d_axis.axhspan(-0.01, 0.01, color="0.8", alpha=0.35, zorder=0)
    p1d_axis.plot(k1d_iMpc, p1d_z_mean, color="C4")
    p1d_axis.fill_between(
        k1d_iMpc, p1d_z_mean - p1d_z_std, p1d_z_mean + p1d_z_std,
        color="C4", alpha=0.2,
    )
    for axis, scale in ((p3d_axis, fit_settings["kmax_3d"]), (p1d_axis, fit_settings["kmax_1d"])):
        axis.axhline(0.0, color="k", ls=":")
        axis.axvline(scale, color="k", ls="--")
        axis.set_xscale("log")
        axis.grid(alpha=0.25)
    p3d_axis.set_title(f"z = {redshift:.2f}")
    p1d_axis.set_xlabel(r"$k_\parallel\,[\mathrm{Mpc}^{-1}]$")
    if column == 0:
        p3d_axis.set_ylabel(r"$P_\mathrm{3D}^\mathrm{emu}/P_\mathrm{3D}^\mathrm{fit}-1$")
        p1d_axis.set_ylabel(r"$P_\mathrm{1D}^\mathrm{emu}/P_\mathrm{1D}^\mathrm{fit}-1$")
    if redshift_index == 0:
        p3d_axis.legend(fontsize=7, ncol=2)

for panel_index in range(len(redshifts), n_rows * n_columns):
    row, column = divmod(panel_index, n_columns)
    axes[2 * row, column].set_visible(False)
    axes[2 * row + 1, column].set_visible(False)

fig.suptitle("Corrected emulator precision resolved by redshift", y=1.002)
fig.tight_layout()

# %% [markdown]
# ## Central-simulation precision resolved by redshift
#
# This final figure retains the individual central realization rather than
# averaging over the hypercube. Each redshift again has a P3D/P1D panel pair.

# %%
central_fitter = ArinyoFitter(**fit_settings)
central_ratios = [
    projected_ratio(snapshot, emulated, central_fitter)
    for snapshot, emulated in zip(central_snapshots, central_emulated, strict=True)
]
central_redshifts = np.asarray([snapshot["z"] for snapshot in central_snapshots])
central_order = np.argsort(central_redshifts)
n_columns = 4
n_rows = int(np.ceil(len(central_order) / n_columns))
fig, axes = plt.subplots(
    2 * n_rows,
    n_columns,
    figsize=(15, 3.8 * n_rows),
    squeeze=False,
)

for panel_position, snapshot_index in enumerate(central_order):
    row, column = divmod(panel_position, n_columns)
    p3d_axis = axes[2 * row, column]
    p1d_axis = axes[2 * row + 1, column]
    ratio = central_ratios[snapshot_index]
    p3d_axis.axhspan(-0.1, 0.1, color="0.8", alpha=0.35, zorder=0)
    for mu_index in range(n_mu_bins):
        valid = np.isfinite(ratio["k3d_iMpc"][:, mu_index])
        label = (
            rf"${ratio['mu_edges'][mu_index]:.2f} \leq \mu "
            + (
                rf"\leq {ratio['mu_edges'][mu_index + 1]:.2f}$"
                if mu_index == n_mu_bins - 1
                else rf"< {ratio['mu_edges'][mu_index + 1]:.2f}$"
            )
        )
        p3d_axis.plot(
            ratio["k3d_iMpc"][valid, mu_index],
            ratio["P3D_ratio"][valid, mu_index],
            label=label,
        )
    p1d_axis.axhspan(-0.01, 0.01, color="0.8", alpha=0.35, zorder=0)
    p1d_axis.plot(ratio["k1d_iMpc"], ratio["P1D_ratio"], color="C4")
    for axis, scale in ((p3d_axis, fit_settings["kmax_3d"]), (p1d_axis, fit_settings["kmax_1d"])):
        axis.axhline(0.0, color="k", ls=":")
        axis.axvline(scale, color="k", ls="--")
        axis.set_xscale("log")
        axis.grid(alpha=0.25)
    p3d_axis.set_title(f"z = {central_redshifts[snapshot_index]:.2f}")
    p1d_axis.set_xlabel(r"$k_\parallel\,[\mathrm{Mpc}^{-1}]$")
    if column == 0:
        p3d_axis.set_ylabel(r"$P_\mathrm{3D}^\mathrm{emu}/P_\mathrm{3D}^\mathrm{fit}-1$")
        p1d_axis.set_ylabel(r"$P_\mathrm{1D}^\mathrm{emu}/P_\mathrm{1D}^\mathrm{fit}-1$")
    if panel_position == 0:
        p3d_axis.legend(fontsize=7, ncol=2)

for panel_position in range(len(central_order), n_rows * n_columns):
    row, column = divmod(panel_position, n_columns)
    axes[2 * row, column].set_visible(False)
    axes[2 * row + 1, column].set_visible(False)

fig.suptitle("Corrected emulator precision for mpg_central", y=1.002)
fig.tight_layout()


# %%
