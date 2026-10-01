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
# # Redshift evolution of Arinyo fits
#
# This notebook studies how the eight Arinyo parameters evolve across MPG
# snapshots. It first fits each redshift independently with the supported
# `ArinyoFitter`, then summarizes the resulting parameter values with simple
# polynomials in redshift.
#
# The polynomial curves are descriptive diagnostics, not a joint redshift
# likelihood or a prior. A joint fit would require an explicit model for the
# redshift dependence and its covariance; this notebook intentionally keeps
# the independently fitted parameters visible.

# %%
# %load_ext autoreload
# %autoreload 2

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from forestflow.archive.gadget_archive import GadgetArchive3D
from forestflow.model_fits import ArinyoFitter

# %% [markdown]
# ## Select one MPG simulation series
#
# Each archive entry is an averaged snapshot with the existing `Arinyo_min`
# result available as a starting point. The supported fitter applies the same
# P3D and P1D scale cuts to every redshift.

# %%
archive = GadgetArchive3D()
simulation_label = "mpg_central"
snapshots = archive.get_testing_data(simulation_label)
print(f"{simulation_label}: {len(snapshots)} snapshots from z={snapshots[-1]['z']:.2f} to z={snapshots[0]['z']:.2f}")

# %% [markdown]
# ## Fit snapshots independently
#
# Keep `run_fits=False` to inspect the archive-provided initial fits quickly.
# Set it to `True` to recompute every snapshot with the current fitter. Results
# are saved with ordinary parameter names, redshifts, chi-squared values, and
# optimizer diagnostics, so they can be reused without the notebook.

# %%
run_fits = True
results_file = Path(f"Arinyo_fit_{simulation_label}_redshift_evolution.npy")


def archive_parameters(entries, parameter_names):
    """Collect the archive-provided per-snapshot Arinyo starting parameters."""
    return {
        name: np.asarray([entry["Arinyo_min"][name] for entry in entries])
        for name in parameter_names
    }


def fit_snapshots(entries):
    """Run independent supported fits and return portable result arrays."""
    parameter_names = ArinyoFitter.PARAM_NAMES
    parameters = {name: np.full(len(entries), np.nan) for name in parameter_names}
    initial_chi2 = np.full(len(entries), np.nan)
    chi2 = np.full(len(entries), np.nan)
    success = np.zeros(len(entries), dtype=bool)
    message = np.empty(len(entries), dtype=object)

    for index, simulation in enumerate(entries):
        fitter = ArinyoFitter(
            zlist=np.array([simulation["z"]]),
            kmin_3d=0.7,
            kmax_3d=4.5,
            kmin_1d=0.3,
            kmax_1d=7.0,
        )
        fitter.prepare_simulation(simulation)
        initial_chi2[index] = fitter.chi2(
            fitter.params_from_dict(fitter.data.ini_params)
        )
        result = fitter.fit_iterative()
        for name, value in fitter.params_to_dict(fitter.best_params).items():
            parameters[name][index] = value
        chi2[index] = result.fun
        success[index] = result.success
        message[index] = result.message
        print(f"z={simulation['z']:.2f}: {initial_chi2[index]:.4f} -> {chi2[index]:.4f}")

    return {
        "schema_version": 1,
        "simulation_label": simulation_label,
        "parameter_names": parameter_names,
        "z": np.asarray([entry["z"] for entry in entries]),
        "initial_chi2": initial_chi2,
        "chi2": chi2,
        "success": success,
        "message": message,
        "Arinyo": parameters,
    }


if run_fits:
    fit_results = fit_snapshots(snapshots)
    np.save(results_file, fit_results)
    print(f"Saved recomputed fits to {results_file.resolve()}")
elif results_file.is_file():
    fit_results = np.load(results_file, allow_pickle=True).item()
    print(f"Loaded saved fits from {results_file}")
else:
    fit_results = {
        "simulation_label": simulation_label,
        "parameter_names": ArinyoFitter.PARAM_NAMES,
        "z": np.asarray([entry["z"] for entry in snapshots]),
        "Arinyo": archive_parameters(snapshots, ArinyoFitter.PARAM_NAMES),
    }
    print("Using archive-provided Arinyo_min values; set run_fits=True to recompute them.")

# %% [markdown]
# ## Plot independent fits and descriptive redshift trends
#
# A quadratic is drawn only to make broad trends easier to see. It should not
# be interpreted as an interpolator outside the fitted redshift range.

# %%
z = np.asarray(fit_results["z"])
order = np.argsort(z)
z_plot = np.linspace(z.min(), z.max(), 200)

fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
for axis, name in zip(axes.flat, fit_results["parameter_names"]):
    values = np.asarray(fit_results["Arinyo"][name])
    finite = np.isfinite(values)
    axis.plot(z[finite], values[finite], "o", label="independent fits")
    if np.count_nonzero(finite) >= 3:
        coefficients = np.polyfit(z[finite], values[finite], deg=2)
        axis.plot(z_plot, np.polyval(coefficients, z_plot), "--", label="quadratic guide")
    axis.set_title(name)
    axis.grid(alpha=0.25)

axes[0, 0].set_ylabel("parameter value")
axes[1, 0].set_ylabel("parameter value")
for axis in axes[1]:
    axis.set_xlabel("redshift")
axes[0, 0].legend(fontsize=8)
fig.tight_layout()

# %%
if "chi2" in fit_results:
    fig, axis = plt.subplots()
    axis.plot(z[order], np.asarray(fit_results["chi2"])[order], "o-")
    axis.set(xlabel="redshift", ylabel="final chi2", title="Independent-fit diagnostic")
    axis.grid(alpha=0.25)

# %%
