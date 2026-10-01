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
# # Fit Arinyo parameters to MPG simulations
#
# This notebook demonstrates the supported ForestFlow fitting API on MPG
# simulation snapshots. `ArinyoFitter` prepares the archive measurements,
# evaluates the Arinyo P3D and P1D model, and minimizes their joint fractional
# residual. The default errors are the documented scale-dependent effective
# errors of the fitter; they define the fitting weights, not observational
# error bars.
#
# The first workflow fits one snapshot and plots it. The optional final section
# fits a whole MPG set and writes a portable NumPy results mapping.

# %%
# %load_ext autoreload
# %autoreload 2

from pathlib import Path

import numpy as np

from forestflow.archive.gadget_archive import GadgetArchive3D
from forestflow.model_fits import ArinyoFitter

# %% [markdown]
# ## Load an MPG snapshot
#
# `GadgetArchive3D` supplies simulation measurements and the existing Arinyo
# initialization (`Arinyo_min`) used to start each fit. The fitter retains the
# input grids in Mpc units and applies its configured P3D/P1D scale cuts.
#
# Set `postproc="Cabayol23_fixp3d"` to use the corrected reshaped P3D files for
# the MPG hypercube training simulations (`mpg_0` through `mpg_29`). The test
# simulations, including `mpg_central`, intentionally keep their legacy files:
# no corrected test post-processing exists. Use `"Cabayol23"` to reproduce the
# original Cabayol et al. processing everywhere.

# %%
# postproc = "Cabayol23"
postproc = "Cabayol23_fixp3d"
archive = GadgetArchive3D(postproc=postproc, average="both")

if postproc != "Cabayol23":
    standard_archive = GadgetArchive3D(postproc="Cabayol23", average="both")

# %%
# `get_training_data` uses ForestFlow's standard emulator inputs by default.
# Choose any MPG hypercube label to fit the corrected training measurement.
training_label = "mpg_5"
# This filters the archive's already loaded ``training_data`` cache; it does
# not reread the full suite. The returned snapshots retain attached fits when
# they are available for the selected post-processing.
snapshots = archive.get_training_data(training_label)

# For corrected P3D measurements, reuse the matching standard-postprocessing
# snapshot as the initial Arinyo fit when the corrected one has none yet.
# This reads each archive once; ArinyoFitter receives the already loaded list.
standard_snapshots = None
if postproc != "Cabayol23":
    # Fits are attached to ``training_data`` during archive construction.
    # A fresh get_training_data call would return measurements without
    # ``Arinyo_min``.
    standard_snapshots = [
        snapshot
        for snapshot in standard_archive.training_data
        if snapshot["sim_label"] == training_label
    ]

snapshot_index = 0
# snapshot_index = 10
simulation = snapshots[snapshot_index]
print(
    f"{training_label} snapshot {snapshot_index}: z={simulation['z']:.3f} "
    f"(postproc={postproc})"
)

# %% [markdown]
# ## Fit and inspect one snapshot
#
# `fit_iterative` begins with a bounded L-BFGS-B fit and can follow it with
# Nelder--Mead refinements. The returned SciPy result, `best_params`, and
# `best_chi2` are all retained by the fitter.
#
# These are MP-Gadget measurements, so P3D is compared using the default
# hybrid finite-volume average: sparse cells use their exact Fourier modes and
# dense cells use the continuous phase-space average. Do not set
# `is_mpg=False` here: that fallback evaluates only at bin centres and is less
# accurate on small scales.

# %%
# Omit zlist so the fitter uses the exact MP-Gadget archive redshift grid.
# This lets the same fitter be reused safely for every snapshot below.
fitter = ArinyoFitter(
    kmin_3d=0.01,
    kmax_3d=4.5,
    kmin_1d=0.01,
    kmax_1d=6.0,
)

# %%
fitter.prepare_simulation(
    simulation,
    standard_simulations=standard_snapshots,
    is_mpg=True,
)
initial_parameters = fitter.params_from_dict(fitter.data.ini_params)
initial_chi2 = fitter.chi2(initial_parameters)

# %%
result = fitter.fit_iterative()

best_parameters = fitter.params_to_dict(fitter.best_params)
print(f"chi2: {initial_chi2:.4f} -> {fitter.best_chi2:.4f}")
best_parameters

# %%
figures = fitter.plot_fit()
residual_figures = fitter.plot_residuals()

# %% [markdown]
# ## Optionally fit an MPG collection and save its results
#
# Leave `run_all_fits=False` for the tutorial. Enabling it fits every selected
# snapshot, periodically saves progress, and writes parameter names, snapshot
# metadata, initial/final chi2, convergence information, and best-fit Arinyo
# parameters. It also saves the full archive identity fields needed to match
# a corrected measurement to its standard-postprocessing initialization. The
# output uses ordinary parameter names, never plotting labels.

# %%
run_all_fits = False
output_file = Path(f"Arinyo_fit_{training_label}.npy")

if run_all_fits:
    result_fits = {
        name: np.full(len(snapshots), np.nan) for name in fitter.PARAM_NAMES
    }
    initial_chi2_all = np.full(len(snapshots), np.nan)
    final_chi2_all = np.full(len(snapshots), np.nan)
    success = np.zeros(len(snapshots), dtype=bool)
    messages = np.empty(len(snapshots), dtype=object)


    for index, simulation in enumerate(snapshots):
        if index % 50 == 0:
            fitter.save_results(
                output_file,
                snapshots=snapshots,
                initial_chi2=initial_chi2_all,
                chi2=final_chi2_all,
                success=success,
                message=messages,
                arinyo=result_fits,
                simulation_label=training_label,
                postproc=postproc,
            )
        fitter.prepare_simulation(
            simulation,
            standard_simulations=standard_snapshots,
            is_mpg=True,
        )
        initial_chi2_all[index] = fitter.chi2(
            fitter.params_from_dict(fitter.data.ini_params)
        )
        result = fitter.fit_iterative()
        for name, value in fitter.params_to_dict(fitter.best_params).items():
            result_fits[name][index] = value
        final_chi2_all[index] = result.fun
        success[index] = result.success
        messages[index] = result.message
        print(
            f"snapshot {index:3d}: "
            f"{initial_chi2_all[index]:.4f} -> {final_chi2_all[index]:.4f}"
        )

    fitter.save_results(
        output_file,
        snapshots=snapshots,
        initial_chi2=initial_chi2_all,
        chi2=final_chi2_all,
        success=success,
        message=messages,
        arinyo=result_fits,
        simulation_label=training_label,
        postproc=postproc,
    )
    print(f"Saved {len(snapshots)} fits to {output_file.resolve()}")

# %%
