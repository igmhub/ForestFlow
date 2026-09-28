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

from forestflow.archive import GadgetArchive3D
from forestflow.fitting import ArinyoFitter

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

archive = GadgetArchive3D(postproc=postproc)

# %%
# `get_training_data` uses ForestFlow's standard emulator inputs by default.
# Choose any MPG hypercube label to fit the corrected training measurement.
simulation_label = "mpg_0"
snapshots = archive.get_training_data(simulation_label)

snapshot_index = 0
simulation = snapshots[snapshot_index]
print(
    f"{simulation_label} snapshot {snapshot_index}: z={simulation['z']:.3f} "
    f"(postproc={postproc})"
)

# %% [markdown]
# ## Fit and inspect one snapshot
#
# `fit_iterative` begins with a bounded L-BFGS-B fit and can follow it with
# Nelder--Mead refinements. The returned SciPy result, `best_params`, and
# `best_chi2` are all retained by the fitter.

# %%
fitter = ArinyoFitter(
    zlist=np.array([simulation["z"]]),
    kmin_3d=0.7,
    kmax_3d=4.5,
    kmin_1d=0.3,
    kmax_1d=7.0,
)
fitter.prepare_simulation(simulation)
initial_parameters = fitter.params_from_dict(fitter.data.ini_params)
initial_chi2 = fitter.chi2(initial_parameters)
result = fitter.fit_iterative()

best_parameters = fitter.params_to_dict(fitter.best_params)
print(f"chi2: {initial_chi2:.4f} -> {fitter.best_chi2:.4f}")
best_parameters

# %%
figures = fitter.plot_fit()
residual_figures = fitter.plot_residuals()

# %%
{'bias': np.float64(-0.7349681477587546),
 'bias_eta': np.float64(-0.24569888228156794),
 'q1': np.float64(1.0006817430919788),
 'q2': np.float64(0.2840178663021624),
 'kvav': np.float64(1.5787200310459863),
 'av': np.float64(0.7253009499550407),
 'bv': np.float64(1.8001766153488235),
 'kp': np.float64(28.333600445616142)}

# %% [markdown]
# ## Optionally fit an MPG collection and save its results
#
# Leave `run_all_fits=False` for the tutorial. Enabling it fits every selected
# snapshot, periodically saves progress, and writes parameter names, snapshot
# metadata, initial/final chi2, convergence information, and best-fit Arinyo
# parameters. The output uses ordinary parameter names, never plotting labels.

# %%
run_all_fits = False
output_file = Path(f"Arinyo_fit_{simulation_label}.npy")

if run_all_fits:
    result_fits = {
        name: np.full(len(snapshots), np.nan) for name in fitter.PARAM_NAMES
    }
    initial_chi2_all = np.full(len(snapshots), np.nan)
    final_chi2_all = np.full(len(snapshots), np.nan)
    success = np.zeros(len(snapshots), dtype=bool)
    messages = np.empty(len(snapshots), dtype=object)

    def save_results():
        np.save(
            output_file,
            {
                "schema_version": 1,
                "simulation_label": simulation_label,
                "postproc": postproc,
                "parameter_names": fitter.PARAM_NAMES,
                "z": np.asarray([item["z"] for item in snapshots]),
                "ind_snap": np.asarray([item.get("ind_snap") for item in snapshots]),
                "val_scaling": np.asarray([item.get("val_scaling") for item in snapshots]),
                "initial_chi2": initial_chi2_all,
                "chi2": final_chi2_all,
                "success": success,
                "message": messages,
                "Arinyo": result_fits,
            },
        )

    for index, simulation in enumerate(snapshots):
        if index % 50 == 0:
            save_results()
        fitter.prepare_simulation(simulation)
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

    save_results()
    print(f"Saved {len(snapshots)} fits to {output_file.resolve()}")

# %%
