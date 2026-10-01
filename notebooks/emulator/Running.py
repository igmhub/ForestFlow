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
# # Diagnose ForestFlow response to spectral running
#
# ForestFlow is conditioned on compressed linear-power inputs, principally
# $\Delta_p^2$ and $n_p$ at a pivot. This notebook compares `mpg_running`,
# whose cosmology has nonzero running of the scalar spectral index, with
# `mpg_central` at the same redshift. It asks whether the emulator plus the
# Arinyo model reproduces the *relative* P1D and P3D change seen in the
# simulations.
#
# This is a diagnostic, not a fit. The Arinyo projection always uses the full
# cosmology of the selected snapshot; any remaining mismatch can therefore
# indicate information lost by the compressed emulator inputs or an emulator/
# Arinyo-model limitation.

# %%
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
import numpy as np

from forestflow.emulator.p3d_cinn import P3DEmulator
from forestflow.archive.gadget_archive import GadgetArchive3D
from forestflow.model.arinyo import ArinyoModel
from lace.archive.gadget_archive import GadgetArchive
from lace.cosmo.cosmology import Cosmology

# %% [markdown]
# ## Load the archive and pretrained emulator
#
# The archive supplies simulation spectra, compressed emulator inputs, and
# independently fitted Arinyo coefficients. The pretrained emulator returns
# Arinyo coefficients; `ArinyoModel` then uses each snapshot's full cosmology
# to predict P1D and P3D on the simulation grids. The direct best-fit curves
# use the archived Arinyo fit instead, with no emulator evaluation. A fixed
# seed removes Monte-Carlo realization noise from the emulator comparison.

# %%
z = 3.0
n_realizations = 1000
archive = GadgetArchive3D(postproc="Cabayol23_fixp3d")
emulator = P3DEmulator(key="forest_mpg_fix", Nrealizations=n_realizations)


def get_snapshot(simulation_label, redshift):
    """Return exactly one archive snapshot at the requested redshift."""
    # Call LaCE's base method directly because the ForestFlow wrapper also
    # requests a low-k fit file that is not available for every simulation.
    snapshots = GadgetArchive.get_testing_data(
        archive, simulation_label, ind_rescaling=0
    )
    # Attach the regular per-snapshot Arinyo fits used for the direct curves.
    archive.add_Arinyo_minimizer_indiv(
        snapshots, simulation_label, kmax_3d=5, kmax_1d=4
    )
    matches = [item for item in snapshots if np.isclose(item["z"], redshift)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {simulation_label} snapshot at z={redshift}; found {len(matches)}"
        )
    return matches[0]


central = get_snapshot("mpg_central", z)
running = get_snapshot("mpg_running", z)
print("central nrun:", central["cosmo_params"].get("nrun"))
print("running nrun:", running["cosmo_params"].get("nrun"))

# %% [markdown]
# ## Predict P1D and P3D for one snapshot
#
# The emulator call itself only uses its documented input labels. In contrast,
# the subsequent Arinyo evaluation receives the full snapshot cosmology. This
# separation makes clear which part of the calculation can respond directly to
# spectral running.

# %%
def project_arinyo(snapshot, arinyo_parameters):
    """Project Arinyo parameters using the snapshot's full cosmology only."""
    cosmology = Cosmology(cosmo_params_dict=snapshot["cosmo_params"])
    model = ArinyoModel(cosmology)
    linear = model.linear.get_linear_theory(snapshot["z"])
    k1d_Mpc = np.asarray(snapshot["k_Mpc"])
    parameters = {
        name: arinyo_parameters[name]
        for name in model.default_params
        if name in arinyo_parameters
    }
    return {
        "p1d_Mpc": np.asarray(
            model.P1D_Mpc(linear, snapshot["z"], k1d_Mpc, parameters)
        ).squeeze(),
        "p3d_Mpc": np.asarray(
            model.P3D_Mpc_k_mu(
                linear,
                snapshot["z"],
                snapshot["k3d_Mpc"],
                snapshot["mu3d"],
                parameters,
            )
        ),
    }


def predict_snapshot(snapshot):
    """Return emulator-derived Arinyo projections on a snapshot's native grids."""
    emulator_inputs = {
        name: float(snapshot[name]) for name in emulator.input_labels
    }
    arinyo_parameters = emulator.evaluate(
        emulator_inputs,
        Nrealizations=n_realizations,
        seed=0,
    )
    return {
        "inputs": emulator_inputs,
        "arinyo": arinyo_parameters,
    } | project_arinyo(snapshot, arinyo_parameters)


central_prediction = predict_snapshot(central)
running_prediction = predict_snapshot(running)
central_best_fit = project_arinyo(central, central["Arinyo_min"])
running_best_fit = project_arinyo(running, running["Arinyo_min"])

# %% [markdown]
# ## Compare compressed inputs and inferred Arinyo parameters
#
# A ratio close to one for an emulator input means that the compression presents
# little distinction between the two cosmologies. The Arinyo ratios show how
# ForestFlow translates those inputs into its nonlinear model parameters.

# %%
for name in emulator.input_labels:
    ratio = running_prediction["inputs"][name] / central_prediction["inputs"][name]
    print(f"{name:12s} running / central = {ratio:.6f}")

for name in emulator.output_labels:
    ratio = running_prediction["arinyo"][name] / central_prediction["arinyo"][name]
    print(f"{name:12s} Arinyo running / central = {ratio:.6f}")

# %% [markdown]
# ## Absolute P1D spectra
#
# Before taking ratios, compare the dimensional spectra themselves. Solid
# curves are simulations; dashed curves are the corresponding ForestFlow plus
# Arinyo predictions; dotted curves are direct projections of the independently
# fitted Arinyo parameters. The conventional $k_\parallel P_{1D}/\pi$ scaling makes
# the P1D shape easier to inspect over the full wavenumber range.

# %%
k1d_Mpc = np.asarray(central["k_Mpc"])
positive_k1d = k1d_Mpc > 0
central_p1d = np.asarray(central["p1d_Mpc"])
running_p1d = np.asarray(running["p1d_Mpc"])

fig, axis = plt.subplots(figsize=(7, 4.5))
axis.plot(
    k1d_Mpc[positive_k1d],
    k1d_Mpc[positive_k1d] * central_p1d[positive_k1d] / np.pi,
    color="C0",
    label="central simulation",
)
axis.plot(
    k1d_Mpc[positive_k1d],
    k1d_Mpc[positive_k1d] * central_prediction["p1d_Mpc"][positive_k1d] / np.pi,
    "--",
    color="C0",
    label="central ForestFlow + Arinyo",
)
axis.plot(
    k1d_Mpc[positive_k1d],
    k1d_Mpc[positive_k1d] * central_best_fit["p1d_Mpc"][positive_k1d] / np.pi,
    ":",
    color="C0",
    label="central fitted Arinyo",
)
axis.plot(
    k1d_Mpc[positive_k1d],
    k1d_Mpc[positive_k1d] * running_p1d[positive_k1d] / np.pi,
    color="C1",
    label="running simulation",
)
axis.plot(
    k1d_Mpc[positive_k1d],
    k1d_Mpc[positive_k1d] * running_prediction["p1d_Mpc"][positive_k1d] / np.pi,
    "--",
    color="C1",
    label="running ForestFlow + Arinyo",
)
axis.plot(
    k1d_Mpc[positive_k1d],
    k1d_Mpc[positive_k1d] * running_best_fit["p1d_Mpc"][positive_k1d] / np.pi,
    ":",
    color="C1",
    label="running fitted Arinyo",
)
axis.set(xlabel=r"$k_\parallel$ [Mpc$^{-1}$]", ylabel=r"$k_\parallel P_{1D}/\pi$")
axis.legend(fontsize=8)
axis.set_xlim(0.07, 5)
fig.tight_layout()

# %% [markdown]
# ## Absolute P3D spectra in angular bins
#
# These panels show the same central/running comparison for representative
# orientations. We use $k^3P_{3D}/(2\pi^2)$ to display the three-dimensional
# power. Colour identifies the simulation; line style identifies measurement
# versus prediction.

# %%
k3d_Mpc = np.asarray(central["k3d_Mpc"])
mu3d = np.asarray(central["mu3d"])
central_p3d = np.asarray(central["p3d_Mpc"])
running_p3d = np.asarray(running["p3d_Mpc"])
mu_indices = np.linspace(0, mu3d.shape[1] - 1, 4, dtype=int)

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
for axis, index in zip(axes.flat, mu_indices, strict=True):
    positive_k3d = k3d_Mpc[:, index] > 0
    k_values = k3d_Mpc[positive_k3d, index]
    factor = k_values**3 / (2 * np.pi**2)
    axis.plot(k_values, factor * central_p3d[positive_k3d, index], color="C0", label="central simulation")
    axis.plot(k_values, factor * central_prediction["p3d_Mpc"][positive_k3d, index], "--", color="C0", label="central ForestFlow + Arinyo")
    axis.plot(k_values, factor * central_best_fit["p3d_Mpc"][positive_k3d, index], ":", color="C0", label="central fitted Arinyo")
    axis.plot(k_values, factor * running_p3d[positive_k3d, index], color="C1", label="running simulation")
    axis.plot(k_values, factor * running_prediction["p3d_Mpc"][positive_k3d, index], "--", color="C1", label="running ForestFlow + Arinyo")
    axis.plot(k_values, factor * running_best_fit["p3d_Mpc"][positive_k3d, index], ":", color="C1", label="running fitted Arinyo")
    axis.set_title(rf"$\mu \simeq {np.nanmedian(mu3d[:, index]):.2f}$")

axes[0, 0].set_ylabel(r"$k^3P_{3D}/(2\pi^2)$")
axes[1, 0].set_ylabel(r"$k^3P_{3D}/(2\pi^2)$")
axes[1, 0].set_xlabel(r"$k$ [Mpc$^{-1}$]")
axes[1, 1].set_xlabel(r"$k$ [Mpc$^{-1}$]")
axes[0, 0].legend(fontsize=7)
axes[0, 0].set_xlim(0.07, 5)
fig.tight_layout()

# %% [markdown]
# ## P1D relative response
#
# Ratios isolate the effect of spectral running. Dotted curves use the stored
# best-fitting Arinyo parameters and directly project them for each cosmology.

# %%
simulation_p1d_ratio = running_p1d / central_p1d
emulator_p1d_ratio = running_prediction["p1d_Mpc"] / central_prediction["p1d_Mpc"]
best_fit_p1d_ratio = running_best_fit["p1d_Mpc"] / central_best_fit["p1d_Mpc"]

fig, axis = plt.subplots(figsize=(7, 4.5))
axis.plot(k1d_Mpc[positive_k1d], simulation_p1d_ratio[positive_k1d], label="simulation")
axis.plot(k1d_Mpc[positive_k1d], emulator_p1d_ratio[positive_k1d], "--", label="ForestFlow + Arinyo")
axis.plot(k1d_Mpc[positive_k1d], best_fit_p1d_ratio[positive_k1d], ":", label="fitted Arinyo")
axis.axhline(1.0, color="black", linewidth=1)
axis.set(
    xscale="log",
    xlabel=r"$k_\parallel$ [Mpc$^{-1}$]",
    ylabel=r"$P_{1D}^{\rm running}/P_{1D}^{\rm central}$",
)
axis.legend()
axis.set_xlim(0.07, 5)
axis.set_ylim(0.95, 1.05)
fig.tight_layout()

# %% [markdown]
# ## P3D relative response in angular bins
#
# Four representative $\mu$ bins show whether the relative discrepancy depends
# on orientation to the line of sight.

# %%
simulation_p3d_ratio = running_p3d / central_p3d
emulator_p3d_ratio = running_prediction["p3d_Mpc"] / central_prediction["p3d_Mpc"]
best_fit_p3d_ratio = running_best_fit["p3d_Mpc"] / central_best_fit["p3d_Mpc"]

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
for axis, index in zip(axes.flat, mu_indices, strict=True):
    positive_k3d = k3d_Mpc[:, index] > 0
    axis.plot(k3d_Mpc[positive_k3d, index], simulation_p3d_ratio[positive_k3d, index], label="simulation")
    axis.plot(k3d_Mpc[positive_k3d, index], emulator_p3d_ratio[positive_k3d, index], "--", label="ForestFlow + Arinyo")
    axis.plot(k3d_Mpc[positive_k3d, index], best_fit_p3d_ratio[positive_k3d, index], ":", label="fitted Arinyo")
    axis.axhline(1.0, color="black", linewidth=1)
    axis.set_xscale("log")
    axis.set_title(rf"$\mu \simeq {np.nanmedian(mu3d[:, index]):.2f}$")

axes[0, 0].set_ylabel(r"$P_{3D}^{\rm running}/P_{3D}^{\rm central}$")
axes[1, 0].set_ylabel(r"$P_{3D}^{\rm running}/P_{3D}^{\rm central}$")
axes[1, 0].set_xlabel(r"$k$ [Mpc$^{-1}$]")
axes[1, 1].set_xlabel(r"$k$ [Mpc$^{-1}$]")
axes[0, 0].legend(fontsize=9)
axes[0, 0].set_xlim(0.07, 5)
axes[0, 0].set_ylim(0.95, 1.05)
fig.tight_layout()

# %%

# %%

# %%
