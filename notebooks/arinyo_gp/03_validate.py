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
# # 3. Held-out fits, power impact and ForestFlow disagreement
#
# Primary reference: fitted Arinyo targets from held-out simulation groups.
# Secondary reference: the pretrained flow, which may have seen these groups.
# Neither coefficient agreement nor a smoke run proves simulation-spectrum accuracy.

# %%
from pathlib import Path
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import forestflow
ROOT = Path(forestflow.__file__).resolve().parent.parent
HERE = ROOT / "notebooks" / "arinyo_gp"
sys.path.insert(0, str(HERE))
import experiment as ex
from forestflow.emulator.arinyo_gp import ArinyoGPEmulator, transformed_metrics
ARTIFACTS = HERE / "artifacts"

# %%
run = Path(json.loads((ARTIFACTS / "latest.json").read_text())["run"])
bundle = ArinyoGPEmulator.load(run / "gp.pkl")
records = ex.load_bundle(run / "test_records.pkl")
fit_key = bundle.metadata["fit_key"]
assert {str(r["sim_label"]) for r in records}.isdisjoint(bundle.metadata["groups"]["train"])
rows = [{name: float(r[name]) for name in bundle.input_labels} for r in records]
reference = ex.matrix(bundle.transform, ex.columns(records, bundle.output_labels, fit_key),
                      bundle.output_labels, "output")
prediction_mapping = bundle.predict_transformed(rows)
prediction = np.column_stack([prediction_mapping[k] for k in bundle.output_labels])
report = transformed_metrics(prediction, reference)
for i, name in enumerate(bundle.output_labels):
    print(name, {metric: float(value[i]) for metric, value in report.items()})
physical = bundle.evaluate(rows)
physical_ref = np.column_stack([ex.columns(records, bundle.output_labels, fit_key)[k] for k in bundle.output_labels])
physical_array = np.column_stack([physical[k] for k in bundle.output_labels])
print("Physical absolute RMSE:", ex.metrics(physical_array, physical_ref)["rmse"])
fig, axes = plt.subplots(2, 4, figsize=(13, 6))
for i, ax in enumerate(axes.flat):
    ax.scatter([r["z"] for r in records], prediction[:, i] - reference[:, i], s=12)
    ax.axhline(0, color="k", lw=.5)
    ax.set(title=bundle.output_labels[i], xlabel="z", ylabel="Transformed residual")
plt.tight_layout()
plt.show()
plt.close(fig)

# %% [markdown]
# ## Comparison to the flow
#
# The table reports disagreement, not ground-truth error. Compare independent
# seeds and doubled counts before attributing small differences to the GP.
# The archive target regression and flow central estimator can differ persistently.

# %%
from forestflow.emulator.p3d_cinn import P3DEmulator
flow = P3DEmulator(key="forest_mpg_cinn")
n = 128 if bundle.metadata["smoke"] else 4096
flow_predictions = []
for count, seed in [(n, 1701), (2*n, 2701)]:
    result = flow.evaluate(rows, Nrealizations=count, seed=seed,
                           sampler="sobol_antithetic", statistic="mean",
                           aggregation_space="transformed", draw_policy="nested")
    flow_predictions.append(np.column_stack([result[k] for k in bundle.output_labels]))
print("GP-flow physical RMSE:", ex.metrics(physical_array, flow_predictions[-1])["rmse"])
print("Flow count/seed sensitivity (not a full convergence study):",
      ex.metrics(flow_predictions[0], flow_predictions[1])["rmse"])

# %% [markdown]
# ## P1D/P3D impact: three-way comparison against the fitted coefficients
#
# For each simulation we compute spectra with the same `ArinyoModel`, redshift
# and linear theory, changing only the Arinyo coefficients:
#
# - **fit**: the archive's `arinyo_fixp3d` coefficients (the reference),
# - **GP**: this experiment's Gaussian-process emulator,
# - **cINN**: ForestFlow's pretrained `forest_mpg_cinn` flow.
#
# CAMB is run once per *simulation* (the cosmology is shared by all of its
# snapshots), with the linear theory built on every redshift at once.
# The first few records of a smoke run only demonstrate the calculation; a
# scientific result needs all simulations, grids and errors.
# Keep Mpc units throughout. No velocity-coordinate Jacobian is applied.

# %%
from lace.cosmo import cosmology
from forestflow.model.arinyo import ArinyoModel
k = np.geomspace(0.1, 4.0, 30)
mu = np.linspace(0., 1., 7)
kg, mug = np.meshgrid(k, mu, indexing="ij")
NAMES = bundle.output_labels


def coefficients(matrix):
    """Rows of a (n, n_params) array -> list of named coefficient dicts."""
    return [dict(zip(NAMES, map(float, row))) for row in matrix]


def spectra(model, linear, z, coeff):
    p1 = model.P1D_Mpc(linear, z, k, coeff)
    p3 = model.P3D_Mpc_k_mu(linear, z, kg, mug, coeff)
    if not (np.isfinite(p1).all() and np.all(p1 > 0) and np.isfinite(p3).all() and np.all(p3 > 0)):
        raise ValueError("Nonfinite or nonpositive power; inspect this prediction")
    return p1, p3


def compare_simulation(sim_records, gp_rows, flow_rows):
    """Compare GP and cINN to the fit for all snapshots of ONE simulation.

    ``gp_rows`` and ``flow_rows`` hold the coefficients predicted for each
    record, in record order. CAMB and the linear theory are built once.
    """
    cosmo_params = sim_records[0]["cosmo_params"]
    assert all(r["cosmo_params"] == cosmo_params for r in sim_records)
    model = ArinyoModel(cosmology.Cosmology(cosmo_params_dict=cosmo_params))
    linear = model.linear.get_linear_theory(np.unique([float(r["z"]) for r in sim_records]))
    out = []
    for record, gp_coeff, flow_coeff in zip(sim_records, gp_rows, flow_rows):
        z = float(record["z"])
        fit_coeff = {name: record[fit_key][name] for name in NAMES}
        fit1, fit3 = spectra(model, linear, z, fit_coeff)
        gp1, gp3 = spectra(model, linear, z, gp_coeff)
        fl1, fl3 = spectra(model, linear, z, flow_coeff)
        out.append(dict(record=record, z=z, gp1=gp1/fit1 - 1, fl1=fl1/fit1 - 1,
                        gp3=(gp3/fit3 - 1).ravel(), fl3=(fl3/fit3 - 1).ravel(),
                        model=model, linear=linear, fit_coeff=fit_coeff,
                        gp_coeff=gp_coeff, flow_coeff=flow_coeff))
    return out


def summarise(label, results):
    for kind in ("1", "3"):
        for who, name in (("gp", "GP"), ("fl", "cINN")):
            v = np.asarray([r[who + kind] for r in results])
            print(f"{label} P{kind}D {name}/fit - 1: RMS {np.sqrt(np.mean(v**2)):.4f}, "
                  f"p99 {np.percentile(np.abs(v), 99):.4f}, max {np.abs(v).max():.4f}")


def plot_by_redshift(results, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    colours = plt.cm.viridis(np.linspace(0, 1, len(results)))
    for result, colour in zip(sorted(results, key=lambda r: r["z"]), colours):
        axes[0].semilogx(k, 100*result["gp1"], color=colour, label=f'z={result["z"]:.2f}')
        axes[1].semilogx(k, 100*result["fl1"], color=colour)
    for ax, name in zip(axes, ("GP", "cINN (forest_mpg_cinn)")):
        ax.axhline(0, color="k", lw=.5)
        ax.set(xlabel="k [1/Mpc]", title=f"{title}: {name}")
    axes[0].set_ylabel("P1D / fitted-target P1D - 1 [%]")
    axes[0].legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


# %% [markdown]
# ### Seed simulation (one line per redshift)
#
# `mpg_seed` is not in the hypercube, so neither emulator was trained on it.
# Its snapshots are the archive's phase/axis averages at the unrescaled mean flux.

# %%
from forestflow.archive.gadget_archive import GadgetArchive3D
seed_records = GadgetArchive3D(average="both").get_testing_data("mpg_seed")
seed_inputs = [{name: float(r[name]) for name in bundle.input_labels} for r in seed_records]
seed_gp = bundle.evaluate(seed_inputs)
seed_flow = flow.evaluate(seed_inputs, Nrealizations=n, seed=1701,
                          sampler="sobol_antithetic", statistic="mean",
                          aggregation_space="transformed", draw_policy="nested")
seed_results = compare_simulation(
    seed_records,
    coefficients(np.column_stack([seed_gp[name] for name in NAMES])),
    coefficients(np.column_stack([seed_flow[name] for name in NAMES])),
)
summarise("mpg_seed", seed_results)
plot_by_redshift(seed_results, "mpg_seed")

# %% [markdown]
# ### Held-out hypercube simulations
#
# Same comparison for the held-out simulations used above (their records are
# grouped by simulation so CAMB runs once per simulation, not per snapshot).
# The cINN rows reuse the predictions computed in the flow comparison.

# %%
power_idx = list(range(2)) if bundle.metadata["smoke"] else list(range(len(records)))
gp_rows = coefficients(physical_array)
flow_rows = coefficients(flow_predictions[-1])
heldout_results = []
for label in sorted({str(records[i]["sim_label"]) for i in power_idx}):
    idx = [i for i in power_idx if str(records[i]["sim_label"]) == label]
    heldout_results += compare_simulation([records[i] for i in idx],
                                          [gp_rows[i] for i in idx],
                                          [flow_rows[i] for i in idx])
summarise("held-out", heldout_results)

# P1D deviations, one panel per method, coloured by simulation
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
labels = sorted({str(r["record"]["sim_label"]) for r in heldout_results})
colour_of = dict(zip(labels, plt.cm.tab10(np.arange(len(labels)))))
for r in heldout_results:
    lab = str(r["record"]["sim_label"])
    for ax, key in zip(axes, ("gp1", "fl1")):
        ax.semilogx(k, 100*r[key], color=colour_of[lab], alpha=.35, lw=.8)
for ax, name in zip(axes, ("GP", "cINN (forest_mpg_cinn)")):
    ax.axhline(0, color="k", lw=.5)
    ax.set(xlabel="k [1/Mpc]", title=name)
for lab in labels:
    axes[0].plot([], [], color=colour_of[lab], label=lab)
axes[0].set_ylabel("P1D / fitted-target P1D - 1 [%]")
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.show()
plt.close(fig)

# %% [markdown]
# ### Measured-spectrum check (native grid)
#
# Compare each model's P1D to the *measured* archive P1D for the held-out
# records, exposing the fit/model error on top of the regression error.

# %%
for r in heldout_results[:10]:
    record, model, linear, z = r["record"], r["model"], r["linear"], r["z"]
    native_k = np.asarray(record["k_Mpc"]); native_p = np.asarray(record["p1d_Mpc"])
    mask = (native_k >= k.min()) & (native_k <= k.max()) & (native_p > 0)
    rms = lambda coeff: np.sqrt(np.mean((model.P1D_Mpc(linear, z, native_k[mask], coeff)/native_p[mask] - 1)**2))
    print(str(record["sim_label"]), f"z={z:.2f}", "P1D vs measured, RMS fit/GP/cINN:",
          f'{rms(r["fit_coeff"]):.4f} {rms(r["gp_coeff"]):.4f} {rms(r["flow_coeff"]):.4f}')

# %%
