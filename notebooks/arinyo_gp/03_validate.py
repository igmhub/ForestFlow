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
flow = P3DEmulator(key="forest_mpg_fix")
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
# ## P1D/P3D regression impact using matched cosmology
#
# Compute both spectra with identical ArinyoModel, redshift and linear theory.
# Reference spectra here use the fitted coefficients, isolating regression error.
# We also compare native-grid P1D with the measured archive P1D, exposing the
# additional fit/model error. A scientific result needs all relevant simulations,
# grids and errors; the first few records below only demonstrate the calculation.
# Keep Mpc units throughout. No velocity-coordinate Jacobian is applied.

# %%
from lace.cosmo import cosmology
from forestflow.model.arinyo import ArinyoModel
k = np.geomspace(0.1, 4.0, 30)
mu = np.linspace(0., 1., 7)
kg, mug = np.meshgrid(k, mu, indexing="ij")
power_rows = records[:2] if bundle.metadata["smoke"] else records
p1_errors, p3_errors = [], []
fig, ax = plt.subplots(figsize=(8, 4))
for record in power_rows:
    model = ArinyoModel(cosmology.Cosmology(cosmo_params_dict=record["cosmo_params"]))
    z = float(record["z"])
    linear = model.linear.get_linear_theory([z])
    call = {name: float(record[name]) for name in bundle.input_labels}
    gp_coeff = bundle.evaluate(call)
    fit_coeff = {name: record[fit_key][name] for name in bundle.output_labels}
    gp1 = model.P1D_Mpc(linear, z, k, gp_coeff)
    ref1 = model.P1D_Mpc(linear, z, k, fit_coeff)
    gp3 = model.P3D_Mpc_k_mu(linear, z, kg, mug, gp_coeff)
    ref3 = model.P3D_Mpc_k_mu(linear, z, kg, mug, fit_coeff)
    if not all(np.isfinite(v).all() and np.all(v > 0) for v in [gp1, ref1, gp3, ref3]):
        raise ValueError("Nonfinite or nonpositive power; inspect this held-out prediction")
    p1_errors.append(gp1/ref1 - 1)
    p3_errors.append((gp3/ref3 - 1).ravel())
    ax.semilogx(k, 100*(gp1/ref1 - 1), label=f'{record["sim_label"]}, z={z:.2f}')
    native_k = np.asarray(record["k_Mpc"])
    native_p = np.asarray(record["p1d_Mpc"])
    mask = (native_k >= k.min()) & (native_k <= k.max()) & (native_p > 0)
    predicted = model.P1D_Mpc(linear, z, native_k[mask], gp_coeff)
    fitted = model.P1D_Mpc(linear, z, native_k[mask], fit_coeff)
    print(str(record["sim_label"]), z, "P1D measured-spectrum RMS, GP/fit:",
          np.sqrt(np.mean((predicted/native_p[mask]-1)**2)),
          np.sqrt(np.mean((fitted/native_p[mask]-1)**2)))
for label, errors in [("P1D", p1_errors), ("P3D", p3_errors)]:
    values = np.asarray(errors)
    print(label, "fractional RMS / p99 / max:",
          np.sqrt(np.mean(values**2)), np.percentile(np.abs(values), 99), np.abs(values).max())
ax.axhline(0, color="k", lw=.5)
ax.set(xlabel="k [1/Mpc]", ylabel="GP / fitted-target P1D - 1 [%]")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
plt.close(fig)

# %%
