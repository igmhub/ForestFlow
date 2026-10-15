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
# # 2. Predict transformed and physical Arinyo coefficients
#
# Run notebook 01 first. This notebook needs only the GP artifact for inference.
# The flow is loaded later solely for a latency comparison. GP standard deviation
# is in transformed space; it is not a calibrated observational error model.

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
from forestflow.emulator.arinyo_gp import ArinyoGPEmulator
ARTIFACTS = HERE / "artifacts"

# %%
from threadpoolctl import threadpool_limits
run = Path(json.loads((ARTIFACTS / "latest.json").read_text())["run"])
bundle = ArinyoGPEmulator.load(run / "gp.pkl")
records = ex.load_bundle(run / "test_records.pkl")
rows = [{name: float(r[name]) for name in bundle.input_labels} for r in records[:4]]
print("Smoke artifact:", bundle.metadata["smoke"])
print("Selected kernel:", bundle.gp.kernel_)
transformed = bundle.predict_transformed(rows[0])
physical = bundle.evaluate(rows[0])
print("Transformed:", transformed)
print("Physical:", physical)

# %% [markdown]
# The public coefficient prediction is inverse_transform(mean_transformed).
# Because bias and kvav have nonlinear inverse transforms, this is generally
# different from the physical GP posterior mean. We keep ForestFlow's convention.
# Check scalar/batch consistency and the shared forward/inverse mapping explicitly.

# %%
batch = bundle.evaluate(rows)
for name in bundle.output_labels:
    np.testing.assert_allclose(batch[name][0], physical[name], rtol=1e-10, atol=1e-10)
back = bundle.transform.transf_stand(batch, type_stand="output", direct=True)
mean, std = bundle.predict_transformed(rows, return_std=True)
np.testing.assert_allclose(np.column_stack([back[k] for k in bundle.output_labels]),
                           np.column_stack([mean[k] for k in bundle.output_labels]), atol=1e-10)
print("Posterior mean/std shapes:", np.column_stack(list(mean.values())).shape, np.column_stack(list(std.values())).shape)

# %% [markdown]
# ## CPU latency for means versus variances
#
# Both methods are warmed. Fixed single-thread timing includes normalization and
# inverse transformation, but not loading. Flow latency depends on realization
# count. These are coefficient timings, not end-to-end likelihood speedups.

# %%
from forestflow.emulator.p3d_cinn import P3DEmulator
flow = P3DEmulator(key="forest_mpg_cinn")
options = dict(Nrealizations=128 if bundle.metadata["smoke"] else 4096,
               seed=901, sampler="sobol_antithetic", statistic="mean",
               aggregation_space="transformed", draw_policy="nested")
with threadpool_limits(limits=1):
    for calls in [rows[0], rows]:
        print("Batch size:", 1 if isinstance(calls, dict) else len(calls))
        print("GP physical mean:", ex.latency(lambda: bundle.evaluate(calls)))
        print("GP transformed mean + std:", ex.latency(lambda: bundle.predict_transformed(calls, return_std=True)))
        print("Flow physical:", ex.latency(lambda: flow.evaluate(calls, **options), repeats=5))
print("Flow realization count:", options["Nrealizations"])

# %%
