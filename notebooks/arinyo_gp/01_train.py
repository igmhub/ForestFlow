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
# # 1. Fit a GP to transformed Arinyo coefficients
#
# The six conditioning inputs predict the eight fitted Arinyo targets from
# the corrected archive. No flow-generated training labels are used.
# We reuse LaCE's sklearn/Matérn machinery and (outside smoke mode) its optimizer.
# The P1D-specific LaCE GP wrapper is intentionally not instantiated.
#
# Run this first. Each run gets its own artifact directory. Smoke mode verifies
# execution with a small training subset; it cannot establish scientific precision.

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
ARTIFACTS = HERE / "artifacts"

# %%
from datetime import datetime, timezone
import platform
import subprocess
import sklearn
from threadpoolctl import threadpool_limits
from forestflow.archive.gadget_archive import GadgetArchive3D
from forestflow.emulator.p3d_cinn import P3DEmulator
from forestflow.emulator.arinyo_gp import (
    ArinyoGPConfig, ArinyoGPTrainingData, ArinyoGPEmulator, train_arinyo_gp,
)
from forestflow.emulator.training import Transf_data

SMOKE = False
TRANSFORM_MODE = "bundle"  # exact forest_mpg_fix scaling; alternative: "train_fold"
FIT_KEY = "arinyo_fixp3d"
SEED = 57
flow = P3DEmulator(key="forest_mpg_fix")
archive = GadgetArchive3D(postproc="Cabayol23_fixp3d", addcentral=True)
records = [r for r in archive.training_data if r["z"] <= 4.6]
inputs, outputs = tuple(flow.input_labels), tuple(flow.output_labels)
print("Archive rows:", len(records), "labels:", inputs, outputs)

# %% [markdown]
# ## Split whole simulations before fitting
#
# All redshifts and rescalings of a simulation stay together. The archive includes
# central snapshots to match the documented training recipe. Test groups remain
# untouched until notebook 03. ForestFlow may already have seen them, so later
# GP-flow differences are not a fair independent accuracy ranking.
# Fixed bundle moments also may contain held-out information. For independent
# generalization estimates use TRANSFORM_MODE="train_fold", which retains exactly
# the same transform rules but estimates scaling from GP training records alone.

# %%
groups = np.array(sorted({str(r["sim_label"]) for r in records}))
rng = np.random.default_rng(SEED)
groups = rng.permutation(groups)
test_groups, val_groups, train_groups = groups[:3], groups[3:6], groups[6:]
splits = {name: [r for r in records if str(r["sim_label"]) in selected]
          for name, selected in [("train", train_groups), ("validation", val_groups), ("test", test_groups)]}
assert set(train_groups).isdisjoint(val_groups) and set(train_groups).isdisjoint(test_groups)
assert set(val_groups).isdisjoint(test_groups)
if SMOKE:
    for name, cap in [("train", 96), ("validation", 24), ("test", 24)]:
        rows = splits[name]
        splits[name] = [rows[i] for i in rng.choice(len(rows), min(cap, len(rows)), replace=False)]
print({name: len(rows) for name, rows in splits.items()})
train_x = ex.columns(splits["train"], inputs)
train_y = ex.columns(splits["train"], outputs, FIT_KEY)
if TRANSFORM_MODE == "bundle":
    transform = flow.transf_data
elif TRANSFORM_MODE == "train_fold":
    transform = Transf_data(dict_all_params={"input_par": train_x, "output_par": train_y})
else:
    raise ValueError("Unknown transform mode")
x = ex.matrix(transform, train_x, inputs, "input")
y = ex.matrix(transform, train_y, outputs, "output")
training_data = ArinyoGPTrainingData(
    x, y, inputs, outputs, tuple(str(row["sim_label"]) for row in splits["train"]), FIT_KEY
)
vx = ex.matrix(transform, ex.columns(splits["validation"], inputs), inputs, "input")
vy = ex.matrix(transform, ex.columns(splits["validation"], outputs, FIT_KEY), outputs, "output")
print("Exact duplicate conditioning rows:", len(x) - len(np.unique(x, axis=0)))
print("Dense kernel storage, excluding optimizer work:", x.shape[0]**2 * 8 / 1e6, "MB")

# %% [markdown]
# ## Kernel and nugget comparison
#
# nu=0.5 reproduces LaCE's kernel family. nu=1.5 and 2.5 allow smoother functions.
# alpha is a variance in standardized target units, not a fractional power error.
# The shared kernel fits all outputs but does not learn their cross-covariance.
# Record optimizer warnings; bound hits are diagnostics, not proof of success.
# Production selection should include power errors on validation simulations.

# %%
nu_grid = (0.5, 1.5) if SMOKE else (0.5, 1.5, 2.5)
alpha_grid = (1e-4,) if SMOKE else (1e-6, 1e-4, 1e-2)
models, scores = [], []
with threadpool_limits(limits=1):
    for nu in nu_grid:
        for alpha in alpha_grid:
            candidate = train_arinyo_gp(
                training_data, transform,
                ArinyoGPConfig(nu=nu, alpha=alpha, optimize_kernel=True, random_state=32),
            )
            score = float(np.mean((candidate.gp.predict(vx) - vy)**2))
            models.append(candidate)
            scores.append({"nu": nu, "alpha": alpha, "validation_mse": score,
                           "kernel": str(candidate.gp.kernel_)})
            print(scores[-1])
best = int(np.argmin([row["validation_mse"] for row in scores]))
emulator = models[best]

# %%
fig, ax = plt.subplots(figsize=(7, 3))
ax.bar([str((s["nu"], s["alpha"])) for s in scores], [s["validation_mse"] for s in scores])
ax.set(xlabel="(Matérn nu, alpha)", ylabel="Validation transformed MSE", yscale="log")
plt.tight_layout()

# %%
{'nu': 2.5, 'alpha': 0.01, 'validation_mse': 0.011180849656661822, 'kernel': 'Matern(length_scale=[3.05, 13.4, 2.3, 5.51, 6.1, 3.82], nu=2.5)'}

# %% [markdown]
# ## Save the frozen choice and held-out records
#
# No test result participates in this choice. Repeatedly tuning after inspecting
# notebook 03 invalidates its final-test interpretation: use a new holdout.
# Local pickle artifacts are for this installed environment, not a portable
# production format. Exact recipe compatibility still requires a bundle audit.

# %%
ARTIFACTS.mkdir(parents=True, exist_ok=True)
run = ARTIFACTS / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
run.mkdir()
metadata = dict(smoke=SMOKE, fit_key=FIT_KEY, transform_mode=TRANSFORM_MODE,
                seed=SEED, scores=scores, selected=best, sklearn=sklearn.__version__,
                python=platform.python_version(), training_hash=ex.fingerprint(x, y),
                groups={k: list(map(str, v)) for k, v in
                        [("train", train_groups), ("validation", val_groups), ("test", test_groups)]})
for name, repo in [("forestflow", ROOT), ("lace", ROOT.parent / "LaCE")]:
    metadata[name + "_sha"] = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    metadata[name + "_dirty"] = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()
emulator.metadata.update(metadata)
emulator.save(run / "gp.pkl")
ex.save_bundle(run / "test_records.pkl", splits["test"])
(run / "metadata.json").write_text(json.dumps(metadata, indent=2))
(ARTIFACTS / "latest.json").write_text(json.dumps({"run": str(run)}))
reloaded = ArinyoGPEmulator.load(run / "gp.pkl")
np.testing.assert_allclose(reloaded.gp.predict(vx), emulator.gp.predict(vx))
print("Saved and reload-checked:", run)

# %%
