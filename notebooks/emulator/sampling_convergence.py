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
# # ForestFlow latent-sampling convergence
#
# Compare the Monte-Carlo average used by `forest_mpg_fix` across ordinary
# Gaussian, antithetic, and scrambled-Sobol latent draws.  The default
# estimator remains the historical mean in transformed Arinyo space; the
# optional physical-space and median estimators are shown separately because
# they define different predictions.

# %%
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch

from forestflow.emulator.p3d_cinn import P3DEmulator

torch.set_num_threads(1)

# %% [markdown]
# ## Configuration
#
# Use smoke mode first. The reference must be independently checked before it
# is treated as converged. Sobol counts are powers of two; antithetic counts
# must be even.

# %%
SMOKE = True
COUNTS = (32, 128, 512) if SMOKE else tuple(2**power for power in range(5, 15))
REFERENCE_COUNT = 4096 if SMOKE else 65536
SEEDS = (3, 17, 91) if SMOKE else tuple(range(16))
METHODS = (
    ("gaussian", "mean", "transformed", "nested"),
    ("antithetic", "mean", "transformed", "nested"),
    ("sobol", "mean", "transformed", "nested"),
    ("gaussian", "median", "transformed", "nested"),
    ("gaussian", "mean", "physical", "nested"),
    ("gaussian", "median", "physical", "nested"),
)

INPUT = {
    "Delta2_p": 0.18489945277410613,
    "n_p": -2.331713201486465,
    "mF": 0.23475637218289533,
    "sigT_Mpc": 0.10040737452608385,
    "gamma": 1.2115605945334802,
    "kF_Mpc": 14.191866950067904,
}


# %% [markdown]
# ## Load the bundle and define compact diagnostics

# %%
emulator = P3DEmulator(key="forest_mpg_fix", compile_model=True)
labels = tuple(emulator.output_labels)
print("Outputs:", labels)
print("Device:", next(emulator.emulator.parameters()).device)


def vector(prediction):
    return np.asarray([prediction[name] for name in labels], dtype=float)


def evaluate(method, count, seed):
    sampler, statistic, space, policy = method
    start = perf_counter()
    prediction = emulator.evaluate(
        INPUT,
        Nrealizations=count,
        seed=seed,
        sampler=sampler,
        statistic=statistic,
        aggregation_space=space,
        draw_policy=policy,
    )
    return vector(prediction), perf_counter() - start


# Warm compiled shapes before timing.
for method in METHODS:
    _ = evaluate(method, COUNTS[0], SEEDS[0])


# %% [markdown]
# ## Coefficient convergence against per-estimator references
#
# Every estimator has its own reference. A persistent displacement between
# estimators is a scientific estimator change, rather than finite-N noise.

# %%
references = {}
for method in METHODS:
    vectors = [evaluate(method, REFERENCE_COUNT, seed)[0] for seed in SEEDS]
    references[method] = np.mean(vectors, axis=0)

rows = []
for method in METHODS:
    reference = references[method]
    for count in COUNTS:
        for seed in SEEDS:
            prediction, seconds = evaluate(method, count, seed)
            rows.append(
                {
                    "method": method,
                    "count": count,
                    "seed": seed,
                    "seconds": seconds,
                    "max_relative_error": np.max(
                        np.abs(prediction - reference)
                        / np.maximum(np.abs(reference), 1e-12)
                    ),
                    "prediction": prediction,
                }
            )

for method in METHODS:
    selected = [row for row in rows if row["method"] == method]
    x = np.asarray([row["count"] for row in selected])
    y = np.asarray([row["max_relative_error"] for row in selected])
    plt.scatter(x, y, label="/".join(method), alpha=0.75)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("ForestFlow realizations")
plt.ylabel("max relative Arinyo error")
plt.legend(fontsize=7)
plt.show()


# %% [markdown]
# ## P1D and DR1 χ² follow-up
#
# The next implementation step can add P1D projection and cup1d DR1 likelihood
# calls here. Keep the cosmology, quadrature, covariance, contaminants, and
# fixed parameter point unchanged across the realization-count scan. Use a
# saved Forest-MPG fit through cup1d's supported result-restoration API rather
# than running a new fit inside this notebook.
