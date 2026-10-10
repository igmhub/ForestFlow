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
# COUNTS = (1000, 5000, 10000)
COUNTS = (1000, 2000)
REFERENCE_COUNT = 100000
SEEDS = (3, 17, 91, 134, 201, 303, 404, 505, 606, 707)
METHODS = (
    # ("gaussian", "mean", "transformed", "nested"),
    # ("antithetic", "mean", "transformed", "nested"),
    # ("gaussian", "median", "transformed", "nested"),
    # ("gaussian", "mean", "physical", "nested"),
    # ("gaussian", "median", "physical", "nested"),
    ("sobol", "mean", "transformed", "nested"),
    ("sobol", "mean", "physical", "nested"),
    ("sobol", "median", "transformed", "nested"),
    ("sobol", "median", "physical", "nested"),
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

summary = []
for method in METHODS:
    for count in COUNTS:
        selected = [row for row in rows if row["method"] == method and row["count"] == count]
        precision = np.asarray([row["max_relative_error"] for row in selected])
        seconds = np.asarray([row["seconds"] for row in selected])
        summary.append({"method": method, "count": count,
                        "precision_mean": precision.mean(), "precision_std": precision.std(ddof=1),
                        "time_mean_seconds": seconds.mean(), "time_std_seconds": seconds.std(ddof=1)})

fig, (precision_axis, time_axis) = plt.subplots(1, 2, figsize=(12, 4))
for method in METHODS:
    selected = [row for row in summary if row["method"] == method]
    counts = np.asarray([row["count"] for row in selected])
    precision_axis.errorbar(counts, [row["precision_mean"] for row in selected],
                            yerr=[row["precision_std"] for row in selected], marker="o", capsize=3,
                            label="/".join(method))
    time_axis.errorbar(counts, [row["time_mean_seconds"] for row in selected],
                       yerr=[row["time_std_seconds"] for row in selected], marker="o", capsize=3,
                       label="/".join(method))
for axis in (precision_axis, time_axis):
    axis.set_xscale("log")
    axis.legend(fontsize=6)
precision_axis.set_yscale("log")
precision_axis.set_xlabel("ForestFlow realizations")
precision_axis.set_ylabel("mean max relative Arinyo error ± seed std")
time_axis.set_xlabel("ForestFlow realizations")
time_axis.set_ylabel("mean evaluation time [s] ± seed std")
fig.tight_layout()
for row in summary:
    print(f"{'/'.join(row['method'])}, N={row['count']:6d}: "
          f"precision={row['precision_mean']:.3e} ± {row['precision_std']:.3e}; "
          f"time={row['time_mean_seconds']:.4f} ± {row['time_std_seconds']:.4f} s")


# %% [markdown]
# ## P1D and DR1 χ² follow-up
#
# The next implementation step can add P1D projection and cup1d DR1 likelihood
# calls here. Keep the cosmology, quadrature, covariance, contaminants, and
# fixed parameter point unchanged across the realization-count scan. Use a
# saved Forest-MPG fit through cup1d's supported result-restoration API rather
# than running a new fit inside this notebook.
