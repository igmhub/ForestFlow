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
# # 5. Speed: cINN `forest_mpg_fix` versus GP `forest_mpg_gp`
#
# This notebook only measures how long it takes to turn emulator inputs
# (`Delta2_p, n_p, mF, sigT_Mpc, gamma, kF_Mpc`) into the eight Arinyo
# parameters. It does not compute P1D/P3D and says nothing about accuracy (see
# notebook 4).
#
# The cINN draws `Nrealizations` latent samples per input and aggregates them, so
# its cost scales with that number; the GP is a deterministic posterior mean.
# A likelihood call evaluates one input per redshift (about 10 to 15 rows), so
# small batches are the relevant case; large batches show the throughput.
# Timings depend on the machine and thread count: compare the ratios, and rerun
# on the machine you care about.

# %%
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import forestflow
from forestflow.archive.gadget_archive import GadgetArchive3D
from forestflow.emulator.p3d_cinn import P3DEmulator
from forestflow.emulator.arinyo_gp import ArinyoGPEmulator

ROOT = Path(forestflow.__file__).resolve().parent.parent
BATCH_SIZES = [1, 12]
N_REALIZATIONS = [2048]
# [100, 1000, 2048]      # cINN latent draws (2048 is the default)
REPEATS = 10
print("torch threads:", torch.get_num_threads())

# %% [markdown]
# ## Inputs
#
# Inputs are rows drawn from the corrected training archive (so they are inside
# the training hull).

# %%
archive = GadgetArchive3D(postproc="Cabayol23_fixp3d")
pool = [r for r in archive.training_data if r["z"] <= 4.6]
rng = np.random.default_rng(0)
INPUT_LABELS = ("Delta2_p", "n_p", "mF", "sigT_Mpc", "gamma", "kF_Mpc")


def make_inputs(n):
    rows = [pool[i] for i in rng.choice(len(pool), n, replace=n > len(pool))]
    return [{k: float(r[k]) for k in INPUT_LABELS} for r in rows]


# %% [markdown]
# ## Load the emulators (and time loading)

# %%
t = time.perf_counter()
gp = ArinyoGPEmulator.load(ROOT / "data" / "gp_emulators" / "forest_mpg_gp.pkl")
print(f"forest_mpg_gp load: {time.perf_counter() - t:.2f} s; "
      f"{gp.gp.X_train_.shape[0]} training rows")
t = time.perf_counter()
cinn = P3DEmulator(key="forest_mpg_fix", Nrealizations=max(N_REALIZATIONS))
print(f"forest_mpg_fix load: {time.perf_counter() - t:.2f} s")
assert tuple(cinn.input_labels) == INPUT_LABELS == tuple(gp.input_labels)

# %% [markdown]
# ## Timing
#
# Each configuration is warmed up once and then timed `REPEATS` times; we report
# the median and the minimum. The cINN uses its production settings
# (`sampler="sobol_antithetic"`, transformed-space mean).

# %%
def timeit(function, repeats=REPEATS):
    function()  # warm-up
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return float(np.median(times)), float(np.min(times))


def cinn_call(rows, n):
    return lambda: cinn.evaluate(
        rows,
        Nrealizations=n,
        seed=0,
        sampler="sobol_antithetic",
        statistic="mean",
        aggregation_space="transformed",
        draw_policy="nested",
    )


results = {"GP forest_mpg_gp": {}}
results.update({f"cINN forest_mpg_fix (N={n})": {} for n in N_REALIZATIONS})
for batch in BATCH_SIZES:
    rows = make_inputs(batch)
    results["GP forest_mpg_gp"][batch] = timeit(lambda: gp.evaluate(rows))
    for n in N_REALIZATIONS:
        results[f"cINN forest_mpg_fix (N={n})"][batch] = timeit(cinn_call(rows, n))
    print(batch, {k: f"{v[batch][0] * 1e3:.2f} ms" for k, v in results.items()})

# %% [markdown]
# ## Results

# %%
print(f"{'median time per call [ms]':32s}" + "".join(f"{b:>10d}" for b in BATCH_SIZES))
for name, per_batch in results.items():
    print(f"{name:32s}" + "".join(f"{per_batch[b][0] * 1e3:10.2f}" for b in BATCH_SIZES))
print()
gp_times = results["GP forest_mpg_gp"]
for name, per_batch in results.items():
    if name.startswith("cINN"):
        print(f"{name:32s} slower than GP by " +
              " ".join(f"{per_batch[b][0] / gp_times[b][0]:7.1f}x" for b in BATCH_SIZES))

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for index, (name, per_batch) in enumerate(results.items()):
    median = np.asarray([per_batch[b][0] for b in BATCH_SIZES])
    best = np.asarray([per_batch[b][1] for b in BATCH_SIZES])
    # one colour per emulator configuration (GP first, then each cINN setting)
    style = dict(color=f"C{index}", lw=2 if name.startswith("GP") else 1.5)
    axes[0].loglog(BATCH_SIZES, median * 1e3, "o-", label=name, **style)
    axes[0].fill_between(
        BATCH_SIZES, best * 1e3, median * 1e3, alpha=0.15, color=style["color"]
    )
    axes[1].loglog(
        BATCH_SIZES, median / np.asarray(BATCH_SIZES) * 1e3, "o-", label=name, **style
    )
axes[0].set(
    xlabel="number of inputs per call", ylabel="time per call [ms]", title="Latency"
)
axes[1].set(
    xlabel="number of inputs per call", ylabel="time per input [ms]", title="Throughput"
)
for axis in axes:
    axis.grid(alpha=0.25, which="both")
axes[0].legend(fontsize=7)
fig.tight_layout()

# %% [markdown]
# ## Compiled cINN
#
# `P3DEmulator(compile_model=True)` (used by cup1d and lya_interface) compiles the
# network on first use for each input shape, then reuses it. We report the
# first-call (compilation) cost and the steady-state time at a likelihood-like
# batch size.

# %%
LIKELIHOOD_BATCH = 12
rows = make_inputs(LIKELIHOOD_BATCH)
compiled = P3DEmulator(key="forest_mpg_fix", compile_model=True)
call = lambda emulator: emulator.evaluate(
    rows,
    Nrealizations=N_REALIZATIONS[0],
    seed=0,
    sampler="sobol_antithetic",
    statistic="mean",
    aggregation_space="transformed",
    draw_policy="nested",
)
start = time.perf_counter()
call(compiled)
first = time.perf_counter() - start
steady = timeit(lambda: call(compiled))
eager = timeit(lambda: call(cinn))
gp_time = timeit(lambda: gp.evaluate(rows))
print(f"batch {LIKELIHOOD_BATCH}, N={N_REALIZATIONS[0]}:")
print(
    f"  cINN compiled   first call {first:.2f} s, steady median {steady[0] * 1e3:.2f} ms"
)
print(f"  cINN not compiled           steady median {eager[0] * 1e3:.2f} ms")
print(f"  GP forest_mpg_gp                        median {gp_time[0] * 1e3:.2f} ms")
print(f"  compiled cINN / GP = {steady[0] / gp_time[0]:.1f}x")

# %%
