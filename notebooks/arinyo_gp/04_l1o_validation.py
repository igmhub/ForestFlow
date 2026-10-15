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
# # 4. GP trained on the full hypercube, and leave-one-out GPs
#
# This notebook trains
#
# - one GP on **all** training simulations (the 30 hypercube simulations plus
#   `mpg_central`), and
# - leave-one-out (L1O) GPs, each holding out **one** of those simulations,
#
# and validates them against the cINN `forest_mpg_fix` and the fitted
# `arinyo_fixp3d` parameters, with the same plots as `emulator/Emulator_precision`:
#
# 1. redshift evolution of the Arinyo parameters for `mpg_seed`,
# 2. P1D precision for `mpg_central`, one panel per redshift,
# 3. P3D precision for `mpg_central`, one panel per redshift.
#
# `mpg_seed` is never in a training set, so every model is a fair test there.
# For `mpg_central`, the fair GP is the one that held it out. The full GP and the
# cINN have seen it, so they only measure the fit/regression error.
# P1D/P3D ratios are emulator divided by the projection of the saved Arinyo fit,
# minus one, using the same hybrid finite-volume P3D average as the fits.
#
# The exact `forest_mpg_fix` input/output transform is reused for all GPs (as in
# notebook 1, `TRANSFORM_MODE="bundle"`); its fixed moments can contain information
# about held-out simulations, so absolute L1O numbers are slightly optimistic.

# %%
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import forestflow
ROOT = Path(forestflow.__file__).resolve().parent.parent
HERE = ROOT / "notebooks" / "arinyo_gp"
sys.path.insert(0, str(HERE))
from threadpoolctl import threadpool_limits
from forestflow.archive.gadget_archive import GadgetArchive3D
from forestflow.emulator.p3d_cinn import P3DEmulator
from forestflow.emulator.arinyo_gp import (
    ArinyoGPConfig, ArinyoGPEmulator, prepare_arinyo_gp_training_data,
    select_groups, train_arinyo_gp,
)
from forestflow.model_fits import ArinyoFitter
from forestflow.statistics.rebin_p3d import rebin_P3D_Mpc_mode_weighted

# %% [markdown]
# ## Configuration
#
# `HOLD_OUT` lists the simulations for which an L1O GP is trained. Use
# `HOLD_OUT = "all"` for the complete leave-one-out set (31 GPs; slow). The
# kernel (`NU`, `ALPHA`) is the one selected in notebook 1. Trained models are
# cached in `artifacts/l1o/` and reused unless `RETRAIN = True`.

# %%
HOLD_OUT = ["mpg_central", "mpg_18"]   # or "all"
NU, ALPHA = 2.5, 1e-2
OPTIMIZE_KERNEL = True                 # False: fixed unit length scales (smoke test only)
RETRAIN = False
FIT_KEY = "arinyo_fixp3d"
N_REALIZATIONS = 1000                  # cINN latent draws, as in Emulator_precision
CACHE = HERE / "artifacts" / "l1o"
CACHE.mkdir(parents=True, exist_ok=True)

# %%
flow = P3DEmulator(key="forest_mpg_fix", Nrealizations=N_REALIZATIONS)
archive = GadgetArchive3D(postproc="Cabayol23_fixp3d", addcentral=True)
records = [r for r in archive.training_data if r["z"] <= 4.6]
inputs, outputs = tuple(flow.input_labels), tuple(flow.output_labels)
data = prepare_arinyo_gp_training_data(records, flow.transf_data, inputs, outputs, fit_key=FIT_KEY)
groups = sorted(set(data.group_ids))
print(len(records), "rows;", len(groups), "simulations:", groups[:3], "...", groups[-2:])
holdouts = groups if HOLD_OUT == "all" else list(HOLD_OUT)
assert set(holdouts) <= set(groups), set(holdouts) - set(groups)

# %% [markdown]
# ## Train (or load) the GPs

# %%
def get_gp(name, training):
    path = CACHE / f"{name}.pkl"
    if path.is_file() and not RETRAIN:
        return ArinyoGPEmulator.load(path)
    with threadpool_limits(limits=1):
        model = train_arinyo_gp(
            training, flow.transf_data,
            ArinyoGPConfig(nu=NU, alpha=ALPHA, optimize_kernel=OPTIMIZE_KERNEL, random_state=32),
            metadata={"name": name, "fit_key": FIT_KEY},
        )
    model.save(path)
    return model


gps = {"all": get_gp("all", data)}
for group in holdouts:
    keep = set(groups) - {group}
    gps[f"l1O {group}"] = get_gp(f"l1O_{group}", select_groups(data, keep))
    assert group not in gps[f"l1O {group}"].metadata["groups"]
for name, model in gps.items():
    print(name, "->", model.gp.kernel_)

# %% [markdown]
# ## Held-out accuracy in transformed parameter space
#
# For each L1O GP, RMSE of the transformed Arinyo parameters on its held-out
# simulation, next to the GP trained on everything (which has seen it).

# %%
def transformed_rmse(model, group):
    rows = [r for r in records if str(r["sim_label"]) == group]
    pred = model.predict_transformed([{k: float(r[k]) for k in inputs} for r in rows])
    ref = prepare_arinyo_gp_training_data(rows, flow.transf_data, inputs, outputs, fit_key=FIT_KEY).targets
    pred = np.column_stack([pred[k] for k in outputs])
    return np.sqrt(np.mean((pred - ref) ** 2, axis=0))


print(f"{'held-out':12s} {'model':16s} " + " ".join(f"{k:>7s}" for k in outputs))
for group in holdouts:
    for name in ("all", f"l1O {group}"):
        print(f"{group:12s} {name:16s} " + " ".join(f"{v:7.3f}" for v in transformed_rmse(gps[name], group)))

# %% [markdown]
# ## Predictions on the validation snapshots
#
# The cINN is evaluated in batches with common latent samples, exactly as in
# `Emulator_precision`. GP predictions are deterministic posterior means.

# %%
def emulator_input(snapshot):
    return {name: float(snapshot[name]) for name in inputs}


def cinn_predict(snapshots, chunk_size=128):
    out = {name: [] for name in flow.output_labels}
    rows = [emulator_input(s) for s in snapshots]
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start:start + chunk_size]
        pred = flow.evaluate(chunk, Nrealizations=N_REALIZATIONS, seed=0,
                             latent_indices=np.zeros(len(chunk), dtype=int))
        for name in flow.output_labels:
            out[name].extend(np.asarray(pred[name]).reshape(-1))
    return [{name: float(out[name][i]) for name in flow.output_labels} for i in range(len(snapshots))]


def gp_predict(model, snapshots):
    pred = model.evaluate([emulator_input(s) for s in snapshots])
    return [{name: float(pred[name][i]) for name in outputs} for i in range(len(snapshots))]


seed_snapshots = archive.get_testing_data("mpg_seed")
central_snapshots = archive.get_testing_data("mpg_central")
for s in seed_snapshots + central_snapshots:
    assert FIT_KEY in s, "run the fits for the testing simulations first"

COLOURS = {"cINN forest_mpg_fix": "k", "all": "C0"}
for i, name in enumerate(n for n in gps if n != "all"):
    COLOURS[name] = f"C{i + 1}"


def predictions(snapshots, models):
    result = {"cINN forest_mpg_fix": cinn_predict(snapshots)}
    for name in models:
        result[name] = gp_predict(gps[name], snapshots)
    return result


# the seed simulation is unseen by every model; for the central only the L1O GP
# that held it out is a fair test, the others are shown for reference
seed_pred = predictions(seed_snapshots, list(gps))
central_models = ["all"] + [n for n in gps if n == "l1O mpg_central"]
if "l1O mpg_central" not in gps:
    print("WARNING: add 'mpg_central' to HOLD_OUT for a fair central-simulation test")
central_pred = predictions(central_snapshots, central_models)

# %% [markdown]
# ## 1. Redshift evolution of the Arinyo parameters: `mpg_seed`
#
# Points are the independent `arinyo_fixp3d` fits. No model has seen this
# simulation, so differences are genuine generalisation error plus fit noise.

# %%
z_seed = np.asarray([s["z"] for s in seed_snapshots])
order = np.argsort(z_seed)
fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharex=True)
for axis, name in zip(axes.flat, ArinyoFitter.PARAM_NAMES):
    fit = np.asarray([s[FIT_KEY][name] for s in seed_snapshots])
    axis.plot(z_seed[order], fit[order], "o", color="C9", ms=4, label="arinyo_fixp3d (fit)")
    for label, rows in seed_pred.items():
        values = np.asarray([row[name] for row in rows])
        axis.plot(z_seed[order], values[order], color=COLOURS[label], lw=1.4,
                  ls="--" if label == "cINN forest_mpg_fix" else "-", label=label)
    axis.set_title(name)
    axis.grid(alpha=0.25)
for axis in axes[1]:
    axis.set_xlabel("redshift")
axes[0, 0].set_ylabel("Arinyo parameter")
axes[1, 0].set_ylabel("Arinyo parameter")
axes[0, 0].legend(fontsize=7)
fig.suptitle("mpg_seed: redshift evolution of the Arinyo parameters", y=1.0)
fig.tight_layout()

# %% [markdown]
# ## 2. P1D and 3. P3D precision for `mpg_central`, one panel per redshift
#
# Ratios of each emulator's projection to the projection of the saved
# `arinyo_fixp3d` fit, minus one. Colour = emulator; for P3D, line style = mu bin.

# %%
fit_settings = dict(kmin_3d=0.01, kmax_3d=4.5, kmin_1d=0.01, kmax_1d=6.0)
n_mu_bins = 4
MU_STYLES = ["-", "--", "-.", ":"]


def projected_ratio(snapshot, emulated, fitter):
    """Emulator / Arinyo-fit P3D and P1D ratios on the fit geometry."""
    fitter.prepare_simulation(snapshot, is_mpg=True)
    fit_p3d, fit_p1d = fitter.predict(fitter.params_from_dict(snapshot[FIT_KEY]))
    emu_p3d, emu_p1d = fitter.predict(fitter.params_from_dict(emulated))
    k3d, _, fit_p3d, mu_edges = rebin_P3D_Mpc_mode_weighted(
        fitter.data.k3d, fitter.data.mu3d, fit_p3d, fitter._mpg_k_mu_modes, n_mu_bins=n_mu_bins)
    _, _, emu_p3d, _ = rebin_P3D_Mpc_mode_weighted(
        fitter.data.k3d, fitter.data.mu3d, emu_p3d, fitter._mpg_k_mu_modes, n_mu_bins=n_mu_bins)
    return {"k3d": k3d, "mu_edges": mu_edges, "P3D": emu_p3d / fit_p3d - 1,
            "k1d": fitter.data.k1d.copy(), "P1D": emu_p1d / fit_p1d - 1}


fitter = ArinyoFitter(**fit_settings)
central_ratios = {
    label: [projected_ratio(snap, params, fitter) for snap, params in zip(central_snapshots, rows)]
    for label, rows in central_pred.items()
}
z_central = np.asarray([s["z"] for s in central_snapshots])
central_order = np.argsort(z_central)


def plot_by_redshift(kind):
    n_cols = 4
    n_rows = int(np.ceil(len(central_order) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.3 * n_rows), squeeze=False, sharey=True)
    tolerance = 0.1 if kind == "P3D" else 0.01
    for position, index in enumerate(central_order):
        axis = axes[divmod(position, n_cols)]
        axis.axhspan(-tolerance, tolerance, color="0.8", alpha=0.35, zorder=0)
        for label, ratios in central_ratios.items():
            ratio = ratios[index]
            if kind == "P1D":
                axis.plot(ratio["k1d"], ratio["P1D"], color=COLOURS[label], label=label)
            else:
                for mu in range(n_mu_bins):
                    valid = np.isfinite(ratio["k3d"][:, mu])
                    axis.plot(ratio["k3d"][valid, mu], ratio["P3D"][valid, mu], color=COLOURS[label],
                              ls=MU_STYLES[mu], lw=1.0, label=label if mu == 0 else None)
        axis.axvline(fit_settings["kmax_1d" if kind == "P1D" else "kmax_3d"], color="k", ls="--", lw=.6)
        axis.axhline(0, color="k", ls=":", lw=.6)
        axis.set_xscale("log")
        axis.set_title(f"z = {z_central[index]:.2f}")
        axis.grid(alpha=0.25)
        if position == 0:
            axis.legend(fontsize=7)
    for position in range(len(central_order), n_rows * n_cols):
        axes[divmod(position, n_cols)].set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel(r"$k_\parallel\,[\mathrm{Mpc}^{-1}]$" if kind == "P1D" else r"$k\,[\mathrm{Mpc}^{-1}]$")
    for axis in axes[:, 0]:
        axis.set_ylabel(rf"$P_\mathrm{{{kind[1:]}}}^\mathrm{{emu}}/P_\mathrm{{{kind[1:]}}}^\mathrm{{fit}}-1$")
    suffix = "; line style = mu bin" if kind == "P3D" else ""
    fig.suptitle(f"mpg_central {kind} precision{suffix}", y=1.0)
    fig.tight_layout()


plot_by_redshift("P1D")

# %%
plot_by_redshift("P3D")

# %% [markdown]
# ## Summary for `mpg_central`
#
# Fractional P1D and P3D deviation from the fit over all redshifts (RMS over
# snapshots and scales within the fitted range).

# %%
for label, ratios in central_ratios.items():
    p1 = np.concatenate([r["P1D"] for r in ratios])
    p3 = np.concatenate([r["P3D"][np.isfinite(r["P3D"])] for r in ratios])
    print(f"{label:22s} P1D RMS {np.sqrt(np.mean(p1**2)):.4f}  P3D RMS {np.sqrt(np.mean(p3**2)):.4f}")

# %%
