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
# - one GP (`forest_mpg_gp`) on **all** training simulations (the 30 hypercube simulations plus
#   `mpg_central`), and
# - leave-one-out (L1O) GPs, each holding out **one** of those simulations,
#
# and validates them against the cINN `forest_mpg_cinn` and the fitted
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
# The exact `forest_mpg_cinn` input/output transform is reused for all GPs (as in
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
# cached in `data/gp_emulators/` and reused unless `RETRAIN = True`.

# %%
# HOLD_OUT = ["mpg_central", "mpg_18"]   # or "all"
HOLD_OUT = "all"
NU, ALPHA = 2.5, 1e-2
OPTIMIZE_KERNEL = True                 # False: fixed unit length scales (smoke test only)
RETRAIN = False
FIT_KEY = "arinyo_fixp3d"
N_REALIZATIONS = 1000                  # cINN latent draws, as in Emulator_precision
CACHE = ROOT / "data" / "gp_emulators"
CACHE.mkdir(parents=True, exist_ok=True)

# %%
flow = P3DEmulator(key="forest_mpg_cinn", Nrealizations=N_REALIZATIONS)
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
            training,
            flow.transf_data,
            ArinyoGPConfig(
                nu=NU, alpha=ALPHA, optimize_kernel=OPTIMIZE_KERNEL, random_state=32
            ),
            metadata={"name": name, "fit_key": FIT_KEY, "model_domain": dict(flow.model_domain)},
        )
    model.save(path)
    return model


gps = {"forest_mpg_gp": get_gp("forest_mpg_gp", data)}
for group in holdouts:
    print(f"Training l1O GP for {group}...")
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
    for name in ("forest_mpg_gp", f"l1O {group}"):
        print(f"{group:12s} {name:16s} " + " ".join(f"{v:7.3f}" for v in transformed_rmse(gps[name], group)))

# %% [markdown]
# ## Predictions on the validation snapshots
#
# The cINN is evaluated in batches with common latent samples, exactly as in
# `Emulator_precision`. GP predictions are deterministic posterior means.

# %%
def emulator_input(snapshot):
    return {name: float(snapshot[name]) for name in inputs}


def cinn_predict(snapshots, chunk_size=128, model=None):
    flow = globals()["flow"] if model is None else model
    out = {name: [] for name in flow.output_labels}
    rows = [emulator_input(s) for s in snapshots]
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start : start + chunk_size]
        pred = flow.evaluate(
            chunk,
            Nrealizations=N_REALIZATIONS,
            seed=0,
            latent_indices=np.zeros(len(chunk), dtype=int),
        )
        for name in flow.output_labels:
            out[name].extend(np.asarray(pred[name]).reshape(-1))
    return [
        {name: float(out[name][i]) for name in flow.output_labels}
        for i in range(len(snapshots))
    ]


def gp_predict(model, snapshots):
    pred = model.evaluate([emulator_input(s) for s in snapshots])
    return [
        {name: float(pred[name][i]) for name in outputs} for i in range(len(snapshots))
    ]


test_snapshots = {
    "mpg_seed": archive.get_testing_data("mpg_seed"),
    "mpg_central": archive.get_testing_data("mpg_central"),
    "mpg_central_seed": archive.get_central_seed_average(),
}
for snapshots in test_snapshots.values():
    for s in snapshots:
        assert FIT_KEY in s, "run the fits for the testing simulations first"
central_snapshots = test_snapshots["mpg_central"]

COLOURS = {"cINN forest_mpg_cinn": "k", "forest_mpg_gp": "C0"}
for i, name in enumerate(n for n in gps if n != "forest_mpg_gp"):
    COLOURS[name] = f"C{i + 1}"


def predictions(snapshots, models):
    result = {"cINN forest_mpg_cinn": cinn_predict(snapshots)}
    for name in models:
        result[name] = gp_predict(gps[name], snapshots)
    return result


# emulators are evaluated for mpg_central only; the fits of all three test
# simulations are plotted for comparison
test_pred = {"mpg_central": predictions(test_snapshots["mpg_central"], list(gps))}
# the seed simulation is unseen by every model; for the central only the L1O GP
# that held it out is a fair test, the others are shown for reference
central_models = ["forest_mpg_gp"] + [n for n in gps if n == "l1O mpg_central"]
if "l1O mpg_central" not in gps:
    print("WARNING: add 'mpg_central' to HOLD_OUT for a fair central-simulation test")
central_pred = {
    label: test_pred["mpg_central"][label]
    for label in ["cINN forest_mpg_cinn", *central_models]
}

# %% [markdown]
# ## 1. Redshift evolution of the Arinyo parameters
#
# The emulators are evaluated for `mpg_central` only: the dashed line is the
# cINN `forest_mpg_cinn` and the solid line the GP, both trained on all
# simulations (the L1O GPs are not shown here). The points are the
# best-fitting `arinyo_fixp3d` parameters of `mpg_central`, `mpg_seed` and
# `mpg_central_seed`. Both emulators saw `mpg_central` in training, so their
# curves mainly show the regression error; the seed fits show how much the
# fitted parameters themselves vary between nearly identical simulations.

# %%
SIM_COLOURS = {"mpg_central": "C0", "mpg_seed": "C1", "mpg_central_seed": "C2"}
fig, axes = plt.subplots(4, 2, figsize=(10, 13), sharex=True)
for axis, name in zip(axes.flat, ArinyoFitter.PARAM_NAMES):
    for sim, colour in SIM_COLOURS.items():
        snapshots = test_snapshots[sim]
        z = np.asarray([s["z"] for s in snapshots])
        order = np.argsort(z)
        fit = np.asarray([s[FIT_KEY][name] for s in snapshots])
        axis.plot(z[order], fit[order], "o", color=colour, ms=5, mfc="none",
                  label=f"{sim}: fit" if name == "bias" else None)
    z = np.asarray([s["z"] for s in test_snapshots["mpg_central"]])
    order = np.argsort(z)
    for label, ls, text in (("forest_mpg_gp", "-", "forest_mpg_gp"), ("cINN forest_mpg_cinn", "--", "cINN forest_mpg_cinn")):
        values = np.asarray([row[name] for row in test_pred["mpg_central"][label]])
        axis.plot(z[order], values[order], color="k", ls=ls, lw=1.3,
                  label=f"mpg_central: {text}" if name == "bias" else None)
    axis.set_title(name)
    axis.grid(alpha=0.25)
for axis in axes[-1]:
    axis.set_xlabel("redshift")
for axis in axes[:, 0]:
    axis.set_ylabel("Arinyo parameter")
axes[0, 0].legend(fontsize=7)
fig.suptitle("Redshift evolution of the Arinyo parameters", y=1.0)
fig.tight_layout()

# %% [markdown]
# ## 2. P1D and 3. P3D precision for `mpg_central`, one panel per redshift
#
# Ratios of each emulator's projection to the projection of the saved
# `arinyo_fixp3d` fit, minus one. Colour = emulator. Grey lines are the actual
# simulation measurements divided by the same fit projection (so they show how
# well the Arinyo model describes the simulation), and, for P1D only, the red
# dashed line is LaCE's default smoothed fit (used to reprocess the training data
# before optimising the LaCE emulator). P3D is averaged in the four mu bins
# 0-0.25, 0.25-0.5, 0.5-0.75 and 0.75-1 (line style = mu bin) with the
# mode-weighted rebinning of the fit geometry, as in `Emulator_precision`.

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

# Simulation measurements and LaCE's default P1D smoothing, relative to the same
# Arinyo-fit projection as the emulators. The smoothing is the one LaCE applies
# to reprocess the training data before optimising its MPG emulator
# (CH24_mpgcen_gpr, the `lace_mpg` default); only P1D is smoothed.
from lace.emulator import emulator_manager
from lace.utils.smoothing_manager import apply_smoothing

lace_emulator = emulator_manager.set_emulator(emulator_label="CH24_mpgcen_gpr")
smoothed_snapshots = [dict(s) for s in central_snapshots]
apply_smoothing(lace_emulator, smoothed_snapshots, fprint=lambda *a, **k: None)


def measurement_ratios(snapshot, smoothed, fitter):
    """Measured P1D/P3D and LaCE-smoothed P1D, divided by the fit projection, minus one."""
    fitter.prepare_simulation(snapshot, is_mpg=True)
    fit_p3d, fit_p1d = fitter.predict(fitter.params_from_dict(snapshot[FIT_KEY]))
    k3d, _, fit_p3d, _ = rebin_P3D_Mpc_mode_weighted(
        fitter.data.k3d, fitter.data.mu3d, fit_p3d, fitter._mpg_k_mu_modes, n_mu_bins=n_mu_bins)
    _, _, meas_p3d, _ = rebin_P3D_Mpc_mode_weighted(
        fitter.data.k3d, fitter.data.mu3d, fitter.data.p3d, fitter._mpg_k_mu_modes, n_mu_bins=n_mu_bins)
    k1d = fitter.data.k1d
    smooth_p1d = np.interp(k1d, smoothed["k_Mpc"], smoothed["p1d_Mpc_smooth"])
    return {"k3d": k3d, "k1d": k1d, "P3D": meas_p3d / fit_p3d - 1,
            "P1D": fitter.data.p1d / fit_p1d - 1, "P1D_smooth": smooth_p1d / fit_p1d - 1}


central_reference = [measurement_ratios(snap, sm, fitter)
                     for snap, sm in zip(central_snapshots, smoothed_snapshots)]


def plot_by_redshift(kind):
    n_cols = 4
    n_rows = int(np.ceil(len(central_order) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.3 * n_rows), squeeze=False, sharey=True)
    tolerance = 0.1 if kind == "P3D" else 0.01
    for position, index in enumerate(central_order):
        axis = axes[divmod(position, n_cols)]
        axis.axhspan(-tolerance, tolerance, color="0.8", alpha=0.35, zorder=0)
        # simulation measurement (and, for P1D, LaCE's smoothed fit) vs the Arinyo fit
        reference = central_reference[index]
        if kind == "P1D":
            axis.plot(reference["k1d"], reference["P1D"], color="0.6", lw=1.2, label="measurement")
            axis.plot(reference["k1d"], reference["P1D_smooth"], color="C3", lw=1.2, ls="--",
                      label="LaCE smoothed")
        else:
            for mu in range(n_mu_bins):
                valid = np.isfinite(reference["k3d"][:, mu])
                axis.plot(reference["k3d"][valid, mu], reference["P3D"][valid, mu], color="0.6",
                          ls=MU_STYLES[mu], lw=1.0, label="measurement" if mu == 0 else None)
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
    if kind == "P3D":
        from matplotlib.lines import Line2D
        edges = central_ratios[next(iter(central_ratios))][0]["mu_edges"]
        handles = [Line2D([], [], color="k", ls=MU_STYLES[mu], lw=1.0,
                          label=rf"${edges[mu]:.2f} \leq \mu \leq {edges[mu + 1]:.2f}$") for mu in range(n_mu_bins)]
        axes[0, 1].legend(handles=handles, fontsize=7, title="mu bin (line style)", title_fontsize=7)
    suffix = ": average over four mu bins" if kind == "P3D" else ""
    fig.suptitle(f"mpg_central {kind} relative to the Arinyo fit{suffix}", y=1.0)
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

# %% [markdown]
# ## 4. Leave-one-out test on all 30 hypercube simulations
#
# For each hypercube simulation `mpg_i` (i = 0..29) we use the cINN and the GP
# trained **without** it (`data/emulator_models/l1O/forest_mpg_cinn_l1O_i` and
# `data/gp_emulators/l1O_mpg_i.pkl`) to predict that simulation's snapshots
# (all redshifts and optical-depth rescalings), and compare with the
# best-fitting `arinyo_fixp3d` model:
#
# - **parameters:** relative difference, emulator / fit - 1 (snapshots where the
#   fitted value is below 10% of that parameter's median |fit| are excluded, since
#   `q1` and `q2` pass near zero and their relative difference is then meaningless),
# - **P1D and P3D:** emulator projection divided by the projection of the fit,
#   minus one, on the fit geometry (hybrid P3D average; four mu bins).
#
# Finally we plot the mean and standard deviation over all simulations and
# rescalings. The standard deviation is the spread across snapshots, not an
# uncertainty on the mean. The cINN bundles record which simulation they
# excluded; this is checked below.

# %%
import json

L1O_SIMS = [f"mpg_{i}" for i in range(30)]
l1o = {name: {"z": [], "sim": [], "emu": [], "fit": [], "p1d": [], "p3d": []}
       for name in ("cINN", "GP")}
l1o_fitter = ArinyoFitter(**fit_settings)
k1d_l1o = k3d_l1o = mu_edges_l1o = None
for i, sim in enumerate(L1O_SIMS):
    snaps = [r for r in records if str(r["sim_label"]) == sim]
    assert snaps and all(FIT_KEY in r for r in snaps), sim
    # leave-one-out cINN: confirm the bundle excluded this simulation
    key = f"l1O/forest_mpg_cinn_l1O_{i}"
    manifest = ROOT / "data" / "emulator_models" / f"{key}_manifest.json"
    if manifest.is_file():
        excluded = json.loads(manifest.read_text())["training_provenance"]["excluded_simulation"]
        assert excluded == sim, (key, excluded)
    cinn = P3DEmulator(key=key, Nrealizations=N_REALIZATIONS)
    gp = get_gp(f"l1O_{sim}", select_groups(data, set(groups) - {sim}))
    assert sim not in gp.metadata["groups"]
    predicted = {"cINN": cinn_predict(snaps, model=cinn), "GP": gp_predict(gp, snaps)}
    del cinn, gp
    for name, params in predicted.items():
        for snap, par in zip(snaps, params):
            ratio = projected_ratio(snap, par, l1o_fitter)
            if k1d_l1o is None:
                k1d_l1o, k3d_l1o, mu_edges_l1o = ratio["k1d"], ratio["k3d"], ratio["mu_edges"]
            l1o[name]["z"].append(float(snap["z"]))
            l1o[name]["sim"].append(sim)
            l1o[name]["emu"].append([par[k] for k in ArinyoFitter.PARAM_NAMES])
            l1o[name]["fit"].append([snap[FIT_KEY][k] for k in ArinyoFitter.PARAM_NAMES])
            l1o[name]["p1d"].append(ratio["P1D"])
            l1o[name]["p3d"].append(ratio["P3D"])
    print(sim, len(snaps), "snapshots")
for name in l1o:
    for key in ("z", "emu", "fit", "p1d", "p3d"):
        l1o[name][key] = np.asarray(l1o[name][key])

# Relative parameter difference emulator/fit - 1. q1 and q2 pass through zero, so
# their relative difference is meaningless for snapshots whose fitted value is
# tiny; for each parameter we drop snapshots with |fit| < 10% of the median |fit|.
fit_all = l1o["GP"]["fit"]
small = np.abs(fit_all) < 0.1 * np.median(np.abs(fit_all), axis=0)
for name, res in l1o.items():
    res["params"] = np.where(small, np.nan, res["emu"] / res["fit"] - 1)
print("snapshots excluded from the relative parameter statistics:",
      dict(zip(ArinyoFitter.PARAM_NAMES, small.sum(axis=0))), "of", len(fit_all))

# %% [markdown]
# ### Arinyo parameters: relative difference to the fit, versus redshift

# %%
L1O_COLOURS = {"cINN": "k", "GP": "C0"}
redshifts_l1o = np.unique(np.round(l1o["GP"]["z"], 6))
fig, axes = plt.subplots(4, 2, figsize=(10, 13), sharex=True)
for p, (axis, pname) in enumerate(zip(axes.flat, ArinyoFitter.PARAM_NAMES)):
    for name, res in l1o.items():
        mean = np.asarray([np.nanmean(res["params"][np.isclose(res["z"], z), p]) for z in redshifts_l1o])
        std = np.asarray([np.nanstd(res["params"][np.isclose(res["z"], z), p]) for z in redshifts_l1o])
        axis.plot(redshifts_l1o, mean, color=L1O_COLOURS[name], label=name if p == 0 else None)
        axis.fill_between(redshifts_l1o, mean - std, mean + std, color=L1O_COLOURS[name], alpha=0.2)
    axis.axhline(0, color="k", ls=":", lw=.6)
    axis.set_title(pname)
    axis.grid(alpha=0.25)
for axis in axes[-1]:
    axis.set_xlabel("redshift")
for axis in axes[:, 0]:
    axis.set_ylabel("emulator / fit - 1")
axes[0, 0].legend(fontsize=8)
fig.suptitle("Leave-one-out: Arinyo parameters (relative difference), mean ± std over the 30 hypercube simulations", y=1.0)
fig.tight_layout()

# %% [markdown]
# ### P1D and P3D: mean and std over all simulations, rescalings and redshifts

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=False)
for name, res in l1o.items():
    mean, std = res["p1d"].mean(axis=0), res["p1d"].std(axis=0)
    axes[0].plot(k1d_l1o, mean, color=L1O_COLOURS[name], label=name)
    axes[0].fill_between(k1d_l1o, mean - std, mean + std, color=L1O_COLOURS[name], alpha=0.2)
for column, name in zip(axes[1:], l1o):
    p3 = l1o[name]["p3d"]
    mean, std = np.nanmean(p3, axis=0), np.nanstd(p3, axis=0)
    for mu in range(n_mu_bins):
        valid = np.isfinite(k3d_l1o[:, mu])
        label = rf"${mu_edges_l1o[mu]:.2f} \leq \mu < {mu_edges_l1o[mu + 1]:.2f}$"
        column.plot(k3d_l1o[valid, mu], mean[valid, mu], color=f"C{mu}", label=label)
        column.fill_between(k3d_l1o[valid, mu], (mean - std)[valid, mu], (mean + std)[valid, mu],
                            color=f"C{mu}", alpha=0.2)
    column.set_title(f"P3D: {name}")
    column.legend(fontsize=7)
axes[0].set_title("P1D")
axes[0].legend()
axes[0].set_ylabel(r"$P^\mathrm{emu}/P^\mathrm{fit}-1$")
axes[0].set_xlabel(r"$k_\parallel\,[\mathrm{Mpc}^{-1}]$")
for axis in axes[1:]:
    axis.set_xlabel(r"$k\,[\mathrm{Mpc}^{-1}]$")
for axis, tolerance, kmax in zip(axes, (0.01, 0.1, 0.1), (fit_settings["kmax_1d"],) + (fit_settings["kmax_3d"],) * 2):
    axis.axhspan(-tolerance, tolerance, color="0.8", alpha=0.35, zorder=0)
    axis.axhline(0, color="k", ls=":", lw=.6)
    axis.axvline(kmax, color="k", ls="--", lw=.6)
    axis.set_xscale("log")
    axis.grid(alpha=0.25)
fig.suptitle("Leave-one-out: mean ± std over the 30 hypercube simulations", y=1.02)
fig.tight_layout()

# %% [markdown]
# ### P1D and P3D resolved by redshift

# %%
def l1o_by_redshift(kind):
    n_cols = 4
    n_rows = int(np.ceil(len(redshifts_l1o) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.3 * n_rows), squeeze=False, sharey=True)
    for position, z in enumerate(redshifts_l1o):
        axis = axes[divmod(position, n_cols)]
        axis.axhspan(-(0.01 if kind == "P1D" else 0.1), 0.01 if kind == "P1D" else 0.1,
                     color="0.8", alpha=0.35, zorder=0)
        for name, res in l1o.items():
            sel = np.isclose(res["z"], z)
            if kind == "P1D":
                mean, std = res["p1d"][sel].mean(axis=0), res["p1d"][sel].std(axis=0)
                axis.plot(k1d_l1o, mean, color=L1O_COLOURS[name], label=name)
                axis.fill_between(k1d_l1o, mean - std, mean + std, color=L1O_COLOURS[name], alpha=0.2)
            else:
                mean = np.nanmean(res["p3d"][sel], axis=0)
                for mu in range(n_mu_bins):
                    valid = np.isfinite(k3d_l1o[:, mu])
                    axis.plot(k3d_l1o[valid, mu], mean[valid, mu], color=L1O_COLOURS[name],
                              ls=MU_STYLES[mu], lw=1.0, label=name if mu == 0 else None)
        axis.axhline(0, color="k", ls=":", lw=.6)
        axis.axvline(fit_settings["kmax_1d" if kind == "P1D" else "kmax_3d"], color="k", ls="--", lw=.6)
        axis.set_xscale("log")
        axis.set_title(f"z = {z:.2f}")
        axis.grid(alpha=0.25)
        if position == 0:
            axis.legend(fontsize=7)
    for position in range(len(redshifts_l1o), n_rows * n_cols):
        axes[divmod(position, n_cols)].set_visible(False)
    suffix = " (mean; line style = mu bin)" if kind == "P3D" else " (mean ± std)"
    fig.suptitle(f"Leave-one-out {kind} deviation from the fit by redshift{suffix}", y=1.0)
    fig.tight_layout()


l1o_by_redshift("P1D")

# %%
l1o_by_redshift("P3D")

# %% [markdown]
# ### Summary
#
# RMS over all snapshots of the leave-one-out deviations from the fit.

# %%
for name, res in l1o.items():
    print(f"{name:5s} P1D RMS {np.sqrt(np.mean(res['p1d']**2)):.4f}  "
          f"P3D RMS {np.sqrt(np.nanmean(res['p3d']**2)):.4f}  "
          "params RMS " + " ".join(f"{v:.3f}" for v in np.sqrt(np.nanmean(res['params']**2, axis=0))))

# %%
