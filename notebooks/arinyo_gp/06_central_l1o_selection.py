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
# # 6. GP setup selection on held-out `mpg_central`
#
# Every candidate GP is trained on all training simulations **except**
# `mpg_central` and evaluated on the `mpg_central` snapshots, comparing with
# their best-fitting `arinyo_fixp3d` parameters:
#
# - the Arinyo parameters (relative difference, emulator / fit - 1),
# - P3D, averaged in four mu bins with the mode-weighted rebinning of the fit
#   geometry (as in notebook 4), required within **3%**,
# - P1D, required within **1%**,
#
# both P3D and P1D as emulator projection / fit projection - 1 within the fitted
# scales (kmax_3d = 4.5, kmax_1d = 6 Mpc^-1).
#
# Candidates (all with one length scale per input dimension):
#
# - `shared`: one GP for the 8 transformed Arinyo parameters, as in production
#   (`forest_mpg_gp` uses nu=2.5, alpha=1e-2), with a fixed nugget alpha;
# - `shared_white`: one GP with fitted amplitude and fitted noise level
#   (ConstantKernel * Matern + WhiteKernel) instead of a fixed alpha;
# - `per_output`: one GP per Arinyo parameter, each with its own amplitude,
#   length scales and noise level.
#
# The input/output transformation is the frozen `forest_mpg_cinn` one, as for
# `forest_mpg_gp`. Trained candidates are cached in `artifacts/central_l1o/`.

# %%
from pathlib import Path
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
import forestflow
from forestflow.archive.gadget_archive import GadgetArchive3D
from forestflow.emulator import load_emulator
from forestflow.emulator.arinyo_gp import ArinyoGPEmulator, prepare_arinyo_gp_training_data, select_groups
from forestflow.model_fits import ArinyoFitter
from forestflow.statistics.rebin_p3d import rebin_P3D_Mpc_mode_weighted
from lace.emulator.gp_emulator_multi import optimizer as lace_gp_optimizer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

ROOT = Path(forestflow.__file__).resolve().parent.parent
CACHE = ROOT / "notebooks" / "arinyo_gp" / "artifacts" / "central_l1o"
CACHE.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Configuration

# %%
HELD_OUT = "mpg_central"
FIT_KEY = "arinyo_fixp3d"
RETRAIN = False
P1D_TOL, P3D_TOL = 0.01, 0.03
fit_settings = dict(kmin_3d=0.01, kmax_3d=4.5, kmin_1d=0.01, kmax_1d=6.0)
n_mu_bins = 4

CANDIDATES = {
    "shared nu2.5 a1e-2": dict(kind="shared", nu=2.5, alpha=1e-2),  # production setup
    "shared nu1.5 a1e-2": dict(kind="shared", nu=1.5, alpha=1e-2),
    "shared nu0.5 a1e-6": dict(kind="shared", nu=0.5, alpha=1e-6),  # LaCE-like
    "shared_white nu2.5": dict(kind="shared_white", nu=2.5),
    "shared_white nu1.5": dict(kind="shared_white", nu=1.5),
    "per_output nu2.5": dict(kind="per_output", nu=2.5),
    "per_output nu1.5": dict(kind="per_output", nu=1.5),
    "per_output nu0.5": dict(kind="per_output", nu=0.5),
}

# %% [markdown]
# ## Training data and held-out snapshots

# %%
reference = load_emulator("forest_mpg_gp")  # frozen transform and labels
transform = reference.transform
inputs, outputs = reference.input_labels, reference.output_labels

archive = GadgetArchive3D(postproc="Cabayol23_fixp3d", addcentral=True)
records = [r for r in archive.training_data if r["z"] <= 4.6]
data = prepare_arinyo_gp_training_data(records, transform, inputs, outputs, fit_key=FIT_KEY)
groups = sorted(set(data.group_ids))
train = select_groups(data, set(groups) - {HELD_OUT})
print(len(data.group_ids), "rows;", train.inputs.shape[0], "for training without", HELD_OUT)

central = archive.get_testing_data(HELD_OUT)
central = sorted(central, key=lambda s: s["z"])
z_central = np.array([s["z"] for s in central])
print(len(central), "held-out snapshots at z =", z_central)


# %% [markdown]
# ## Models

# %%
def matern(nu, ndim):
    return Matern(length_scale=np.ones(ndim), nu=nu, length_scale_bounds=(1e-3, 1e3))


def fit_gp(kernel, x, y, alpha):
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=False,
        optimizer=lace_gp_optimizer,
        random_state=32,
        n_restarts_optimizer=0,
    ).fit(x, y)


class CandidateGP:
    """One shared GP or one GP per output, in transformed ForestFlow space."""

    def __init__(self, kind, nu, alpha=None):
        self.kind, self.nu, self.alpha = kind, nu, alpha

    def fit(self, x, y):
        ndim = x.shape[1]
        start = time.time()
        if self.kind == "shared":
            self.gps = [fit_gp(matern(self.nu, ndim), x, y, self.alpha)]
        elif self.kind == "shared_white":
            kernel = ConstantKernel(1.0, (1e-2, 1e2)) * matern(
                self.nu, ndim
            ) + WhiteKernel(1e-3, (1e-8, 1.0))
            self.gps = [fit_gp(kernel, x, y, 1e-10)]
        elif self.kind == "per_output":
            self.gps = []
            for j in range(y.shape[1]):
                kernel = ConstantKernel(1.0, (1e-2, 1e2)) * matern(
                    self.nu, ndim
                ) + WhiteKernel(1e-3, (1e-8, 1.0))
                self.gps.append(fit_gp(kernel, x, y[:, j], 1e-10))
        self.train_time = time.time() - start
        return self

    def predict(self, x):
        if self.kind == "per_output":
            return np.column_stack([gp.predict(x) for gp in self.gps])
        return self.gps[0].predict(x)

    def kernels(self):
        return [str(gp.kernel_) for gp in self.gps]

    def evaluate(self, snapshots):
        """Physical Arinyo parameters for archive snapshots."""
        x = np.column_stack(
            [
                transform.transf_stand(
                    {k: np.array([s[k] for s in snapshots]) for k in inputs},
                    type_stand="input",
                    direct=True,
                )[k]
                for k in inputs
            ]
        )
        pred = self.predict(x)
        phys = transform.transf_stand(
            {k: pred[:, j] for j, k in enumerate(outputs)},
            type_stand="output",
            direct=False,
        )
        return [{k: float(phys[k][i]) for k in outputs} for i in range(len(snapshots))]


def get_candidate(name, setup):
    path = CACHE / (name.replace(" ", "_") + ".pkl")
    if path.is_file() and not RETRAIN:
        with open(path, "rb") as f:
            return pickle.load(f)
    print("training", name, "...", flush=True)
    model = CandidateGP(**setup).fit(train.inputs, train.targets)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  {model.train_time:.0f} s", model.kernels()[0], flush=True)
    return model


models = {name: get_candidate(name, setup) for name, setup in CANDIDATES.items()}

# %% [markdown]
# Sanity check: the `shared nu2.5 a1e-2` candidate is the production setup, so it
# must reproduce the stored leave-one-out GP `l1O_mpg_central`.

# %%
stored = ArinyoGPEmulator.load(ROOT / "data" / "gp_emulators" / "l1O_mpg_central.pkl")
rows = [{k: float(s[k]) for k in inputs} for s in central]
a = stored.evaluate(rows)
b = models["shared nu2.5 a1e-2"].evaluate(central)
print("max |difference| with stored l1O_mpg_central:",
      max(np.max(np.abs(np.asarray(a[k]) - np.array([r[k] for r in b]))) for k in outputs))

# %% [markdown]
# ## Projection to P3D and P1D

# %%
fitter = ArinyoFitter(**fit_settings)


def projected_ratio(snapshot, emulated):
    """Emulator / Arinyo-fit P3D (four mu bins) and P1D, minus one."""
    fitter.prepare_simulation(snapshot, is_mpg=True)
    fit_p3d, fit_p1d = fitter.predict(fitter.params_from_dict(snapshot[FIT_KEY]))
    emu_p3d, emu_p1d = fitter.predict(fitter.params_from_dict(emulated))
    k3d, _, fit_p3d, mu_edges = rebin_P3D_Mpc_mode_weighted(
        fitter.data.k3d, fitter.data.mu3d, fit_p3d, fitter._mpg_k_mu_modes, n_mu_bins=n_mu_bins)
    _, _, emu_p3d, _ = rebin_P3D_Mpc_mode_weighted(
        fitter.data.k3d, fitter.data.mu3d, emu_p3d, fitter._mpg_k_mu_modes, n_mu_bins=n_mu_bins)
    return {"k3d": k3d, "mu_edges": mu_edges, "P3D": emu_p3d / fit_p3d - 1,
            "k1d": fitter.data.k1d.copy(), "P1D": emu_p1d / fit_p1d - 1}


results = {}
for name, model in models.items():
    params = model.evaluate(central)
    ratios = [projected_ratio(s, p) for s, p in zip(central, params)]
    par_rel = np.array([[p[k] / s[FIT_KEY][k] - 1 for k in ArinyoFitter.PARAM_NAMES]
                        for s, p in zip(central, params)])
    results[name] = {
        "ratios": ratios,
        "par_rel": par_rel,
        "p1d_max": np.array([np.max(np.abs(r["P1D"])) for r in ratios]),
        "p3d_max": np.array([np.nanmax(np.abs(r["P3D"])) for r in ratios]),
    }

# %% [markdown]
# ## Summary
#
# Maximum |emulator/fit - 1| over scales, per redshift. A redshift passes when
# P1D <= 1% and P3D <= 3%.

# %%
print(f"{'candidate':22s} {'pass':>5s} {'P1D max (median, worst)':>25s} {'P3D max (median, worst)':>25s}  train time")
for name, res in results.items():
    ok = (res["p1d_max"] <= P1D_TOL) & (res["p3d_max"] <= P3D_TOL)
    print(f"{name:22s} {ok.sum():2d}/{len(ok):2d} "
          f"{np.median(res['p1d_max']):12.4f} {res['p1d_max'].max():12.4f} "
          f"{np.median(res['p3d_max']):12.4f} {res['p3d_max'].max():12.4f}  {models[name].train_time:6.0f} s")

# %%
print("P1D max |emu/fit - 1| per redshift")
print(f"{'candidate':22s} " + " ".join(f"{z:6.2f}" for z in z_central))
for name, res in results.items():
    print(f"{name:22s} " + " ".join(f"{v:6.3f}" for v in res["p1d_max"]))
print()
print("P3D max |emu/fit - 1| per redshift")
for name, res in results.items():
    print(f"{name:22s} " + " ".join(f"{v:6.3f}" for v in res["p3d_max"]))

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
for i, (name, res) in enumerate(results.items()):
    ls = "-" if "per_output" in name else ("--" if "white" in name else ":")
    axes[0].plot(z_central, res["p1d_max"], ls, marker="o", ms=3, color=f"C{i}", label=name)
    axes[1].plot(z_central, res["p3d_max"], ls, marker="o", ms=3, color=f"C{i}")
for axis, tol, kind in zip(axes, (P1D_TOL, P3D_TOL), ("P1D", "P3D")):
    axis.axhline(tol, color="k", lw=0.8)
    axis.set_yscale("log")
    axis.set_xlabel("z")
    axis.set_ylabel(f"max |{kind} emu/fit - 1|")
    axis.grid(alpha=0.25)
axes[0].legend(fontsize=7)
fig.suptitle(f"Held-out {HELD_OUT}: maximum deviation from the Arinyo fit")
fig.tight_layout()

# %% [markdown]
# ## Arinyo parameters
#
# Relative difference with the fit, per parameter (columns) and candidate.

# %%
fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True)
for j, (axis, par) in enumerate(zip(axes.flat, ArinyoFitter.PARAM_NAMES)):
    for i, (name, res) in enumerate(results.items()):
        ls = "-" if "per_output" in name else ("--" if "white" in name else ":")
        axis.plot(z_central, res["par_rel"][:, j], ls, color=f"C{i}", label=name)
    axis.axhline(0, color="k", lw=0.5)
    axis.set_title(par)
    axis.grid(alpha=0.25)
for axis in axes[-1]:
    axis.set_xlabel("z")
axes[0, 0].legend(fontsize=6)
fig.tight_layout()

# %% [markdown]
# ## P1D and P3D per redshift for selected candidates

# %%
SHOW = list(results)


def plot_by_redshift(kind, show=SHOW):
    n_cols = 4
    n_rows = int(np.ceil(len(central) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.3 * n_rows), squeeze=False, sharey=True)
    tol = P3D_TOL if kind == "P3D" else P1D_TOL
    for iz in range(len(central)):
        axis = axes[divmod(iz, n_cols)]
        axis.axhspan(-tol, tol, color="0.8", alpha=0.35, zorder=0)
        for i, name in enumerate(results):
            if name not in show:
                continue
            r = results[name]["ratios"][iz]
            if kind == "P1D":
                axis.plot(r["k1d"], r["P1D"], color=f"C{i}", label=name)
            else:
                # average over the mu bins, to keep the panel readable
                valid = np.isfinite(r["P3D"]).all(axis=1)
                axis.plot(r["k3d"][valid, 0], np.mean(r["P3D"][valid], axis=1), color=f"C{i}", label=name)
        axis.axhline(0, color="k", ls=":", lw=0.6)
        axis.set_xscale("log")
        axis.set_ylim(-3 * tol, 3 * tol)
        axis.set_title(f"z = {z_central[iz]:.2f}")
        axis.grid(alpha=0.25)
    for iz in range(len(central), n_rows * n_cols):
        axes[divmod(iz, n_cols)].set_visible(False)
    axes[0, 0].legend(fontsize=6)
    suffix = " (mean over the four mu bins)" if kind == "P3D" else ""
    fig.suptitle(f"Held-out {HELD_OUT}: {kind} emulator / fit - 1{suffix}", y=1.0)
    fig.tight_layout()


plot_by_redshift("P1D")

# %%
plot_by_redshift("P3D")

# %% [markdown]
# ## Fitted kernels

# %%
for name, model in models.items():
    print(name)
    for label, kernel in zip(outputs if len(model.gps) > 1 else ["all"], model.kernels()):
        print(f"   {label:9s} {kernel}")

# %% [markdown]
# ## Refinements around `shared_white nu2.5`
#
# `shared_white nu2.5` is within 1% in P1D on large scales at all redshifts. It
# only departs from the fit at small scales and low z, beyond the scales
# measured by DESI DR1 (k_max = 1.76 Mpc^-1 at z=2.2, rising to 3.55 at z=4.2;
# converted from s/km with Planck18). We try nearby setups, keeping the shared
# kernel:
#
# - more optimizer restarts (is the default optimum the best one?),
# - an RBF kernel, and a two-scale kernel (Matern 2.5 + Matern 0.5),
# - a fixed small nugget with free amplitude, instead of a fitted noise level,
# - output weights: multiplying a transformed output by w before training
#   lowers its effective noise relative to the shared WhiteKernel by w^2, so the
#   GP follows that output more closely. The small-scale drop comes mostly from
#   `kp` (and `kvav`, `av`, `bv`), so we up-weight those.

# %%
from sklearn.gaussian_process.kernels import RBF

# DESI DR1 P1D maximum wavenumber (Mpc^-1) per redshift, from DESIY1_QMLE3
Z_DESI = np.array([2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2])
KMAX_DESI = np.array([1.76, 1.91, 2.07, 2.23, 2.44, 2.61, 2.79, 2.98, 3.19, 3.32, 3.55])
kmax_desi = np.interp(z_central, Z_DESI, KMAX_DESI)  # constant beyond the ends

SMALL_SCALE = ("kvav", "av", "bv", "kp")


class RefinedGP(CandidateGP):
    """Shared-kernel GP with optional output weights and alternative kernels."""

    def __init__(self, base="matern", nu=2.5, restarts=0, alpha=None, weights=None):
        self.kind, self.base, self.nu, self.restarts = "refined", base, nu, restarts
        self.alpha = alpha
        self.weights = np.array([(weights or {}).get(k, 1.0) for k in outputs])

    def _kernel(self, ndim):
        amp = lambda: ConstantKernel(1.0, (1e-2, 1e3))
        if self.base == "matern":
            kernel = amp() * matern(self.nu, ndim)
        elif self.base == "rbf":
            kernel = amp() * RBF(np.ones(ndim), length_scale_bounds=(1e-3, 1e3))
        elif self.base == "matern+exp":
            kernel = amp() * matern(2.5, ndim) + amp() * matern(0.5, ndim)
        if self.alpha is None:
            kernel = kernel + WhiteKernel(1e-3, (1e-8, 1.0))
        return kernel

    def fit(self, x, y):
        start = time.time()
        gp = GaussianProcessRegressor(
            kernel=self._kernel(x.shape[1]),
            alpha=1e-10 if self.alpha is None else self.alpha,
            normalize_y=False, optimizer=lace_gp_optimizer,
            random_state=32, n_restarts_optimizer=self.restarts,
        )
        self.gps = [gp.fit(x, y * self.weights)]
        self.train_time = time.time() - start
        return self

    def predict(self, x):
        return self.gps[0].predict(x) / self.weights


REFINED = {
    "R0 white nu2.5": None,  # the cached shared_white nu2.5
    "R1 white nu2.5 restarts3": dict(restarts=3),
    "R2 white rbf": dict(base="rbf"),
    "R3 white matern2.5+0.5": dict(base="matern+exp"),
    "R4 nu2.5 a1e-4 (no white)": dict(alpha=1e-4),
    "R5 white nu2.5 w_kp=3": dict(weights={"kp": 3.0}),
    "R6 white nu2.5 w_kp=10": dict(weights={"kp": 10.0}),
    "R7 white nu2.5 w_small=3": dict(weights={k: 3.0 for k in SMALL_SCALE}),
}


def get_refined(name, setup):
    if setup is None:
        return models["shared_white nu2.5"]
    path = CACHE / (name.split()[0] + ".pkl")
    if path.is_file() and not RETRAIN:
        with open(path, "rb") as f:
            return pickle.load(f)
    print("training", name, "...", flush=True)
    model = RefinedGP(**setup).fit(train.inputs, train.targets)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  {model.train_time:.0f} s", model.kernels()[0], flush=True)
    return model


refined = {name: get_refined(name, setup) for name, setup in REFINED.items()}
refined["production (shared nu2.5 a1e-2)"] = models["shared nu2.5 a1e-2"]


def scores(model):
    params = model.evaluate(central)
    ratios = [projected_ratio(s, p) for s, p in zip(central, params)]
    in_desi = [np.abs(r["P1D"][r["k1d"] <= km]) for r, km in zip(ratios, kmax_desi)]
    return {
        "ratios": ratios,
        "par_rel": np.array([[p[k] / s[FIT_KEY][k] - 1 for k in ArinyoFitter.PARAM_NAMES]
                             for s, p in zip(central, params)]),
        "p1d_desi": np.array([np.max(a) for a in in_desi]),
        "p1d_max": np.array([np.max(np.abs(r["P1D"])) for r in ratios]),
        "p3d_max": np.array([np.nanmax(np.abs(r["P3D"])) for r in ratios]),
    }


results_ref = {name: scores(model) for name, model in refined.items()}

# %% [markdown]
# Maximum |P1D emu/fit - 1| up to the DESI k_max of each redshift, up to the
# fitted k_max = 6 Mpc^-1, and maximum |P3D emu/fit - 1|.

# %%
for key, title in (("p1d_desi", "P1D, k <= DESI k_max"), ("p1d_max", "P1D, k <= 6"),
                   ("p3d_max", "P3D, k <= 4.5")):
    print(title)
    print(f"{'':34s} " + " ".join(f"{z:6.2f}" for z in z_central) + "   pass")
    tol = P3D_TOL if key == "p3d_max" else P1D_TOL
    for name, res in results_ref.items():
        print(f"{name:34s} " + " ".join(f"{v:6.3f}" for v in res[key])
              + f"   {np.sum(res[key] <= tol):2d}/{len(z_central)}")
    print()

# %%
for name, model in refined.items():
    print(f"{name:34s} {model.kernels()[0]}")

# %%
results_backup = results
results = results_ref
plot_by_redshift("P1D", show=list(results_ref))
for axis, km in zip(plt.gcf().axes, kmax_desi):
    axis.axvline(km, color="k", ls="--", lw=0.8)
results = results_backup

# %%
fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True)
for j, (axis, par) in enumerate(zip(axes.flat, ArinyoFitter.PARAM_NAMES)):
    for i, (name, res) in enumerate(results_ref.items()):
        axis.plot(z_central, res["par_rel"][:, j], color=f"C{i}", label=name)
    axis.axhline(0, color="k", lw=0.5)
    axis.set_title(par)
    axis.grid(alpha=0.25)
for axis in axes[-1]:
    axis.set_xlabel("z")
axes[0, 0].legend(fontsize=6)
fig.suptitle(f"Held-out {HELD_OUT}: Arinyo parameters, emulator / fit - 1")
fig.tight_layout()

# %% [markdown]
# ## Two mean-flux experts, as in `lace_mpg`
#
# The training rows are split at percentiles of the (transformed) mean flux,
# 0-60% and 40-100%, i.e. 10% overlap around the median, as LaCE does for two
# experts. Each expert is an independent GP with its own hyperparameters, so
# low z (high mF) and high z (low mF) can have different smoothness and noise.
# Predictions use the expert with the nearest mF centre; in the overlap,
#
# - `lace average`: the plain average of both experts (as LaCE; the weight
#   jumps at the overlap edges),
# - `taper`: a weight varying linearly across the overlap (continuous).

# %%
class MeanFluxExperts(CandidateGP):
    """Two GPs split in mean flux, combined as in LaCE or with a linear taper."""

    def __init__(self, combine="taper", **expert_setup):
        self.kind, self.combine, self.expert_setup = "experts", combine, expert_setup
        self.ind_mF = list(inputs).index("mF")

    def fit(self, x, y):
        start = time.time()
        mF = x[:, self.ind_mF]
        # LaCE: percentiles [0, 100] -> bins [0, 50] and [50, 100], widened by 10%
        self.bot = np.percentile(mF, [0, 40])
        self.top = np.percentile(mF, [60, 100])
        self.cen = 0.5 * (self.bot + self.top)
        self.experts = []
        for i in range(2):
            mask = (mF >= self.bot[i]) & (mF <= self.top[i])
            self.experts.append(RefinedGP(**self.expert_setup).fit(x[mask], y[mask]))
        self.gps = [e.gps[0] for e in self.experts]
        self.train_time = time.time() - start
        return self

    def _weight_high(self, mF):
        """Weight of the high-mF expert."""
        lo, hi = self.bot[1], self.top[0]  # overlap region
        if self.combine == "taper":
            return np.clip((mF - lo) / (hi - lo), 0, 1)
        # LaCE: nearest-centre expert, plain average where the other overlaps
        nearest_high = np.abs(mF - self.cen[1]) < np.abs(mF - self.cen[0])
        in_overlap = (mF >= lo) & (mF <= hi)
        return np.where(in_overlap, 0.5, nearest_high.astype(float))

    def predict(self, x):
        w = self._weight_high(x[:, self.ind_mF])[:, None]
        return (1 - w) * self.experts[0].predict(x) + w * self.experts[1].predict(x)

    def kernels(self):
        return [f"low mF: {self.experts[0].kernels()[0]} | high mF: {self.experts[1].kernels()[0]}"]


EXPERTS = {
    "R8 2 experts white nu2.5 lace average": dict(combine="lace"),
    "R9 2 experts white nu2.5 taper": dict(combine="taper"),
    "R10 2 experts nu2.5 a1e-4 taper": dict(combine="taper", alpha=1e-4),
}


def get_experts(name, setup):
    path = CACHE / (name.split()[0] + ".pkl")
    if path.is_file() and not RETRAIN:
        with open(path, "rb") as f:
            return pickle.load(f)
    print("training", name, "...", flush=True)
    model = MeanFluxExperts(**setup).fit(train.inputs, train.targets)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  {model.train_time:.0f} s", model.kernels()[0], flush=True)
    return model


experts = {name: get_experts(name, setup) for name, setup in EXPERTS.items()}
mF_central = np.array([
    transform.transf_stand({"mF": np.array([s["mF"]])}, type_stand="input", direct=True)["mF"][0]
    for s in central])
e = experts["R9 2 experts white nu2.5 taper"]
print("overlap in transformed mF:", np.round([e.bot[1], e.top[0]], 3))
print("mpg_central weight of the high-mF expert per z:",
      " ".join(f"{z:.2f}:{w:.2f}" for z, w in zip(z_central, e._weight_high(mF_central))))

for name, model in experts.items():
    results_ref[name] = scores(model)

# %%
compare = ["R0 white nu2.5", "R4 nu2.5 a1e-4 (no white)", "production (shared nu2.5 a1e-2)", *EXPERTS]
for key, title in (("p1d_desi", "P1D, k <= DESI k_max"), ("p1d_max", "P1D, k <= 6"),
                   ("p3d_max", "P3D, k <= 4.5")):
    print(title)
    print(f"{'':40s} " + " ".join(f"{z:6.2f}" for z in z_central) + "   pass")
    tol = P3D_TOL if key == "p3d_max" else P1D_TOL
    for name in compare:
        res = results_ref[name]
        print(f"{name:40s} " + " ".join(f"{v:6.3f}" for v in res[key])
              + f"   {np.sum(res[key] <= tol):2d}/{len(z_central)}")
    print()

# %%
for name, model in experts.items():
    print(name, "\n   ", model.kernels()[0])

# %%
results_backup = results
results = {name: results_ref[name] for name in compare}
plot_by_redshift("P1D", show=compare)
for axis, km in zip(plt.gcf().axes, kmax_desi):
    axis.axvline(km, color="k", ls="--", lw=0.8)
plot_by_redshift("P3D", show=compare)
results = results_backup

# %% [markdown]
# ## Robustness: leave-one-out on random hypercube simulations
#
# `mpg_central` sits at the centre of the hypercube, where interpolation is
# easiest. Here we compare, for a few random hypercube simulations `mpg_N`,
#
# - `production`: the stored leave-one-out `forest_mpg_gp` GP
#   (`data/gp_emulators/l1O_mpg_N.pkl`, nu=2.5, alpha=1e-2), and
# - `R4`: amplitude * Matern(nu=2.5) with a fixed nugget alpha=1e-4,
#
# both trained on all simulations except `mpg_N` (`mpg_central` included), and
# evaluated on the `mpg_N` snapshots without mean-flux rescaling, against their
# own Arinyo fits.

# %%
N_SIMS = 5
rng = np.random.default_rng(2026)
hyper = [g for g in groups if g.startswith("mpg_") and g[4:].isdigit()]
sims_l1o = sorted(rng.choice(hyper, N_SIMS, replace=False), key=lambda s: int(s[4:]))
print("held-out simulations:", sims_l1o)


def snapshots_of(sim):
    snaps = [r for r in records if str(r["sim_label"]) == sim and r.get("ind_rescaling", 0) == 0]
    assert snaps and all(FIT_KEY in s for s in snaps), sim
    return sorted(snaps, key=lambda s: s["z"])


class StoredGP:
    """Adapter giving a stored ArinyoGPEmulator the candidates' evaluate API."""

    def __init__(self, path):
        self.gp = ArinyoGPEmulator.load(path)

    def evaluate(self, snapshots):
        pred = self.gp.evaluate([{k: float(s[k]) for k in inputs} for s in snapshots])
        return [{k: float(pred[k][i]) for k in outputs} for i in range(len(snapshots))]


def get_r4_l1o(sim):
    path = CACHE / f"R4_l1O_{sim}.pkl"
    if path.is_file() and not RETRAIN:
        with open(path, "rb") as f:
            return pickle.load(f)
    print("training R4 without", sim, "...", flush=True)
    model = RefinedGP(alpha=1e-4).fit(*(lambda d: (d.inputs, d.targets))(
        select_groups(data, set(groups) - {sim})))
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  {model.train_time:.0f} s", model.kernels()[0], flush=True)
    return model


def scores_snaps(model, snaps):
    z = np.array([s["z"] for s in snaps])
    kmax = np.interp(z, Z_DESI, KMAX_DESI)
    params = model.evaluate(snaps)
    ratios = [projected_ratio(s, p) for s, p in zip(snaps, params)]
    return {
        "z": z,
        "ratios": ratios,
        "par_rel": np.array([[p[k] / s[FIT_KEY][k] - 1 for k in ArinyoFitter.PARAM_NAMES]
                             for s, p in zip(snaps, params)]),
        "p1d_desi": np.array([np.max(np.abs(r["P1D"][r["k1d"] <= km])) for r, km in zip(ratios, kmax)]),
        "p1d_max": np.array([np.max(np.abs(r["P1D"])) for r in ratios]),
        "p3d_max": np.array([np.nanmax(np.abs(r["P3D"])) for r in ratios]),
    }


l1o_res = {}
for sim in sims_l1o:
    snaps = snapshots_of(sim)
    l1o_res[sim] = {
        "production": scores_snaps(StoredGP(ROOT / "data" / "gp_emulators" / f"l1O_{sim}.pkl"), snaps),
        "R4": scores_snaps(get_r4_l1o(sim), snaps),
    }

# %%
for key, title, tol in (("p1d_desi", "P1D, k <= DESI k_max", P1D_TOL),
                        ("p1d_max", "P1D, k <= 6", P1D_TOL),
                        ("p3d_max", "P3D, k <= 4.5", P3D_TOL)):
    print(title)
    for sim, res in l1o_res.items():
        print(f"  {sim:8s} {'z':10s} " + " ".join(f"{z:6.2f}" for z in res["R4"]["z"]))
        for name, r in res.items():
            print(f"  {'':8s} {name:10s} " + " ".join(f"{v:6.3f}" for v in r[key])
                  + f"   pass {np.sum(r[key] <= tol):2d}/{len(r[key])}")
    print()

# %%
print(f"{'':12s} {'P1D DESI':>16s} {'P1D k<=6':>16s} {'P3D':>16s}   (median / worst over simulations and z)")
for name in ("production", "R4"):
    row = []
    for key in ("p1d_desi", "p1d_max", "p3d_max"):
        v = np.concatenate([res[name][key] for res in l1o_res.values()])
        row.append(f"{np.median(v):7.4f} {v.max():7.4f}")
    passed = sum(np.sum((res[name]["p1d_desi"] <= P1D_TOL) & (res[name]["p3d_max"] <= P3D_TOL))
                 for res in l1o_res.values())
    total = sum(len(res[name]["z"]) for res in l1o_res.values())
    print(f"{name:12s} " + " ".join(f"{c:>16s}" for c in row) + f"   pass (P1D DESI & P3D) {passed}/{total}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
for i, (sim, res) in enumerate(l1o_res.items()):
    for name, ls in (("production", ":"), ("R4", "-")):
        for axis, key in zip(axes, ("p1d_desi", "p1d_max", "p3d_max")):
            axis.plot(res[name]["z"], res[name][key], ls, marker="o", ms=3, color=f"C{i}",
                      label=f"{sim} {name}")
for axis, key, tol in zip(axes, ("P1D, k <= DESI k_max", "P1D, k <= 6", "P3D, k <= 4.5"),
                          (P1D_TOL, P1D_TOL, P3D_TOL)):
    axis.axhline(tol, color="k", lw=0.8)
    axis.set_yscale("log")
    axis.set_xlabel("z")
    axis.set_title(f"max |emu/fit - 1|, {key}")
    axis.grid(alpha=0.25)
axes[0].legend(fontsize=6, ncol=2)
fig.suptitle("Leave-one-out on hypercube simulations: production (dotted) vs R4 (solid)")
fig.tight_layout()

# %%
fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True)
for j, (axis, par) in enumerate(zip(axes.flat, ArinyoFitter.PARAM_NAMES)):
    for i, (sim, res) in enumerate(l1o_res.items()):
        for name, ls in (("production", ":"), ("R4", "-")):
            axis.plot(res[name]["z"], res[name]["par_rel"][:, j], ls, color=f"C{i}",
                      label=f"{sim} {name}")
    axis.axhline(0, color="k", lw=0.5)
    axis.set_title(par)
    axis.grid(alpha=0.25)
for axis in axes[-1]:
    axis.set_xlabel("z")
axes[0, 0].legend(fontsize=6, ncol=2)
fig.suptitle("Leave-one-out: Arinyo parameters, emulator / fit - 1")
fig.tight_layout()

# %% [markdown]
# ### P1D and P3D per redshift, for each held-out simulation
#
# Emulator / fit - 1 for production (dotted) and R4 (solid). P3D is shown per mu
# bin (colour shade: light = mu ~ 0, dark = mu ~ 1). The dashed vertical line in
# P1D is the DESI DR1 k_max at that redshift; the grey band is the tolerance.

# %%
def plot_l1o_by_redshift(sim, kind):
    res = l1o_res[sim]
    z = res["R4"]["z"]
    n_cols = 4
    n_rows = int(np.ceil(len(z) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.3 * n_rows), squeeze=False, sharey=True)
    tol = P3D_TOL if kind == "P3D" else P1D_TOL
    for iz in range(len(z)):
        axis = axes[divmod(iz, n_cols)]
        axis.axhspan(-tol, tol, color="0.8", alpha=0.35, zorder=0)
        for name, ls, colour in (("production", ":", "C0"), ("R4", "-", "C3")):
            r = res[name]["ratios"][iz]
            if kind == "P1D":
                axis.plot(r["k1d"], r["P1D"], ls, color=colour, label=name)
            else:
                for mu in range(n_mu_bins):
                    valid = np.isfinite(r["P3D"][:, mu])
                    shade = 0.35 + 0.65 * mu / (n_mu_bins - 1)
                    axis.plot(r["k3d"][valid, mu], r["P3D"][valid, mu], ls, color=colour, alpha=shade,
                              label=name if mu == n_mu_bins - 1 else None)
        if kind == "P1D":
            axis.axvline(np.interp(z[iz], Z_DESI, KMAX_DESI), color="k", ls="--", lw=0.8)
        axis.axhline(0, color="k", ls=":", lw=0.6)
        axis.set_xscale("log")
        axis.set_ylim(-3 * tol, 3 * tol)
        axis.set_title(f"z = {z[iz]:.2f}")
        axis.grid(alpha=0.25)
    for iz in range(len(z), n_rows * n_cols):
        axes[divmod(iz, n_cols)].set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel(r"$k_\parallel\,[\mathrm{Mpc}^{-1}]$" if kind == "P1D" else r"$k\,[\mathrm{Mpc}^{-1}]$")
    for axis in axes[:, 0]:
        axis.set_ylabel(f"{kind} emu/fit - 1")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Leave-one-out {sim}: {kind}, production (dotted) vs R4 (solid)", y=1.0)
    fig.tight_layout()


for sim in sims_l1o:
    plot_l1o_by_redshift(sim, "P1D")

# %%
for sim in sims_l1o:
    plot_l1o_by_redshift(sim, "P3D")

# %% [markdown]
# ## Near the centre: full emulators on `mpg_seed`
#
# Data prefer a universe close to `mpg_central`, so accuracy there matters most.
# `mpg_seed` has the `mpg_central` parameters but different initial phases, and
# is in no training set. Both **full** emulators (trained on all simulations)
# can therefore be tested on it directly:
#
# - `production`: `forest_mpg_gp`,
# - `R4`: amplitude * Matern(nu=2.5), alpha=1e-4, trained on all simulations.
#
# We also show the average of `mpg_central` and `mpg_seed` (`mpg_central_seed`);
# note that `mpg_central` itself is in both training sets.

# %%
def get_r4_full():
    path = CACHE / "R4_full.pkl"
    if path.is_file() and not RETRAIN:
        with open(path, "rb") as f:
            return pickle.load(f)
    print("training R4 on all simulations ...", flush=True)
    model = RefinedGP(alpha=1e-4).fit(data.inputs, data.targets)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  {model.train_time:.0f} s", model.kernels()[0], flush=True)
    return model


full_models = {
    "production": StoredGP(ROOT / "data" / "gp_emulators" / "forest_mpg_gp.pkl"),
    "R4": get_r4_full(),
}
seed_sets = {
    "mpg_seed": sorted(archive.get_testing_data("mpg_seed"), key=lambda s: s["z"]),
    "mpg_central_seed": sorted(archive.get_central_seed_average(), key=lambda s: s["z"]),
}
for snaps in seed_sets.values():
    assert all(FIT_KEY in s for s in snaps)

# reuse the leave-one-out tables and plots
for label, snaps in seed_sets.items():
    l1o_res[label] = {name: scores_snaps(model, snaps) for name, model in full_models.items()}

for key, title, tol in (("p1d_desi", "P1D, k <= DESI k_max", P1D_TOL),
                        ("p1d_max", "P1D, k <= 6", P1D_TOL),
                        ("p3d_max", "P3D, k <= 4.5", P3D_TOL)):
    print(title)
    for label in seed_sets:
        res = l1o_res[label]
        print(f"  {label:17s} {'z':10s} " + " ".join(f"{z:6.2f}" for z in res["R4"]["z"]))
        for name, r in res.items():
            print(f"  {'':17s} {name:10s} " + " ".join(f"{v:6.3f}" for v in r[key])
                  + f"   pass {np.sum(r[key] <= tol):2d}/{len(r[key])}")
    print()

# %%
for label in seed_sets:
    plot_l1o_by_redshift(label, "P1D")
    plt.gcf().suptitle(f"{label}: P1D, full emulators, production (dotted) vs R4 (solid)", y=1.0)

# %%
for label in seed_sets:
    plot_l1o_by_redshift(label, "P3D")
    plt.gcf().suptitle(f"{label}: P3D, full emulators, production (dotted) vs R4 (solid)", y=1.0)
