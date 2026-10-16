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
# # Convergence of the P3D to P1D quadrature
#
# P1D is ``int dln(k_perp) k_perp**2 P3D / (2 pi)`` over `k_perp` in
# [1e-3, 100]/Mpc. Two quadratures are compared:
#
# * `P1DIntegrator`: nodes in `ln(k_perp)` (Simpson or Gauss-Legendre).
# * `P1DIntegratorKMu` (default of `ArinyoModel.P1D_Mpc`): with
#   `k_perp dk_perp = k dk`, Gauss-Legendre in `ln k` at fixed `k_par`, over
#   the same domain. Optionally split at `k_split` (default 0.4/Mpc, 20 + 20
#   nodes): for small `k_par` the integrand peaks at `k ~ k_par` and crosses
#   the BAO wiggles of the linear power, so a dense segment below `k_split`
#   and a sparse one above beat any single smooth rule.
#
# The cost of every likelihood call scales linearly with the number of nodes,
# so we want the smallest number that is converged.
#
# Integration accuracy depends on the Arinyo coefficients, not on the
# emulator, so we test it on the saved `arinyo_fixp3d` fits of the central
# simulation plus a random subset of training simulations, all evaluated with
# the central cosmology (the linear power only sets the overall shape).
#
# The reference is a 1024-node Gauss-Legendre rule. Errors are shown
# (i) as a relative P1D error and (ii) in units of the leave-one-out emulator
# error stored in `data/covariance/l1O_cov_<emulator>.npy`: a quadrature error
# well below 1 (we suggest < 0.1) of that error is harmless.

# %%
# %load_ext autoreload
# %autoreload 2

import os

import matplotlib.pyplot as plt
import numpy as np

import forestflow
from forestflow.archive.gadget_archive import GadgetArchive3D
from forestflow.model.arinyo import ArinyoModel
from forestflow.statistics.p1d import P1D_Mpc, P1DIntegrator, P1DIntegratorKMu
from lace.cosmo import cosmology

# %% [markdown]
# ## Settings

# %%
emulator_label = "forest_mpg_gp"  # selects the emulator-error file
zs_test = [2.0, 3.0, 4.5]
n_random_sims = 12
seed = 0

# parallel wavenumbers: emulator range of the l1O covariance
k_par = np.geomspace(0.09, 6.0, 64)

nodes_simpson = [17, 25, 33, 48, 64, 75, 96, 99, 128, 192, 256]
nodes_gl = [8, 12, 16, 24, 32, 48, 64, 96]
# P1DIntegratorKMu: total nodes without split, nodes per segment with split
nodes_kmu_nosplit = [16, 24, 32, 40, 48, 64, 96]
nodes_kmu_split = [8, 12, 16, 20, 24, 32]
k_split = 0.4
n_reference = 1024

# %% [markdown]
# ## Test points: Arinyo coefficients and linear power

# %%
archive = GadgetArchive3D(addcentral=True)

central = [
    s for s in archive.get_testing_data("mpg_central") if s["z"] in zs_test
]
rng = np.random.default_rng(seed)
train = [s for s in archive.training_data if s["z"] in zs_test]
train = [train[i] for i in rng.choice(len(train), n_random_sims, replace=False)]
points = central + train
print(f"{len(central)} central and {len(train)} training points")

sim0 = central[0]
cosmo_params = {
    par: (0.0 if par == "omk" else value)
    for par, value in sim0["cosmo_params"].items()
}
model = ArinyoModel(cosmology.Cosmology(cosmo_params_dict=cosmo_params))


def p1d_all(integrator):
    """P1D (Mpc) of every test point, shape (n_points, n_k)."""
    if isinstance(integrator, P1DIntegratorKMu):
        p3d = model.P3D_Mpc_k_mu
    else:
        p3d = model.P3D_Mpc_kpar_kperp
    out = []
    for sim in points:
        out.append(
            P1D_Mpc(
                model.linear.get_linear_theory(sim["z"]),
                sim["z"],
                k_par,
                p3d,
                sim["arinyo_fixp3d"],
                integrator=integrator,
            )
        )
    return np.array(out)


# %% [markdown]
# ## Emulator error to compare with

# %%
cov = np.load(
    os.path.join(
        os.path.dirname(forestflow.__path__[0]),
        "data",
        "covariance",
        f"l1O_cov_{emulator_label}.npy",
    ),
    allow_pickle=True,
).item()

sigma_emu = np.zeros((len(points), len(k_par)))
for ii, sim in enumerate(points):
    sel = cov["zz_zk"] == cov["zz"][np.argmin(np.abs(cov["zz"] - sim["z"]))]
    sigma = np.sqrt(np.diag(cov["cov_zk"])[sel])
    sigma_emu[ii] = np.interp(k_par, cov["k_Mpc_zk"][sel], sigma)

# %% [markdown]
# ## Reference and node scans

# %%
ref = p1d_all(
    P1DIntegrator(n_k_perp=n_reference, method="gauss_legendre")
)


def scan(method, nodes, **kwargs):
    rel = {}
    for n in nodes:
        res = p1d_all(P1DIntegrator(n_k_perp=n, method=method, **kwargs))
        rel[n] = res / ref - 1
    return rel


rel_simpson = scan("simpson", nodes_simpson)
rel_gl = scan("gauss_legendre", nodes_gl)
rel_kmu_nosplit = {
    n: p1d_all(
        P1DIntegratorKMu(n_low=n // 2, n_high=n - n // 2, k_split_iMpc=None)
    ) / ref - 1
    for n in nodes_kmu_nosplit
}
rel_kmu_split = {
    2 * n: p1d_all(P1DIntegratorKMu(n_low=n, n_high=n, k_split_iMpc=k_split)) / ref - 1
    for n in nodes_kmu_split
}
schemes = {
    "Simpson ln k_perp": rel_simpson,
    "GL ln k_perp": rel_gl,
    "GL ln k, no split": rel_kmu_nosplit,
    f"GL ln k, split {k_split}": rel_kmu_split,
}


def summary(rel):
    """Worst relative error and worst error in units of the emulator error."""
    n = list(rel)
    max_rel = np.array([np.abs(rel[i]).max() for i in n])
    max_sig = np.array([(np.abs(rel[i]) / sigma_emu).max() for i in n])
    return np.array(n), max_rel, max_sig


# %% [markdown]
# ## Convergence curves
#
# Worst case over all test points and k, versus the total number of nodes.
# The dashed line marks 10% of the emulator error and the vertical line the
# default of `P1DIntegratorKMu` (40 nodes).

# %%
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for (label, rel), marker in zip(schemes.items(), "os^v"):
    n, max_rel, max_sig = summary(rel)
    axs[0].loglog(n, max_rel, marker=marker, label=label)
    axs[1].loglog(n, max_sig, marker=marker, label=label)
axs[0].set_ylabel("max |P1D / P1D_ref - 1|")
axs[1].set_ylabel("max error / emulator error")
axs[1].axhline(0.1, color="k", ls="--")
for ax in axs:
    ax.axvline(40, color="gray", ls=":")
    ax.set_xlabel("total number of nodes")
    ax.legend()
plt.tight_layout()

# %%
for label, rel in schemes.items():
    print(label)
    for n, max_rel, max_sig in zip(*summary(rel)):
        print(f"  n={n:4d}  max rel = {max_rel:.2e}   max / sigma_emu = {max_sig:.2e}")

# %% [markdown]
# ## Error versus k_parallel
#
# The error of the `ln k_perp` rules and of the unsplit `ln k` rule is
# concentrated at small `k_par`, where the integrand runs through the BAO.

# %%
fig, ax = plt.subplots(figsize=(6, 4))
for label, rel, n in (
    ("Simpson ln k_perp", rel_simpson, 99),
    ("GL ln k, no split", rel_kmu_nosplit, 40),
    (f"GL ln k, split {k_split}", rel_kmu_split, 40),
):
    ax.semilogx(k_par, np.abs(rel[n]).max(axis=0), label=f"{label}, {n} nodes")
ax.semilogx(k_par, np.median(sigma_emu, axis=0) * 0.1, "k--", label="0.1 sigma_emu")
ax.set_yscale("log")
ax.set_xlabel(r"$k_\parallel$ [1/Mpc]")
ax.set_ylabel("max |P1D / P1D_ref - 1|")
ax.legend()

# %% [markdown]
# ## Position of the split
#
# The split should sit just above the BAO range. Below ~0.3/Mpc it cuts
# through the wiggles; far above it starves the low-k segment.

# %%
for n in (16, 20, 24):
    row = [
        np.abs(
            p1d_all(P1DIntegratorKMu(n_low=n, n_high=n, k_split_iMpc=ks)) / ref - 1
        ).max()
        for ks in (0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6)
    ]
    print(f"{n:2d} + {n:2d}  k_split = 0.25..0.6:", " ".join(f"{x:.1e}" for x in row))

# %% [markdown]
# ## Integration range
#
# The nodes are log spaced between `k_perp_min` and `k_perp_max`. Truncating
# the range biases P1D; extending it only wastes nodes. To isolate truncation
# from node-count error, each range is integrated with `n_reference`
# Gauss-Legendre nodes and compared with the widest range (1e-4 to 200/Mpc,
# 1024 nodes). LaCE linear power is only valid up to ~200/Mpc.

# %%
ref_wide = p1d_all(
    P1DIntegrator(1e-4, 200.0, n_k_perp=n_reference, method="gauss_legendre")
)


def range_scan(kmin_list, kmax_list):
    out = {}
    for kmin in kmin_list:
        for kmax in kmax_list:
            res = p1d_all(
                P1DIntegrator(
                    kmin, kmax, n_k_perp=n_reference, method="gauss_legendre"
                )
            )
            out[(kmin, kmax)] = res / ref_wide - 1
    return out


range_res = range_scan([1e-4, 1e-3, 1e-2, 3e-2], [10.0, 30.0, 100.0, 200.0])
print("k_perp_min  k_perp_max   max rel   max/sigma_emu")
for (kmin, kmax), rel in range_res.items():
    print(
        f"{kmin:9.0e} {kmax:10.0f}  {np.abs(rel).max():.1e}"
        f"   {(np.abs(rel) / sigma_emu).max():.1e}"
    )

# %% [markdown]
# ## Take-away
#
# Pick the smallest node count whose worst-case error / emulator error stays
# below ~0.1 for the method you use, and a range whose truncation error is
# negligible. Update the defaults in `P1DIntegratorKMu` (and
# `_DEFAULT_P1D_K_MU_INTEGRATOR`) accordingly. Rerun this notebook after changing
# the Arinyo parameter priors or the k range of the emulator.
