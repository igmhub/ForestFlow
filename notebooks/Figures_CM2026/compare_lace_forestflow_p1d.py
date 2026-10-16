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
# # P1D: `lace_mpg` vs `forest_mpg_gp`
#
# For points of the DESI DR1 P1D chain within the training domain, we take the six emulator inputs
# (`Delta2_p, n_p, mF, gamma, sigT_Mpc, kF_Mpc`) at each redshift and the chain's
# `(As, ns)`, and compute the P1D with
#
# - **`lace_mpg`**: LaCE `CH24_mpgcen_gpr`, which emulates the P1D directly,
# - **`forest_mpg_gp`**: the ForestFlow GP, which emulates the Arinyo parameters, and the P1D
#   is then obtained by projecting the Arinyo P3D (cosmology-dependent linear power from `As, ns`).
#
# The inputs are read from `int_data_figs/desi_priors_cosmo_IGM_from_p1d.npy` (already
# derived from the chain), so neither `cup1d` nor the raw chain is needed.

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

from forestflow.emulator import P1DEmulator
from lace.cosmo import cosmology
from lace.emulator.emulator_manager import set_emulator

# %% [markdown]
# ## Setup

# %%
n_points = 100  # in-domain chain points to evaluate at each redshift
seed = 0
k_Mpc = np.geomspace(0.1, 4.0, 60)  # lace_mpg is valid up to kmax_Mpc = 4.25
emu_keys = ["Delta2_p", "n_p", "mF", "gamma", "sigT_Mpc", "kF_Mpc"]

# Fiducial cosmology of the ForestFlow linear theory; (As, ns) are overridden per point.
# It must be Planck18, the cosmology used to compute Delta2_p and n_p from the chain;
# otherwise the linear power seen by ForestFlow and the Delta2_p seen by lace differ.
fid_cosmo = cosmology.Cosmology(cosmo_label="Planck18").input_cosmo_params_dict

# %%
chain = np.load(
    "int_data_figs/desi_priors_cosmo_IGM_from_p1d.npy", allow_pickle=True
).item()
zs = chain["zs"]
nz = len(zs)
n_chain = chain["As"].shape[0]

# %%
lace_mpg = set_emulator("CH24_mpgcen_gpr")

ff_gp = P1DEmulator(name_emu="forest_mpg_gp")
ff_gp.set_cosmology(fid_cosmo)

# %% [markdown]
# ## Training domain
#
# Both emulators are trained on the same MPG simulations. We only compare them at
# chain points whose six inputs all lie within the range of the training inputs;
# elsewhere both emulators extrapolate.

# %%
X_train = ff_gp.emulator.gp.X_train_
train = ff_gp.emulator.transform.transf_stand(
    {lab: X_train[:, i] for i, lab in enumerate(ff_gp.emulator.input_labels)},
    direct=False,
    type_stand="input",
)
in_domain = np.ones((n_chain, nz), dtype=bool)
print(f"{'param':9s} {'training range':>18s}   fraction of chain outside, per z")
for key in emu_keys:
    lo, hi = np.min(train[key]), np.max(train[key])
    out = (chain[key] < lo) | (chain[key] > hi)
    in_domain &= ~out
    print(f"{key:9s} [{lo:7.3f}, {hi:7.3f}]  ", " ".join(f"{f:.2f}" for f in out.mean(0)))
print("in domain", " " * 21, " ".join(f"{f:.2f}" for f in in_domain.mean(0)))

# %%
# for each redshift, a random subsample of the chain points that are in domain there
rng = np.random.default_rng(seed)
sel = []
for iz in range(nz):
    good = np.flatnonzero(in_domain[:, iz])
    sel.append(rng.permutation(good)[:n_points])
n_sel = np.array([len(ss) for ss in sel])
for iz in range(nz):
    print(f"z={zs[iz]:.2f}: {n_sel[iz]} in-domain points")

# %% [markdown]
# ## Evaluate both emulators

# %%
# arrays (nz, n_points, nk); NaN where a redshift has fewer than n_points in domain
p1d_lace = np.full((nz, n_points, k_Mpc.size), np.nan)
p1d_gp = np.full((nz, n_points, k_Mpc.size), np.nan)

for iz in range(nz):
    ss = sel[iz]
    if ss.size == 0:
        continue
    # lace_mpg: one call per point
    for jj, ii in enumerate(ss):
        p1d_lace[iz, jj] = np.squeeze(
            lace_mpg.emulate_p1d_Mpc({key: chain[key][ii, iz] for key in emu_keys}, k_Mpc)
        )
    # forest_mpg_gp: batched over points, one redshift at a time
    p1d_gp[iz, : ss.size] = ff_gp.emulate_p1d_Mpc(
        zs[iz : iz + 1],
        np.broadcast_to(k_Mpc, (ss.size, 1, k_Mpc.size)).copy(),
        {key: chain[key][ss, iz : iz + 1] for key in emu_keys},
        cosmology_parameters=[{"As": chain["As"][ii], "ns": chain["ns"][ii]} for ii in ss],
    )[:, 0]

ratio = p1d_gp / p1d_lace - 1

# %% [markdown]
# ## P1D for individual chain points

# %%
iz_plot = [0, 3, 5, 7, 9, 11]
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
for jz, iz in enumerate(iz_plot):
    col = f"C{jz}"
    for ii in range(min(3, n_sel[iz])):
        lab_l = f"z={zs[iz]:.2f}" if ii == 0 else None
        axs[0].plot(k_Mpc, k_Mpc * p1d_lace[iz, ii] / np.pi, col, label=lab_l)
        axs[0].plot(k_Mpc, k_Mpc * p1d_gp[iz, ii] / np.pi, col + "--")
        axs[1].plot(k_Mpc, ratio[iz, ii], col, alpha=0.7)
axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].set_xlabel(r"$k_\parallel$ [Mpc$^{-1}$]")
axs[0].set_ylabel(r"$k_\parallel P_\mathrm{1D}/\pi$")
axs[0].set_title("solid: lace_mpg, dashed: forest_mpg_gp (3 chain points)")
axs[0].legend()
axs[1].axhspan(-0.01, 0.01, color="0.85", zorder=0)
axs[1].axhline(0, color="k", lw=0.5)
axs[1].set_xscale("log")
axs[1].set_xlabel(r"$k_\parallel$ [Mpc$^{-1}$]")
axs[1].set_ylabel(r"$P_\mathrm{1D}^\mathrm{GP}/P_\mathrm{1D}^\mathrm{lace} - 1$")
plt.tight_layout()

# %% [markdown]
# ## Relative difference over in-domain points, per redshift
#
# Median and 16-84 percentile band of `P1D_gp / P1D_lace - 1` over the in-domain
# chain points at each redshift. The grey band marks 1%.

# %%
fig, axs = plt.subplots(3, 4, figsize=(16, 9), sharex=True, sharey=True)
for iz, ax in enumerate(axs.ravel()):
    ax.axhspan(-0.01, 0.01, color="0.85", zorder=0)
    ax.axhline(0, color="k", lw=0.5)
    if n_sel[iz] > 0:
        lo, med, hi = np.percentile(ratio[iz, : n_sel[iz]], [16, 50, 84], axis=0)
        ax.fill_between(k_Mpc, lo, hi, alpha=0.3)
        ax.plot(k_Mpc, med)
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 0.05)
    ax.set_title(f"z={zs[iz]:.2f} ({n_sel[iz]} points)")
for ax in axs[-1]:
    ax.set_xlabel(r"$k_\parallel$ [Mpc$^{-1}$]")
for ax in axs[:, 0]:
    ax.set_ylabel(r"$P^\mathrm{GP}/P^\mathrm{lace}-1$")
plt.tight_layout()

# %% [markdown]
# **Expected floor.** Evaluated at the inputs of the training simulations themselves
# (`mpg_central`, `mpg_0`, `mpg_5`, `mpg_12`, `mpg_20`, with each simulation's
# cosmology), `forest_mpg_gp` reproduces the Arinyo fits to within ~1%, while
# `lace_mpg` differs from the same fits by about +1.5% at k=0.1 and -1% at k=4 Mpc^-1.
# That difference comes from their training targets: the GP is trained on the Arinyo
# fit, and lace on LaCE's smoothing of the measured P1D. So GP/lace differs at the
# 1-2% level even for perfectly sampled inputs.

# %% [markdown]
# ## Summary table (in-domain points, 0.1 < k < 4 Mpc$^{-1}$)

# %%
print(f"{'z':>5} {'npts':>5} {'mean':>8} {'rms':>8} {'max|diff|':>10}")
for iz, z in enumerate(zs):
    r = ratio[iz, : n_sel[iz]]
    if r.size == 0:
        print(f"{z:5.2f} {0:5d}")
        continue
    print(f"{z:5.2f} {n_sel[iz]:5d} {r.mean():8.4f} {np.sqrt((r**2).mean()):8.4f} {np.abs(r).max():10.4f}")
r = ratio[np.isfinite(ratio)]
print(f"{'all':>5} {n_sel.sum():5d} {r.mean():8.4f} {np.sqrt((r**2).mean()):8.4f} {np.abs(r).max():10.4f}")

# %% [markdown]
# ## Scatter across the chain
#
# Do both emulators propagate the chain's scatter in the same way? Compare the
# relative std of `P1D` over the in-domain chain points at fixed `k`, for a given `z`.

# %%
iz = 3
nn = n_sel[iz]
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(k_Mpc, p1d_lace[iz, :nn].std(0) / p1d_lace[iz, :nn].mean(0), label="lace_mpg")
ax.plot(k_Mpc, p1d_gp[iz, :nn].std(0) / p1d_gp[iz, :nn].mean(0), "--", label="forest_mpg_gp")
ax.set_xscale("log")
ax.set_xlabel(r"$k_\parallel$ [Mpc$^{-1}$]")
ax.set_ylabel(r"std($P_\mathrm{1D}$)/mean over chain points")
ax.set_title(f"z={zs[iz]:.2f}")
ax.legend()
plt.tight_layout()

# %%
