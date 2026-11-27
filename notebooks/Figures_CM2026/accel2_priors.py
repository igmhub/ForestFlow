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
# # Validation with ACCEL2

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

from cup1d.p1ds.synthetic.data_accel2 import load_data as load_accel2_data
from forestflow.paths import get_lya_data_path
from forestflow.priors.loaders import load_map_igm_p3d
from forestflow.priors.processing import accel2_k_mu_grid
from forestflow.priors.plots import plot_p3d_validation, plot_bias_beta_zev_val

# %% [markdown]
# #### Provenance of the ACCEL2 P1D chain (checked Nov 2026)
#
# `load_map_igm_p3d(lab_sample="accel2")` maps the cup1d chain in
# `P1D_CHAINS["accel2"]` (`cup1d/postprocessing/forestflow_priors.py`):
# `<LYA_DATA_PATH>/accel2/chains/chain_1/` (written 2026-02-17).
#
# **How the chain was run.** `cup1d/scripts/validation/val_sampler.py accel2`
# (`name_variation = "sim_accel2"`, output under
# `<out_folder>/DESIY1_QMLE3/sim_accel2/CH24_mpgcen_gpr/chain_N`). Settings,
# consistent with the `fitter_results.npy` saved with the chain:
# * `data_label="accel2"`, `synth_cov_label="DESIY1_QMLE3"`, `apply_smoothing=True`, `add_noise=False`;
# * emulator `CH24_mpgcen_gpr` (k_max = 4.25/Mpc), `emu_cov_type="full"`;
# * `fit_type="global_opt"`, 2.2 <= z <= 4.2, cosmology free (As, ns), 4-node IGM
#   (tau_eff, sigT, gamma, kF), and the DR1 nuisance model (SiIII, SiII,
#   SiIIa-SiIIb, HCD damping, resolution). For `sim_` variations, cup1d lowers
#   the lower prior bound of the metal and HCD amplitudes to -10.5;
# * minimizer, then MCMC (`mcmc_conf="explore"`).
#
# `val_sampler.py` called `args.set_baseline`, which only exists in
# `cup1d/old_code/old_input.py`; the script was removed from cup1d (it is in
# the git history) and a re-run needs a YAML configuration (`Args.from_yaml`). The folders
# `DESIY1_QMLE3/sim_accel2/CH24_mpgcen_gpr/chain_*` in this directory are empty.
#
# **How the mock is built** (`cup1d/p1ds/factory.py` -> `Accel2_P1D` in
# `cup1d/p1ds/synthetic/data_accel2.py`):
# 1. ACCEL2 P1D read from the text files `frontier_grid/p1d_from_sim/l160_r25`
#    (`load_data`, average of the 3 axes, k < 10/Mpc); same data as `ACCEL2_6144_160`
#    in the Nyx file. Snapshots z = 2.0, 2.6, 3.0, 3.6, 4.0, 5.0.
# 2. Smoothing at each snapshot with `set_smoothing_Mpc` -> LaCE
#    `apply_smoothing` (`gpolyfit`): the ForestFlow template `ff_mpgcen.npy`
#    interpolated at mF = exp(-0.0025 (1+z)^3.7) times exp of a five-parameter
#    function fitted to ACCEL2 over 0 < k < 4.25/Mpc. Tested against a spline
#    and the template alone in `accel2_p1d_smoothing.ipynb` (referee comment 2).
# 3. `_load_p1d` (default `interp_to_cov=False`): ln P interpolated linearly in
#    (z, k) onto the QMLE3 grid, z = 2.2, 2.4, ..., 4.2 and k = 0.0013 to
#    0.025-0.041 s/km. Only z = 2.6, 3.0 and 3.6 are snapshots; the others are
#    interpolated between them, and z = 2.2 and 2.4 depend on the z = 2.0 snapshot.
#    k below the smallest simulation k is removed.
# 4. QMLE3 total covariance (`cov_syst_type="red"`), nearest DESI bin per point.
# 5. Fiducial contaminants and systematics from the theory object are applied to the mock.

# %%
data_accel2 = load_accel2_data(str(get_lya_data_path() / "accel2" / "frontier_grid"))
knew3d, munew3d = accel2_k_mu_grid(data_accel2)
data = load_map_igm_p3d(lab_sample="accel2", emu_type="legacy", accel2_data=data_accel2)

# %%
data.keys()

# %%
# for key in data.keys():
#     try:
#         print(key, data[key].keys())
#     except AttributeError:
#         print(data[key])

# %% [markdown]
# Quick check forestflow P1D agrees with data

# %%
z_use = 2.6
ind_accel = np.argmin(np.abs(data_accel2["z"] - z_use))
ind_data = np.argmin(np.abs(data["zs"] - z_use))

k1d_Mpc = data_accel2["k1d_Mpc"]
ind = np.argwhere(k1d_Mpc < 4)[:,0]
k1d_Mpc = k1d_Mpc[ind]
plt.plot(k1d_Mpc, k1d_Mpc * data_accel2["p1d_Mpc"][ind_accel, ind], "k", label="accel2")
for ii in range(100):
    plt.plot(k1d_Mpc, k1d_Mpc * data["forest_out"]["p1d"][ii, ind_data, ind], "C1", alpha=0.1)
plt.plot(k1d_Mpc, k1d_Mpc * np.median(data["forest_out"]["p1d"][:, ind_data, ind], axis=0), "C2", label="ForestFlow, median")
plt.xlabel(r"$k_\parallel$ [Mpc$^{-1}$]")
plt.ylabel(r"$k_\parallel P_\mathrm{1D}$")
plt.title(f"z={z_use}")
plt.legend()

# %%
plot_p3d_validation(knew3d, munew3d, data, data_accel2)

# %%
from forestflow.priors.plots import load_accel2_mcmc_fits

# simulation values to compare with:
#   "mcmc": our MCMC fits of ACCEL2 (notebooks/arinyo_fits/Arinyo_fit_accel2), median and 16-84% errors
#   "refit": our minimizer fits of ACCEL2, no errors
#   "paper": the fits in the ACCEL2 article
sim_values = "mcmc"
sim_errorbars = True  # False: simulation values as circles without error bars
accel2_refit_file = "../arinyo_fits/Arinyo_fit_accel2_l160_r25_redshift_evolution.npy"

if sim_values == "mcmc":
    sim_fits = load_accel2_mcmc_fits("../arinyo_fits", tag="k3d3.5_k1d1")
elif sim_values == "refit":
    sim_fits = accel2_refit_file
else:
    sim_fits = None
plot_bias_beta_zev_val(data, sim_fits=sim_fits, sim_errorbars=sim_errorbars)

# %% [markdown]
# Arinyo parameters at z=2.33 with the legacy emulator (mpg_hypercube.pt cINN), for
# comparison (columns: mean, std, 5th and 95th percentiles).
#
# DESI DR1 P1D chain:
# ```
# bias -0.124 0.007 -0.135 -0.113
# beta 1.417 0.044 1.346 1.49
# q1 0.282 0.055 0.193 0.369
# kvav 0.554 0.049 0.485 0.639
# av 0.426 0.048 0.353 0.51
# bv 1.674 0.023 1.642 1.716
# kp 10.817 0.388 10.349 11.529
# q2 0.27 0.059 0.182 0.375
# ```
#
# ACCEL2 DR1-like mock:
# ```
# bias -0.109 0.009 -0.122 -0.094
# beta 1.669 0.079 1.548 1.805
# q1 0.281 0.046 0.205 0.357
# kvav 0.422 0.046 0.342 0.496
# av 0.272 0.059 0.169 0.355
# bv 1.656 0.03 1.608 1.708
# kp 11.833 0.531 11.066 12.709
# q2 0.12 0.042 0.062 0.198
# ```

# %% [markdown]
# Same table with the current emulator

# %%
for lab_sample in ["desi", "accel2"]:
    dd = data if lab_sample == "accel2" else load_map_igm_p3d(lab_sample="desi")
    iz = np.argmin(np.abs(dd["zs"] - 2.33))
    print(lab_sample)
    for par in ["bias", "beta", "q1", "kvav", "av", "bv", "kp", "q2"]:
        sign = -1 if par == "bias" else 1
        vals = sign * np.abs(dd["forest_out"][par][:, iz])
        p5, p95 = np.nanpercentile(vals, [5, 95])
        print(par, np.round(np.nanmean(vals), 3), np.round(np.nanstd(vals), 3), np.round(p5, 3), np.round(p95, 3))
    print()

# %%
# systematic error of the model: difference between the model (Arinyo parameters
# mapped from the Accel2 P1D) and the Arinyo parameters fitted directly to the
# Accel2 P3D, as in plot_bias_beta_zev_val. Relative difference of the median,
# |model/sim - 1|, at the simulation redshifts covered by the model
from forestflow.priors.plots import ACCEL2_ARINYO_FITS

# simulation label: model label
sys_pars = {
    "bias_delta": "bias_delta",
    "beta": "beta",
    "q1": "q1",
    "q2": "q2",
    "k_v": "kvav",
    "a_v": "av",
    "b_v": "bv",
    "k_p": "kp",
}

z_sim = ACCEL2_ARINYO_FITS["z"]
ind_sim = np.argwhere(z_sim >= data["zs"].min() - 1e-3)[:, 0]
iz_model = [np.argmin(np.abs(data["zs"] - z)) for z in z_sim[ind_sim]]
assert np.allclose(data["zs"][iz_model], z_sim[ind_sim]), "redshift grids differ"

sys_shift = {"z": z_sim[ind_sim]}
sys_err_sim = {"z": z_sim[ind_sim]}
for par_sim, par in sys_pars.items():
    if par == "kvav":
        # the simulation fits quote k_v, the model kvav = k_v**av
        val = data["forest_out"]["kvav"] ** (1 / data["forest_out"]["av"])
    else:
        val = data["forest_out"][par]
    med_model = np.nanmedian(val[:, iz_model], axis=0)
    val_sim = ACCEL2_ARINYO_FITS[par_sim]["value"][ind_sim]
    err_sim = ACCEL2_ARINYO_FITS[par_sim]["error"][ind_sim]
    sys_shift[par_sim] = np.abs(med_model / val_sim - 1)
    # relative error of the simulation fit, to compare with the shift
    sys_err_sim[par_sim] = np.abs(err_sim / val_sim)
    print(par_sim, np.round(sys_shift[par_sim], 3), np.round(sys_err_sim[par_sim], 3))

# %% [markdown]
# New GP
#
# ```
# Arinyo mapping from emulator: forest_mpg_gp
# desi
# bias -0.118 0.006 -0.128 -0.11
# beta 1.599 0.054 1.509 1.685
# q1 0.384 0.105 0.196 0.54
# kvav 0.575 0.049 0.499 0.658
# av 0.352 0.049 0.271 0.433
# bv 1.555 0.018 1.531 1.589
# kp 10.169 0.28 9.803 10.683
# q2 0.275 0.091 0.144 0.434
#
# accel2
# bias -0.103 0.007 -0.113 -0.091
# beta 1.83 0.105 1.67 2.018
# q1 0.362 0.095 0.206 0.524
# kvav 0.46 0.045 0.388 0.536
# av 0.235 0.052 0.15 0.322
# bv 1.532 0.025 1.492 1.574
# kp 11.203 0.403 10.617 11.922
# q2 0.124 0.066 0.023 0.239
#
# ```

# %% [markdown]
#

# %% [markdown]
# Systematic shift as a function of redshift, with the relative error of the
# simulation fits and a linear fit, to check whether it can be extrapolated

# %%
fig, ax = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
ax = ax.reshape(-1)
zfit = np.linspace(2.0, 4.4, 50)
for ii, par_sim in enumerate(sys_pars):
    fit = np.polyfit(sys_shift["z"], sys_shift[par_sim], 1)
    ax[ii].errorbar(
        sys_shift["z"], sys_shift[par_sim], sys_err_sim[par_sim], fmt="o", label="Model vs sim"
    )
    ax[ii].plot(zfit, np.polyval(fit, zfit), "C1-", label="Linear fit")
    ax[ii].axhline(0, color="k", ls=":", lw=1)
    ax[ii].set_title(par_sim)
for ii in range(4, 8):
    ax[ii].set_xlabel(r"$z$")
ax[0].set_ylabel(r"$|x_\mathrm{model}/x_\mathrm{sim} - 1|$")
ax[4].set_ylabel(r"$|x_\mathrm{model}/x_\mathrm{sim} - 1|$")
ax[0].legend()
plt.tight_layout()

# %%
np.save("int_data_figs/legacy_emu/accel2_priors_sys_shift.npy", sys_shift)

# %% [markdown]
# ## Systematic error of holography from the excess scatter
#
# The shift above, $|x_{\rm holo}/x_{\rm sim}-1|$ fitted linearly in z, counts the
# statistical noise of both sides as systematic error. Even a perfect model
# gives $|r|\approx0.8\,\sigma_{\rm stat}$ on average. Here we keep only the
# part of the shifts that the errors do not explain.
#
# At each simulation redshift common to the holography samples and our MCMC fits
# of ACCEL2 (`notebooks/arinyo_fits/Arinyo_fit_accel2`), and for each parameter:
#
# * relative shift: $r_z = x_{{\rm holo},z}/x_{{\rm sim},z} - 1$, with the
#   holography posterior median and the MCMC median;
# * its statistical error, all in fractional units:
#   $s_z^2 = (x_{\rm holo}/x_{\rm sim})^2\left[(\sigma_{\rm holo}/x_{\rm holo})^2 + (\sigma_{\rm sim}/x_{\rm sim})^2\right]$,
#   with $\sigma$ half the 16-84% range of each posterior;
# * model: $r_z \sim \mathcal{N}(0,\ s_z^2 + \sigma_{\rm sys}^2)$, with one
#   constant fractional $\sigma_{\rm sys}$ per parameter shared by all
#   redshifts;
# * $-2\ln L(\sigma_{\rm sys}) = \sum_z \left[r_z^2/(s_z^2+\sigma_{\rm sys}^2) + \ln(s_z^2+\sigma_{\rm sys}^2)\right]$,
#   minimized over $\sigma_{\rm sys}\ge0$ on a grid.
#
# If the shifts are consistent with the errors, $\sigma_{\rm sys}\to0$. We also
# report a 68% upper limit, where $-2\ln L$ rises by 1 above its minimum on the
# large-$\sigma_{\rm sys}$ side. With only a few correlated redshifts the
# estimate is uncertain, and the upper limit is the conservative choice
# (`SYS_ESTIMATE = "upper68"`). The four redshifts are treated as independent,
# which makes the upper limit somewhat too tight.
#
# $\sigma_{\rm sys}$ scales with the MCMC errors, i.e. with the
# `PSEUDO_LIKE_NORM` of those fits. The result is saved with the layout of
# `sys_shift` (the constant repeated at each redshift), so `_plot_sys_band` and
# `add_p1d_sys`, which fit a line in z, recover the constant unchanged.

# %%
import glob

MCMC_DIR = "../arinyo_fits"
MCMC_TAG = "k3d3.5_k1d1"  # scale cuts in the MCMC file names
SYS_ESTIMATE = "ml"  # "ml": maximum likelihood; "upper68": 68% upper limit

# ---- MCMC errors of our ACCEL2 fits, one file per redshift
mcmc_err = {}
for fname in sorted(glob.glob(f"{MCMC_DIR}/Arinyo_mcmc_accel2_l160_r25_z*_{MCMC_TAG}_errors.npy")):
    entry = np.load(fname, allow_pickle=True).item()
    mcmc_err[round(entry["z"], 2)] = entry
settings_ref = next(iter(mcmc_err.values()))["settings"]
for zz, entry in mcmc_err.items():
    diff = {k for k in settings_ref if k != "z" and entry["settings"][k] != settings_ref[k]}
    if diff:
        raise ValueError(f"MCMC at z={zz} has different settings: {diff}")
print("MCMC fits at z =", sorted(mcmc_err), "; PSEUDO_LIKE_NORM =", settings_ref["pseudo_like_norm"])

# ---- common redshifts with the holography samples
z_common = np.array([zz for zz in sorted(mcmc_err) if np.min(np.abs(data["zs"] - zz)) < 1e-3])
iz_holo = [int(np.argmin(np.abs(data["zs"] - zz))) for zz in z_common]
print("redshifts used:", z_common)
missing = [zz for zz in ACCEL2_ARINYO_FITS["z"] if (zz >= data["zs"].min() - 1e-3) and (zz not in z_common)]
if missing:
    print("holography redshifts without an MCMC fit:", sorted(missing))


def holography_samples(par_sim):
    """Holography samples of a parameter in the article parametrization."""
    par = sys_pars[par_sim]
    if par == "kvav":
        return data["forest_out"]["kvav"] ** (1 / data["forest_out"]["av"])
    return data["forest_out"][par]


def sigma_sys_ml(rr, ss, n_grid=20001):
    """ML and 68% upper limit of a constant extra scatter, with -2 ln L on a grid."""
    grid = np.linspace(0, 5 * max(np.max(np.abs(rr)), np.max(ss)), n_grid)
    var = ss[None, :] ** 2 + grid[:, None] ** 2
    m2lnl = np.sum(rr[None, :] ** 2 / var + np.log(var), axis=1)
    imin = np.argmin(m2lnl)
    above = np.flatnonzero((np.arange(n_grid) > imin) & (m2lnl > m2lnl[imin] + 1))
    upper = grid[above[0]] if above.size else np.nan
    return grid[imin], upper, grid, m2lnl - m2lnl[imin]


sys_excess = {"z": z_common}
sys_details = {}
print(f"\n{'par':10s} {'sigma_sys':>9s} {'upper68':>8s}   r_z / s_z at z = {z_common}")
for par_sim in sys_pars:
    samples_holo = holography_samples(par_sim)[:, iz_holo]
    p16, x_holo, p84 = np.nanpercentile(samples_holo, [16, 50, 84], axis=0)
    sig_holo = 0.5 * (p84 - p16)
    x_sim = np.array([mcmc_err[zz]["Arinyo_article"][par_sim]["value"] for zz in z_common])
    sig_sim = np.array([mcmc_err[zz]["Arinyo_article"][par_sim]["error"] for zz in z_common])

    ratio = x_holo / x_sim
    rr = ratio - 1
    ss = np.abs(ratio) * np.sqrt((sig_holo / x_holo) ** 2 + (sig_sim / x_sim) ** 2)
    sig_ml, sig_up, grid, dm2lnl = sigma_sys_ml(rr, ss)
    sys_details[par_sim] = {
        "r": rr, "s": ss, "s_holo": np.abs(sig_holo / x_holo), "s_sim": np.abs(sig_sim / x_sim),
        "sigma_sys_ml": sig_ml, "sigma_sys_upper68": sig_up, "grid": grid, "dm2lnl": dm2lnl,
    }
    value = sig_ml if SYS_ESTIMATE == "ml" else sig_up
    sys_excess[par_sim] = np.full(len(z_common), value)
    print(f"{par_sim:10s} {sig_ml:9.4f} {sig_up:8.4f}   {np.round(rr / ss, 2)}")

# %% [markdown]
# Left of each pair: the relative shifts $r_z$ with their statistical errors
# $s_z$. The dark band is $\pm\sigma_{\rm sys}$ (ML) and the light band the 68% upper
# limit. Grey crosses show the previous estimate, $|r|$ against the paper fits.
# Right: $-2\Delta\ln L(\sigma_{\rm sys})$, with the line at 1 that defines the upper
# limit.

# %%
fig, ax = plt.subplots(4, 4, figsize=(15, 12))
for ii, par_sim in enumerate(sys_pars):
    det = sys_details[par_sim]
    axr = ax[(ii // 2), 2 * (ii % 2)]
    axl = ax[(ii // 2), 2 * (ii % 2) + 1]

    axr.axhspan(-det["sigma_sys_upper68"], det["sigma_sys_upper68"], color="C1", alpha=0.15, lw=0)
    axr.axhspan(-det["sigma_sys_ml"], det["sigma_sys_ml"], color="C1", alpha=0.35, lw=0)
    axr.errorbar(z_common, det["r"], det["s"], fmt="o", color="C0", label=r"$r_z\pm s_z$")
    axr.plot(sys_shift["z"], sys_shift[par_sim], "x", color="0.5", label=r"previous $|r|$")
    axr.axhline(0, color="k", ls=":", lw=1)
    axr.set_title(par_sim)
    axr.set_xlabel(r"$z$")
    axr.set_ylabel(r"$x_\mathrm{holo}/x_\mathrm{sim}-1$")

    axl.plot(det["grid"], det["dm2lnl"], color="C1")
    axl.axhline(1, color="k", ls=":", lw=1)
    axl.set_ylim(0, 4)
    finite_up = np.isfinite(det["sigma_sys_upper68"])
    axl.set_xlim(0, 2 * det["sigma_sys_upper68"] if finite_up and det["sigma_sys_upper68"] > 0 else det["grid"][-1])
    axl.set_xlabel(r"$\sigma_\mathrm{sys}$")
    axl.set_ylabel(r"$-2\Delta\ln L$")
ax[0, 0].legend(fontsize=8)
plt.tight_layout()

# %%
# same layout as sys_shift: constant sigma_sys at each redshift, plus how it was obtained
sys_excess_out = dict(sys_excess)
sys_excess_out["meta"] = {
    "method": "excess scatter, constant fractional sigma_sys, maximum likelihood over z",
    "estimate": SYS_ESTIMATE,
    "mcmc_settings": settings_ref,
    "sigma_sys_ml": {par: sys_details[par]["sigma_sys_ml"] for par in sys_pars},
    "sigma_sys_upper68": {par: sys_details[par]["sigma_sys_upper68"] for par in sys_pars},
    "r": {par: sys_details[par]["r"] for par in sys_pars},
    "s": {par: sys_details[par]["s"] for par in sys_pars},
}
sys_excess_file = f"int_data_figs/legacy_emu/accel2_priors_sys_excess_{SYS_ESTIMATE}.npy"
np.save(sys_excess_file, sys_excess_out)
print(f"Saved {sys_excess_file}")

# %% [markdown]
# ## Systematic error of $b_\delta$ and $\beta$: common offset with a floor
#
# This replaces the excess-scatter estimate above for the two parameters we
# propagate. That estimate has three weaknesses: it ignores the sign of the
# shifts (all four $\beta$ shifts are positive), it often returns exactly zero
# with only four points, and it treats the four redshifts as independent.
#
# **Model.** The systematic is a common fractional offset $\mu$ between
# holography and the simulation, the same at all redshifts used:
# $r_z = x_{{\rm holo},z}/x_{{\rm sim},z} - 1 \sim \mathcal{N}(\mu, C)$.
#
# **Covariance.** $C_{ij} = R_i R_j\left[\rho_{ij}\, f_{{\rm holo},i} f_{{\rm holo},j} + \delta_{ij} f_{{\rm sim},i}^2\right]$, with:
# * $R = x_{\rm holo}/x_{\rm sim}$;
# * $f$ the fractional errors, half the 16-84% range of each posterior;
# * $\rho_{ij}$ the correlation of the holography samples across redshift. They
#   all come from the same P1D chain, so they are correlated;
# * the simulation errors independent, since each redshift is a separate MCMC fit.
#
# **Estimate.** Generalized least squares:
# $\hat\mu = \sigma_\mu^2\, \mathbf{1}^T C^{-1} r$, with
# $\sigma_\mu^{-2} = \mathbf{1}^T C^{-1}\mathbf{1}$.
# $\sigma_\mu$ is the smallest offset this test can detect.
#
# **Systematic error propagated:**
# $$\sigma_{\rm sys}^2 = \max\left(\hat\mu^2 - \sigma_\mu^2,\ 0\right) + \left(f_{\rm floor}\,\sigma_\mu\right)^2 .$$
# * When the offset is not detected, $|\hat\mu| \le \sigma_\mu$, this gives
#   $\sigma_{\rm sys} = f_{\rm floor}\,\sigma_\mu$ (`SYS_FLOOR_FRACTION`, 1/3 by
#   default). This is a small systematic we cannot exclude, which inflates the
#   error by about 5% when added in quadrature.
# * Above that, it grows continuously with the excess of the offset over its
#   noise. $\hat\mu^2-\sigma_\mu^2$ is the unbiased estimate of $\mu^2$, so
#   there is no hard detection threshold.
#
# **Redshifts.** `Z_SYS` sets the redshifts used, by default 2.6 and 3.0,
# the closest to the analysis at z=2.33. Other subsets are printed as a
# robustness check.
#
# The output keeps the `sys_shift` layout, with the constant repeated at each
# redshift, and only contains `bias_delta` and `beta`. So `_plot_sys_band` and
# `add_p1d_sys` work unchanged, and the other parameters get no systematic
# band.

# %%
from forestflow.priors.plots import load_accel2_mcmc_fits

SYS_PARS = ["bias_delta", "beta"]
Z_SYS = [2.6, 3.0]  # redshifts used for the systematic error
SYS_FLOOR_FRACTION = 1 / 3  # sigma_sys when the offset is not detected, in units of sigma_mu
MCMC_DIR = "../arinyo_fits"
MCMC_TAG = "k3d3.5_k1d1"  # scale cuts in the MCMC file names

sim_mcmc = load_accel2_mcmc_fits(MCMC_DIR, tag=MCMC_TAG)
z_avail = np.array([zz for zz in sim_mcmc["z"] if np.min(np.abs(data["zs"] - zz)) < 1e-3])
print("redshifts with holography and MCMC:", z_avail)


# holography samples in the article parametrization (k_v = kvav^(1/av))
HOLO_KEYS = {"bias_delta": "bias_delta", "beta": "beta", "q1": "q1", "q2": "q2",
             "a_v": "av", "b_v": "bv", "k_p": "kp"}


def holography_par_samples(par):
    """Holography samples of a parameter; k_v is handled in log space in offset_inputs."""
    return np.asarray(data["forest_out"][HOLO_KEYS[par]])


def offset_inputs(par, z_list):
    """Relative shifts, their covariance and the error budget at z_list."""
    iz_sim = [int(np.argmin(np.abs(sim_mcmc["z"] - zz))) for zz in z_list]
    iz_holo = [int(np.argmin(np.abs(data["zs"] - zz))) for zz in z_list]
    if par == "k_v":
        # k_v = kvav^(1/av) overflows to inf when av -> 0, and percentiles that
        # interpolate between infinite samples are NaN. ln k_v = ln(kvav)/av is
        # finite; percentiles commute with exp, and the correlation is that of ln k_v
        with np.errstate(divide="ignore", invalid="ignore"):
            samp = (
                np.log(np.asarray(data["forest_out"]["kvav"])[:, iz_holo])
                / np.asarray(data["forest_out"]["av"])[:, iz_holo]
            )
        p16, x_holo, p84 = np.exp(np.nanpercentile(samp, [16, 50, 84], axis=0))
    else:
        samp = holography_par_samples(par)[:, iz_holo]
        p16, x_holo, p84 = np.nanpercentile(samp, [16, 50, 84], axis=0)
    f_holo = 0.5 * (p84 - p16) / np.abs(x_holo)
    finite = np.all(np.isfinite(samp), axis=1)
    corr_holo = np.atleast_2d(np.corrcoef(samp[finite], rowvar=False))

    x_sim = sim_mcmc[par]["value"][iz_sim]
    f_sim = np.abs(sim_mcmc[par]["error"][iz_sim] / x_sim)

    ratio = x_holo / x_sim
    rr = ratio - 1
    cov = np.outer(ratio, ratio) * (corr_holo * np.outer(f_holo, f_holo) + np.diag(f_sim**2))
    return {"r": rr, "cov": cov, "corr_holo": corr_holo, "f_holo": f_holo, "f_sim": f_sim}


def common_offset(rr, cov, floor_fraction=SYS_FLOOR_FRACTION):
    """GLS common offset, its error, and the systematic error to propagate."""
    icov = np.linalg.inv(cov)
    ones = np.ones_like(rr)
    sig_mu = 1 / np.sqrt(ones @ icov @ ones)
    mu = sig_mu**2 * (ones @ icov @ rr)
    resid = rr - mu
    return {
        "mu": mu,
        "sigma_mu": sig_mu,
        "sigma_sys": np.sqrt(max(mu**2 - sig_mu**2, 0.0) + (floor_fraction * sig_mu) ** 2),
        "detected": bool(abs(mu) > sig_mu),
        "chi2_zero": float(rr @ icov @ rr),  # no offset, len(rr) dof
        "chi2_offset": float(resid @ icov @ resid),  # around mu, len(rr) - 1 dof
    }


# correlation of the holography samples across redshift
for par in SYS_PARS:
    print(f"\n{par}: correlation of holography samples, z = {z_avail}")
    print(np.round(offset_inputs(par, z_avail)["corr_holo"], 2))

# robustness: the same estimate for several redshift subsets
subsets = [list(Z_SYS), list(z_avail), [2.6, 3.0, 3.6], [3.0, 3.6, 4.0], [2.6], [3.0]]
subsets = [
    sub for ii, sub in enumerate(subsets)
    if all(np.any(np.isclose(z_avail, zz)) for zz in sub) and sub not in subsets[:ii]
]
print("\nshift r = x_holo/x_sim - 1; floor =", round(SYS_FLOOR_FRACTION, 3), "sigma_mu")
for par in SYS_PARS:
    print(f"\n{par}")
    print(f"  {'z used':22s} {'mu':>7s} {'sig_mu':>7s} {'|mu|/sig':>8s} {'chi2(0)':>8s} {'chi2(mu)':>8s} {'sig_sys':>8s}")
    for sub in subsets:
        inp = offset_inputs(par, sub)
        res = common_offset(inp["r"], inp["cov"])
        tag = "  <- Z_SYS" if sub == list(Z_SYS) else ""
        print(
            f"  {str(sub):22s} {100 * res['mu']:+6.2f}% {100 * res['sigma_mu']:6.2f}% "
            f"{abs(res['mu']) / res['sigma_mu']:8.2f} {res['chi2_zero']:8.2f} {res['chi2_offset']:8.2f} "
            f"{100 * res['sigma_sys']:7.2f}%{tag}"
        )

sys_offset_results = {}
for par in SYS_PARS:
    inp = offset_inputs(par, Z_SYS)
    sys_offset_results[par] = {**inp, **common_offset(inp["r"], inp["cov"])}

# %%
fig, ax = plt.subplots(1, len(SYS_PARS), figsize=(11, 4), sharey=False)
par_latex = {"bias_delta": r"$b_\delta$", "beta": r"$\beta$"}
for ii, par in enumerate(SYS_PARS):
    inp_all = offset_inputs(par, z_avail)
    used = np.array([np.any(np.isclose(Z_SYS, zz)) for zz in z_avail])
    err_all = np.sqrt(np.diag(inp_all["cov"]))
    res = sys_offset_results[par]

    ax[ii].errorbar(z_avail[used], inp_all["r"][used], err_all[used], fmt="o", color="C0", label="used")
    if np.any(~used):
        ax[ii].errorbar(
            z_avail[~used], inp_all["r"][~used], err_all[~used],
            fmt="o", mfc="none", color="C0", alpha=0.6, label="not used",
        )
    ax[ii].axhspan(res["mu"] - res["sigma_mu"], res["mu"] + res["sigma_mu"], color="C1", alpha=0.25, lw=0,
                   label=r"$\hat\mu\pm\sigma_\mu$")
    ax[ii].axhline(res["mu"], color="C1")
    for sign in (-1, 1):
        ax[ii].axhline(sign * res["sigma_sys"], color="C3", ls="--",
                       label=r"$\pm\sigma_\mathrm{sys}$" if sign == 1 else None)
    ax[ii].axhline(0, color="k", ls=":", lw=1)
    ax[ii].set_xlabel(r"$z$")
    ax[ii].set_ylabel(r"$x_\mathrm{holo}/x_\mathrm{sim}-1$")
    ax[ii].set_title(
        f"{par_latex[par]}: " + rf"$\hat\mu$={100 * res['mu']:+.1f}$\pm${100 * res['sigma_mu']:.1f}%, "
        + rf"$\sigma_\mathrm{{sys}}$={100 * res['sigma_sys']:.1f}%"
    )
ax[0].legend(fontsize=9)
plt.tight_layout()

# %%
# same layout as sys_shift: the constant sigma_sys repeated at each redshift (at least two,
# since the downstream functions fit a line in z), only for the propagated parameters
assert len(z_avail) >= 2
sys_offset = {"z": z_avail}
for par in SYS_PARS:
    sys_offset[par] = np.full(len(z_avail), sys_offset_results[par]["sigma_sys"])
sys_offset["meta"] = {
    "method": "common fractional offset, GLS with holography correlations; "
              "sigma_sys^2 = max(mu^2 - sigma_mu^2, 0) + (floor * sigma_mu)^2",
    "z_used": np.asarray(Z_SYS, dtype=float),
    "floor_fraction": SYS_FLOOR_FRACTION,
    "mcmc_tag": MCMC_TAG,
    **{
        par: {key: sys_offset_results[par][key] for key in
              ["mu", "sigma_mu", "sigma_sys", "detected", "chi2_zero", "chi2_offset", "r", "cov", "f_holo", "f_sim"]}
        for par in SYS_PARS
    },
}
sys_offset_file = "int_data_figs/legacy_emu/accel2_priors_sys_offset.npy"
np.save(sys_offset_file, sys_offset)
print(f"Saved {sys_offset_file}")
for par in SYS_PARS:
    res = sys_offset_results[par]
    print(f"{par}: sigma_sys = {100 * res['sigma_sys']:.2f}% (mu = {100 * res['mu']:+.2f} +- {100 * res['sigma_mu']:.2f}%, "
          f"{'detected' if res['detected'] else 'not detected, floor'})")

# %% [markdown]
# For curiosity only: the same estimate for all Arinyo parameters, in the
# article parametrization. It is saved to a separate file,
# `accel2_priors_sys_offset_all_arinyo.npy`, which the figures do not read;
# only $b_\delta$ and $\beta$ are propagated. `chi2(mu)` checks whether a single offset across redshifts
# describes the shifts: it has `len(z) - 1` degrees of freedom.

# %%
ARINYO_PARS = ["bias_delta", "beta", "q1", "q2", "k_v", "a_v", "b_v", "k_p"]
sys_offset_all = {}  # {"Z_SYS" or "all_z": {par: result}}
for label, z_list in [("Z_SYS", list(Z_SYS)), ("all_z", list(z_avail))]:
    sys_offset_all[label] = {"z_used": np.asarray(z_list, dtype=float)}
    print(f"\nz used: {z_list}   (floor = {SYS_FLOOR_FRACTION:.2f} sigma_mu)")
    print(f"  {'par':10s} {'mu':>8s} {'sig_mu':>7s} {'|mu|/sig':>8s} {'chi2(0)':>8s} {'chi2(mu)':>8s} {'sig_sys':>8s}")
    for par in ARINYO_PARS:
        inp = offset_inputs(par, z_list)
        res = common_offset(inp["r"], inp["cov"])
        sys_offset_all[label][par] = {**inp, **res}
        print(
            f"  {par:10s} {100 * res['mu']:+7.2f}% {100 * res['sigma_mu']:6.2f}% "
            f"{abs(res['mu']) / res['sigma_mu']:8.2f} {res['chi2_zero']:8.2f} {res['chi2_offset']:8.2f} "
            f"{100 * res['sigma_sys']:7.2f}%"
        )

# separate file, not read by the figures (they use accel2_priors_sys_offset.npy, b_delta and beta only)
sys_offset_all_file = "int_data_figs/legacy_emu/accel2_priors_sys_offset_all_arinyo.npy"
np.save(sys_offset_all_file, {"floor_fraction": SYS_FLOOR_FRACTION, "mcmc_tag": MCMC_TAG, **sys_offset_all})
print(f"\nSaved {sys_offset_all_file}")

# %%
