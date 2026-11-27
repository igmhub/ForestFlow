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
# # Smoothing of the ACCEL2 P1D mock
#
# Referee comment 2: the ACCEL2 mock P1D is smoothed with the LaCE `gpolyfit`
# prescription, a ForestFlow template (`ff_mpgcen.npy`, ForestFlow evaluated
# along the mpg-central IGM history) interpolated at the mock mean flux and
# multiplied by exp of a five-parameter function fitted to ACCEL2. Does this
# pull the mock toward the ForestFlow model?
#
# We compare, at z = 2.6, 3.0 and 3.6:
# * the raw ACCEL2 P1D, averaged over the three axes (Nyx file, `ACCEL2_6144_160`);
# * the LaCE smoothing used to build the mock, with the same routine and
#   emulator as the cup1d validation (`scripts/validation/val_sampler.py`, now only in
#   the cup1d git history:
#   `CH24_mpgcen_gpr`, `apply_smoothing=True`);
# * a template-free cubic spline with a few knots, fitted directly to ln P1D(ln k).
#
# The figure is in s/km, as the DESI measurements. Residuals are shown relative
# to the LaCE smoothing, together with the fractional DESI DR1 QMLE3 error
# (diagonal of the total covariance, the covariance used for the mock), over
# the wavenumbers used in the fit.

# %%
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
from scipy.interpolate import LSQUnivariateSpline

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

from lace.emulator.emulator_manager import set_emulator
from lace.emulator.smoothing_manager import apply_smoothing
from cup1d.theory import cosmology as cosmologies
from cup1d.p1ds.observations import data_DESIY1

# %%
Z_PLOT = [2.6, 3.0, 3.6]
NYX_FILE = os.path.join(
    os.environ.get("NYX_PATH", "/home/jchaves/Proyectos/projects/lya/data/nyx/"),
    "models_Nyx_Sept2025_include_Nyx_fid_rseed.hdf5",
)
SIM_NYX = "ACCEL2_6144_160"
EMULATOR_LABEL = "CH24_mpgcen_gpr"  # as in cup1d's former scripts/validation/val_sampler.py
COV_LABEL = "DESIY1_QMLE3"
# optional cross-check with the text files read by cup1d's Accel2_P1D (skipped if missing)
PATH_ACCEL2_TXT = "/home/jchaves/Proyectos/projects/lya/data/accel2/frontier_grid"

# %% [markdown]
# #### Raw ACCEL2 P1D from the Nyx file
#
# `k` is the bin centre in 1/Mpc and `Pk1d` the P1D in Mpc. The top-level
# `1d power` is the average over axes; the individual axes give a rough idea
# of the scatter.

# %%
raw = {}
with h5py.File(NYX_FILE, "r") as ff:
    for zz in Z_PLOT:
        grp = ff[f"{SIM_NYX}/redshift_{zz:.1f}/native_parameters"]
        p1d = np.array(grp["1d power"])
        axes = np.array(
            [np.array(grp[f"individual_axes/{ax}/1d power"])["Pk1d"] for ax in "xyz"]
        )
        raw[zz] = {
            "k_Mpc": p1d["k"],
            "p1d_Mpc": p1d["Pk1d"],
            "p1d_axes": axes,
            "Fbar": grp.attrs["Fbar"],
        }

# mean flux as in cup1d's Accel2_P1D (flux rescaled to this value in ACCEL2)
for zz in Z_PLOT:
    raw[zz]["mF"] = np.exp(-0.0025 * (1 + zz) ** 3.7)
    print(f"z={zz}: mF used = {raw[zz]['mF']:.4f}, Fbar in Nyx file = {raw[zz]['Fbar']:.4f}")

# %%
# cross-check: Nyx file vs the text files used by cup1d (axis average, k < 10/Mpc)
if os.path.isdir(PATH_ACCEL2_TXT):
    from cup1d.p1ds.synthetic.data_accel2 import load_data

    txt = load_data(PATH_ACCEL2_TXT)
    for zz in Z_PLOT:
        iz = int(np.argmin(np.abs(txt["z"] - zz)))
        p_nyx = np.interp(txt["k1d_Mpc"], raw[zz]["k_Mpc"], raw[zz]["p1d_Mpc"])
        rel = txt["p1d_Mpc"][iz] / p_nyx - 1
        print(f"z={zz}: max |txt/nyx - 1| = {np.max(np.abs(rel)):.2e}; "
              f"first k txt {txt['k1d_Mpc'][:2]}, nyx {raw[zz]['k_Mpc'][:2]}")
else:
    print("text files not found, skipping cross-check")

# %% [markdown]
# #### LaCE smoothing (the one used for the mock)
#
# Same call chain as cup1d: `BaseMockP1D.set_smoothing_Mpc` ->
# `lace.emulator.smoothing_manager.apply_smoothing`. For `gpolyfit` emulators it
# fits the five-parameter function over 0 < k < `emulator.kmax_Mpc`.

# %%
emulator = set_emulator(EMULATOR_LABEL)
print("emu_type:", emulator.emu_type, " kmax_Mpc:", emulator.kmax_Mpc)

list_data = [
    {"z": zz, "k_Mpc": raw[zz]["k_Mpc"].copy(), "p1d_Mpc": raw[zz]["p1d_Mpc"].copy(), "mF": raw[zz]["mF"]}
    for zz in Z_PLOT
]
apply_smoothing(emulator, list_data)
for ii, zz in enumerate(Z_PLOT):
    raw[zz]["p1d_lace"] = list_data[ii]["p1d_Mpc_smooth"]
    # the ForestFlow template alone, without the five-parameter correction:
    # interpolated at the mock mean flux exactly as inside apply_smoothing
    raw[zz]["p1d_template"] = np.interp(
        raw[zz]["k_Mpc"], emulator.input_norm["k_Mpc"], emulator.norm_imF(raw[zz]["mF"])
    )

# %% [markdown]
# #### DESI DR1 QMLE3 wavenumbers and errors
#
# As in `Accel2_P1D._load_p1d`: k in s/km converted to 1/Mpc with the ACCEL2
# cosmology, keeping only k above the smallest simulation k. Fractional error
# from the diagonal of the total covariance divided by the measured P1D.

# %%
desi = data_DESIY1.P1D_DESIY1(data_label=COV_LABEL)
sim_cosmo = cosmologies.set_cosmo(cosmo_label="accel2")

for zz in Z_PLOT:
    iz = int(np.argmin(np.abs(np.asarray(desi.z) - zz)))
    assert abs(desi.z[iz] - zz) < 1e-3, f"no DESI bin at z={zz}"
    dkms_dMpc = sim_cosmo.get_dkms_diMpc(zz)
    k_kms = np.asarray(desi.k_kms[iz])
    keep = k_kms * dkms_dMpc >= raw[zz]["k_Mpc"][raw[zz]["k_Mpc"] > 0].min()
    raw[zz]["dkms_dMpc"] = dkms_dMpc
    raw[zz]["desi_k_kms"] = k_kms[keep]
    raw[zz]["desi_sig_frac"] = (
        np.sqrt(np.diag(desi.cov_Pk_kms[iz]))[keep] / np.asarray(desi.Pk_kms[iz])[keep]
    )
    kmax_Mpc_desi = k_kms[keep].max() * dkms_dMpc
    if kmax_Mpc_desi > emulator.kmax_Mpc:
        print(f"z={zz}: DESI k_max = {kmax_Mpc_desi:.2f}/Mpc beyond the LaCE smoothing range")
    print(f"z={zz}: DESI k range {k_kms[keep].min():.4f}-{k_kms[keep].max():.4f} s/km, "
          f"sigma/P {raw[zz]['desi_sig_frac'].min():.3f}-{raw[zz]['desi_sig_frac'].max():.3f}")

# %% [markdown]
# #### Template-free spline
#
# Least-squares cubic spline of ln P1D against ln k with only a few interior
# knots, so it describes the measurement smoothly instead of following its
# noise. It is fitted to the raw ACCEL2 P1D over the wavenumbers used in the
# fit (the DESI range at each z, padded by `PAD_FIT` in ln k), with the
# interior knots evenly spaced in ln k. The k bins are linear, so each holds
# about the same number of modes and has similar fractional noise; uniform
# weights in ln P are therefore appropriate. `N_KNOTS_VAR` gives the
# robustness variants.

# %%

N_KNOTS = 3  # interior knots of the baseline spline
N_KNOTS_VAR = [2, 5]  # robustness variants
PAD_FIT = 0.1  # padding of the fit range in ln k, avoids edge effects at the DESI limits


def fit_spline(k_Mpc, p1d_Mpc, kmin_Mpc, kmax_Mpc, n_knots):
    """LSQ cubic spline of ln P(ln k) between kmin and kmax with n_knots interior knots."""
    pad = np.exp(PAD_FIT)
    mask = (k_Mpc >= kmin_Mpc / pad) & (k_Mpc <= kmax_Mpc * pad)
    xx, yy = np.log(k_Mpc[mask]), np.log(p1d_Mpc[mask])
    knots = np.linspace(xx[0], xx[-1], n_knots + 2)[1:-1]
    return LSQUnivariateSpline(xx, yy, knots, k=3), mask


for zz in Z_PLOT:
    d = raw[zz]
    kmin_Mpc = d["desi_k_kms"].min() * d["dkms_dMpc"]
    kmax_Mpc = d["desi_k_kms"].max() * d["dkms_dMpc"]
    d["p1d_spline_var"] = {}
    for nk in [N_KNOTS] + N_KNOTS_VAR:
        spl, mask = fit_spline(d["k_Mpc"], d["p1d_Mpc"], kmin_Mpc, kmax_Mpc, nk)
        pp = np.full_like(d["k_Mpc"], np.nan)
        pp[mask] = np.exp(spl(np.log(d["k_Mpc"][mask])))
        d["p1d_spline_var"][nk] = pp
    d["p1d_spline"] = d["p1d_spline_var"][N_KNOTS]

# %% [markdown]
# #### Figure
#
# Everything in s/km, as the DESI measurements:
# k_kms = k_Mpc / (dv/dx), P_kms = P_Mpc (dv/dx), so k P1D/pi is unchanged.
# Top: k P1D/pi at all redshifts (points raw, solid LaCE smoothing, dashed
# spline, dotted ForestFlow template alone, before the five-parameter
# correction). Below, one panel per redshift: each smoothing divided by the raw
# ACCEL2 measurement, with the fractional DESI DR1 QMLE3 error as a grey band,
# over the wavenumbers used in the fit. The spline variants are only in the
# printed summary below.

# %%
ftsize = 14
YLIM_RES = 0.03  # y range of the residual panels
fig, ax = plt.subplots(
    1 + len(Z_PLOT), 1, figsize=(7, 10), sharex=True,
    gridspec_kw={"height_ratios": [2.2] + [1] * len(Z_PLOT), "hspace": 0.05},
)

for jj, zz in enumerate(Z_PLOT):
    d = raw[zz]
    col = f"C{jj}"
    k_kms = d["k_Mpc"] / d["dkms_dMpc"]
    kmin, kmax = d["desi_k_kms"].min(), d["desi_k_kms"].max()
    in_fit = (k_kms >= kmin) & (k_kms <= kmax)
    kp = k_kms * d["p1d_Mpc"] * d["dkms_dMpc"] / np.pi  # = k_Mpc P_Mpc / pi

    ax[0].plot(k_kms[in_fit], kp[in_fit], ".", color=col, ms=3, alpha=0.6)
    ax[0].plot(k_kms[in_fit], (d["k_Mpc"] * d["p1d_lace"] / np.pi)[in_fit], "-", color=col, label=rf"$z={zz}$")
    ax[0].plot(k_kms[in_fit], (d["k_Mpc"] * d["p1d_spline"] / np.pi)[in_fit], "--", color="k", lw=1)
    ax[0].plot(k_kms[in_fit], (d["k_Mpc"] * d["p1d_template"] / np.pi)[in_fit], ":", color=col, lw=1.5)

    axr = ax[1 + jj]
    axr.fill_between(d["desi_k_kms"], -d["desi_sig_frac"], d["desi_sig_frac"], color="0.85", lw=0)
    axr.plot(k_kms[in_fit], (d["p1d_lace"] / d["p1d_Mpc"] - 1)[in_fit], "-", color=col, lw=1.2)
    axr.plot(k_kms[in_fit], (d["p1d_spline"] / d["p1d_Mpc"] - 1)[in_fit], "--", color="k", lw=1)
    axr.plot(k_kms[in_fit], (d["p1d_template"] / d["p1d_Mpc"] - 1)[in_fit], ":", color=col, lw=1.5)
    axr.axhline(0, color="0.4", lw=0.8)
    axr.set_ylim(-YLIM_RES, YLIM_RES)
    axr.text(0.02, 0.85, rf"$z={zz}$", transform=axr.transAxes, fontsize=ftsize - 2)
    axr.set_ylabel(r"$P_\mathrm{smooth}/P_\mathrm{ACCEL2}-1$", fontsize=ftsize - 2)

# legend entries for the line styles
ax[0].plot([], [], ".", color="0.4", label="ACCEL2")
ax[0].plot([], [], "-", color="0.4", label="LaCE smoothing")
ax[0].plot([], [], "--", color="k", lw=1, label=f"Spline ({N_KNOTS} knots)")
ax[0].plot([], [], ":", color="0.4", lw=1.5, label="ForestFlow template only")
ax[1].fill_between([], [], [], color="0.85", label="DESI DR1 error")
ax[0].legend(fontsize=ftsize - 4, ncol=2)
ax[1].legend(fontsize=ftsize - 4, loc="lower right")

ax[0].set_ylabel(r"$k_\parallel P_\mathrm{1D}/\pi$", fontsize=ftsize)
ax[0].set_xscale("log")
# few labelled ticks; the range is about one decade, so matplotlib labels the minor ones
X_TICKS = [0.003, 0.01, 0.03]  # s/km
ax[-1].xaxis.set_major_locator(FixedLocator(X_TICKS))
ax[-1].xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in X_TICKS]))
ax[-1].xaxis.set_minor_formatter(NullFormatter())
ax[-1].set_xlabel(r"$k_\parallel\,[\mathrm{s\,km^{-1}}]$", fontsize=ftsize)
for axx in ax:
    axx.tick_params(labelsize=ftsize - 2)
plt.tight_layout()
os.makedirs("figs", exist_ok=True)
plt.savefig("figs/accel2_p1d_smoothing.pdf")
plt.savefig("figs/accel2_p1d_smoothing.png")

# %%
# summary over the wavenumbers used in the fit: largest difference of each
# smoothing from the raw ACCEL2 P1D, in per cent and in units of the DESI DR1 error


def max_diff(d, num, den):
    """Max |num/den - 1| on the DESI k grid, in per cent and in DESI sigma."""
    k_kms = d["k_Mpc"] / d["dkms_dMpc"]
    ratio = num / den - 1
    ok = np.isfinite(ratio)
    diff = np.interp(d["desi_k_kms"], k_kms[ok], ratio[ok])
    return 100 * np.max(np.abs(diff)), np.max(np.abs(diff) / d["desi_sig_frac"])


for zz in Z_PLOT:
    print()
    d = raw[zz]
    rows = [("template/raw", d["p1d_template"], d["p1d_Mpc"]), ("LaCE/raw", d["p1d_lace"], d["p1d_Mpc"])]
    for nk in [N_KNOTS] + N_KNOTS_VAR:
        rows.append((f"spline{nk}/raw", d["p1d_spline_var"][nk], d["p1d_Mpc"]))
    for lab, num, den in rows:
        pc, nsig = max_diff(d, num, den)
        print(f"z={zz} {lab:14s} - 1: max {pc:5.2f}%, max |diff|/sigma {nsig:.2f}")

# %%
