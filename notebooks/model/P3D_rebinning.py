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
# # Comparing P3D predictions to finite-volume measurements
#
# A simulation P3D bin is an average over a finite set of Fourier modes.  We
# compare three model predictions, in increasing physical fidelity:
#
# 1. **Bin-centre evaluation:** evaluate at the measured `(k, mu)` centre.  If
#    unavailable, it is naturally estimated from the measured grid.
# 2. **Continuous phase-space average:** evaluate eight times more finely in
#    each geometric bin and average with the 3D Fourier measure.  Edges can be
#    supplied or inferred from centres.
# 3. **Exact mode average:** evaluate every discrete Fourier mode allowed by the
#    simulation volume and average them.  This is the reference prediction for
#    a finite-volume measurement.
#
# The second part repeats this comparison after combining the data into broader
# mu bins.  Padded NaN cells are retained in their original grid positions.

# %%
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
import numpy as np

from lace.cosmo import cosmology
from forestflow.archive.gadget_archive import GadgetArchive3D
from forestflow.model.arinyo import ArinyoModel
from forestflow.statistics.p3d import (
    P3D_Mpc_k_mu_bin_averaged,
    P3D_Mpc_k_mu_hybrid_averaged,
    P3D_Mpc_k_mu_mode_averaged,
)
from forestflow.statistics.rebin_p3d import (
    get_P3D_k_mu_bin_edges,
    get_P3D_k_mu_modes,
    rebin_P3D_Mpc_mode_weighted,
)

# %%
archive = GadgetArchive3D()
simulation = archive.training_data[6]
k_max_iMpc = 4.0
n_mu_bins_broad = 4
max_discrete_modes = 256
fine_factor = 4

k_iMpc_native = simulation["k3d_Mpc"]
mu_native = simulation["mu3d"]
P3D_data_native_Mpc = simulation["p3d_Mpc"]
k_mask = np.isfinite(k_iMpc_native[:, 0]) & (k_iMpc_native[:, 0] <= k_max_iMpc)
k_iMpc_native = k_iMpc_native[k_mask]
mu_native = mu_native[k_mask]
P3D_data_native_Mpc = P3D_data_native_Mpc[k_mask]

model = ArinyoModel(cosmology.Cosmology(simulation["cosmo_params"]))
linear = model.linear.get_linear_theory(simulation["z"])
# Use the post-processing bin definition, never edges inferred from measured
# mode-weighted centres.  The data scale cut acts on those centres, so retain
# exactly one more native edge than retained data rows.
k_edges_all_iMpc, mu_edges_native = archive.get_P3D_k_mu_bin_edges()
k_edges_iMpc = k_edges_all_iMpc[: k_iMpc_native.shape[0] + 1]
k_mu_modes_native = get_P3D_k_mu_modes(k_max_iMpc)

# %% [markdown]
# ## Native simulation bins

# %%
# 1. Values at the measured mode-weighted bin centres.
P3D_centre_native_Mpc = model.P3D_Mpc_k_mu(
    linear, simulation["z"], k_iMpc_native, mu_native, simulation["Arinyo_min"]
)

# 2. Continuous eightfold phase-space average over native post-processing edges.
# Padded archive cells are retained in the centre prediction above; no centres
# are needed here because the bin definition is explicit.
P3D_continuous_native_Mpc = P3D_Mpc_k_mu_bin_averaged(
    linear,
    simulation["z"],
    P3D_model=model.P3D_Mpc_k_mu,
    P3D_params=simulation["Arinyo_min"],
    k_iMpc_edges=k_edges_iMpc,
    mu_edges=mu_edges_native,
    fine_factor=fine_factor,
)

# 3. Exact finite-volume mode average.  The data centres are used only to
# preserve the native output shape; exact k and mu come from k_mu_modes_native.
P3D_modes_native_Mpc = P3D_Mpc_k_mu_mode_averaged(
    linear,
    simulation["z"],
    P3D_model=model.P3D_Mpc_k_mu,
    P3D_params=simulation["Arinyo_min"],
    k_mu_modes=k_mu_modes_native,
    k_iMpc_edges=k_edges_iMpc,
    mu_edges=mu_edges_native,
)

# 4. Hybrid prediction: exact lattice modes in sparse bins and the continuous
# phase-space average in bins with more than ``max_discrete_modes`` modes.
P3D_hybrid_native_Mpc = P3D_Mpc_k_mu_hybrid_averaged(
    linear,
    simulation["z"],
    P3D_model=model.P3D_Mpc_k_mu,
    P3D_params=simulation["Arinyo_min"],
    k_mu_modes=k_mu_modes_native,
    k_iMpc_edges=k_edges_iMpc,
    mu_edges=mu_edges_native,
    max_discrete_modes=max_discrete_modes,
    fine_factor=fine_factor,
)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
for mu_index in range(k_iMpc_native.shape[1]):
    finite = np.isfinite(P3D_modes_native_Mpc[:, mu_index])
    if not np.any(finite):
        continue
    color = f"C{mu_index % 10}"
    axes[0].plot(k_iMpc_native[finite, mu_index], P3D_modes_native_Mpc[finite, mu_index], color=color)
    axes[0].plot(k_iMpc_native[finite, mu_index], P3D_centre_native_Mpc[finite, mu_index], ".", color=color)
    axes[0].plot(k_iMpc_native[finite, mu_index], P3D_continuous_native_Mpc[finite, mu_index], "--", color=color)
    axes[0].plot(k_iMpc_native[finite, mu_index], P3D_hybrid_native_Mpc[finite, mu_index], ":", color=color)
    axes[1].plot(k_iMpc_native[finite, mu_index], P3D_centre_native_Mpc[finite, mu_index] / P3D_modes_native_Mpc[finite, mu_index] - 1, ".", color=color)
    axes[1].plot(k_iMpc_native[finite, mu_index], P3D_continuous_native_Mpc[finite, mu_index] / P3D_modes_native_Mpc[finite, mu_index] - 1, "--", color=color)
    axes[1].plot(k_iMpc_native[finite, mu_index], P3D_hybrid_native_Mpc[finite, mu_index] / P3D_modes_native_Mpc[finite, mu_index] - 1, "-", color=color)
axes[0].plot([], [], "k-", label="exact discrete-mode average")
axes[0].plot([], [], "k.", label="bin-centre evaluation")
axes[0].plot([], [], "k--", label="continuous 8x phase-space average")
axes[0].plot([], [], "k:", label=rf"hybrid ($N_{{\rm modes}}\leq {max_discrete_modes}$ exact)")
axes[0].set(xscale="log", yscale="log", xlabel=r"$k$ [Mpc$^{-1}$]", ylabel=r"$P_{3D}$ [Mpc$^3$]")
axes[0].legend(fontsize=8)
axes[1].axhline(0.0, color="k", lw=0.8)
axes[1].set(xscale="log", xlabel=r"$k$ [Mpc$^{-1}$]", ylabel="approximation / exact-mode - 1", ylim=[-0.2, 0.2])

fig.tight_layout()

# %%
# %%time
for ii in range(10):
    P3D_continuous_native_Mpc = P3D_Mpc_k_mu_bin_averaged(
        linear,
        simulation["z"],
        P3D_model=model.P3D_Mpc_k_mu,
        P3D_params=simulation["Arinyo_min"],
        k_iMpc_edges=k_edges_iMpc,
        mu_edges=mu_edges_native,
        fine_factor=fine_factor,
    )

# %%
# %%time
for ii in range(10):
    P3D_modes_native_Mpc = P3D_Mpc_k_mu_mode_averaged(
        linear,
        simulation["z"],
        P3D_model=model.P3D_Mpc_k_mu,
        P3D_params=simulation["Arinyo_min"],
        k_mu_modes=k_mu_modes_native,
        k_iMpc_edges=k_edges_iMpc,
        mu_edges=mu_edges_native,
    )

# %%
# %%time
for ii in range(10):
    P3D_hybrid_native_Mpc = P3D_Mpc_k_mu_hybrid_averaged(
        linear,
        simulation["z"],
        P3D_model=model.P3D_Mpc_k_mu,
        P3D_params=simulation["Arinyo_min"],
        k_mu_modes=k_mu_modes_native,
        k_iMpc_edges=k_edges_iMpc,
        mu_edges=mu_edges_native,
        max_discrete_modes=max_discrete_modes,
        fine_factor=fine_factor,
    )

# %% [markdown]
# ## Broader mu bins
#
# The broader measured bins retain the finite-volume weighting.  The exact
# native mode prediction can therefore be reweighted directly.  For the
# continuous approximation, the known broad mu edges define the geometry;
# for the centre approximation, use the measured broad-bin centres.

# %%
k_iMpc_broad, mu_broad, P3D_data_broad_Mpc, mu_edges_broad = rebin_P3D_Mpc_mode_weighted(
    k_iMpc_native,
    mu_native,
    P3D_data_native_Mpc,
    k_mu_modes_native,
    n_mu_bins=n_mu_bins_broad,
)
_, _, P3D_modes_broad_Mpc, _ = rebin_P3D_Mpc_mode_weighted(
    k_iMpc_native,
    mu_native,
    P3D_modes_native_Mpc,
    k_mu_modes_native,
    n_mu_bins=n_mu_bins_broad,
)

# The broader bins have their own exact lattice assignment.  This provides the
# same sparse-bin correction directly on the coarsened measurement geometry.
k_mu_modes_broad = get_P3D_k_mu_modes(k_max_iMpc, n_mu_bins=n_mu_bins_broad)
P3D_hybrid_broad_Mpc = P3D_Mpc_k_mu_hybrid_averaged(
    linear,
    simulation["z"],
    P3D_model=model.P3D_Mpc_k_mu,
    P3D_params=simulation["Arinyo_min"],
    k_mu_modes=k_mu_modes_broad,
    k_iMpc_edges=k_edges_iMpc,
    mu_edges=mu_edges_broad,
    max_discrete_modes=max_discrete_modes,
    fine_factor=fine_factor,
)

# 1. Evaluate at the measured broad-bin centres; padded NaNs stay in place.
P3D_centre_broad_Mpc = model.P3D_Mpc_k_mu(
    linear, simulation["z"], k_iMpc_broad, mu_broad, simulation["Arinyo_min"]
)

# 2. Continuous average over the known broad bin geometry.
mu_centres_broad = 0.5 * (mu_edges_broad[:-1] + mu_edges_broad[1:])
P3D_continuous_broad_Mpc = P3D_Mpc_k_mu_bin_averaged(
    linear,
    simulation["z"],
    None,
    None,
    model.P3D_Mpc_k_mu,
    simulation["Arinyo_min"],
    k_iMpc_edges=k_edges_iMpc,
    mu_edges=mu_edges_broad,
    fine_factor=fine_factor,
)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
for mu_index in range(n_mu_bins_broad):
    finite = np.isfinite(P3D_modes_broad_Mpc[:, mu_index])
    color = f"C{mu_index}"
    axes[0].plot(k_iMpc_broad[finite, mu_index], P3D_modes_broad_Mpc[finite, mu_index], color=color)
    axes[0].plot(k_iMpc_broad[finite, mu_index], P3D_centre_broad_Mpc[finite, mu_index], ".", color=color)
    axes[0].plot(k_iMpc_broad[finite, mu_index], P3D_continuous_broad_Mpc[finite, mu_index], "--", color=color)
    axes[0].plot(k_iMpc_broad[finite, mu_index], P3D_hybrid_broad_Mpc[finite, mu_index], ":", color=color)
    axes[1].plot(k_iMpc_broad[finite, mu_index], P3D_centre_broad_Mpc[finite, mu_index] / P3D_modes_broad_Mpc[finite, mu_index] - 1, ".", color=color)
    axes[1].plot(k_iMpc_broad[finite, mu_index], P3D_continuous_broad_Mpc[finite, mu_index] / P3D_modes_broad_Mpc[finite, mu_index] - 1, "--", color=color)
    axes[1].plot(k_iMpc_broad[finite, mu_index], P3D_hybrid_broad_Mpc[finite, mu_index] / P3D_modes_broad_Mpc[finite, mu_index] - 1, "-", color=color)
axes[0].plot([], [], "k-", label="exact discrete-mode average")
axes[0].plot([], [], "k.", label="broad-bin centre")
axes[0].plot([], [], "k--", label="continuous 8x phase-space average")
axes[0].plot([], [], "k:", label=rf"hybrid ($N_{{\rm modes}}\leq {max_discrete_modes}$ exact)")
axes[0].set(xscale="log", yscale="log", xlabel=r"$k$ [Mpc$^{-1}$]", ylabel=r"$P_{3D}$ [Mpc$^3$]")
axes[0].legend(fontsize=8)
axes[1].axhline(0.0, color="k", lw=0.8)
axes[1].set(xscale="log", xlabel=r"$k$ [Mpc$^{-1}$]", ylabel="approximation / exact-mode - 1", ylim=[-0.1, 0.1])
fig.tight_layout()

# %%
