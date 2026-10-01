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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Goodness of fit
# - Cosmic variance in fit
# - Goodness of model (fig 2)

# %%
# %load_ext autoreload
# %autoreload 2

import sys
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from forestflow.archive.gadget_archive import GadgetArchive3D
from forestflow.utils import transform_arinyo_params

# %% [markdown]
# ## LOAD P3D ARCHIVE

# %%
# %%time
# Use the corrected post-processing and attach its independently fitted
# parameters as ``arinyo_fixp3d`` to every loaded snapshot.
Archive3D = GadgetArchive3D(
    postproc="Cabayol23_fixp3d"
)
print(len(Archive3D.training_data))


# %% [markdown]
# ## LOAD SIMULATIONS

# %%
sim_label = "mpg_central"
central = Archive3D.get_testing_data(sim_label)

sim_label = "mpg_seed"
seed = Archive3D.get_testing_data(sim_label)

# The archive owns the identity-validated, mean-flux-consistent combination.
# It attaches ``arinyo_fixp3d`` after the combined fit file below has been
# produced by scripts/fit_p3d/fit_pflux.py.
list_merge = Archive3D.get_central_seed_average()
combo_fit = (
    Path(Archive3D.base_folder)
    / "data/best_arinyo/cabayol23_fixp3d/Arinyo_fit_mpg_central_seed.npy"
)
if not all("arinyo_fixp3d" in snapshot for snapshot in list_merge):
    raise FileNotFoundError(
        "The corrected central--seed combined fit is required. Run:\n"
        "python scripts/fit_p3d/fit_pflux.py mpg_central_seed --output "
        f"{combo_fit}"
    )

for snapshot in list_merge:
    snapshot["arinyo_fixp3d"] = snapshot["arinyo_fixp3d"]
zlist = [snapshot["z"] for snapshot in list_merge]

# %%
from forestflow.statistics.rebin_p3d import p3d_allkmu, get_P3D_k_mu_modes, rebin_P3D_Mpc_mode_weighted

n_mubins = 4
kmax_3d_fit = 5
kmax_1d_fit = 4
kmax_3d = kmax_3d_fit + 1
kmax_1d = kmax_1d_fit + 1

k3d_Mpc = central[0]['k3d_Mpc']
mu3d = central[0]['mu3d']
kmu_modes = get_P3D_k_mu_modes(kmax_3d)
mask_3d = k3d_Mpc[:, 0] <= kmax_3d
nk = np.sum(mask_3d)
mask_1d = central[0]['k_Mpc'] < kmax_1d
k1d_Mpc = central[0]['k_Mpc'][mask_1d]


# %%
from forestflow.utils import transform_arinyo_params

# %%
list_sims = [central, seed, list_merge]
nsims = len(list_sims)

p3d_measured = np.zeros((nsims, len(central), np.sum(mask_3d), n_mubins))
p3d_model = np.zeros((nsims, len(central), np.sum(mask_3d), n_mubins))
params = np.zeros((nsims, len(central), 3))

p1d_measured = np.zeros((nsims, len(central), np.sum(mask_1d)))
p1d_model = np.zeros((nsims, len(central), np.sum(mask_1d)))

for isnap in range(len(central)):
    z = central[isnap]["z"]

    for ii in range(nsims):
        sim = list_sims[ii]

        _ = rebin_P3D_Mpc_mode_weighted(
            k3d_Mpc[mask_3d],
            mu3d[mask_3d],
            sim[isnap]["p3d_Mpc"][mask_3d],
            kmu_modes,
            n_mu_bins=n_mubins,
        )
        knew, munew, p3d_measured[ii, isnap, ...], mu_bins = _
        p1d_measured[ii, isnap, :] = sim[isnap]["p1d_Mpc"][mask_1d]

        pp = sim[isnap]["arinyo_fixp3d"]
        model_p3d, plin = p3d_allkmu(
            sim[isnap]["model"],
            z,
            pp,
            kmu_modes,
            nk=nk,
            nmu=16,
            compute_plin=True,
        )
        _ = rebin_P3D_Mpc_mode_weighted(
            k3d_Mpc[:nk], mu3d[:nk], model_p3d[:nk], kmu_modes, n_mu_bins=n_mubins
        )
        knew, munew, rebin_model_p3d, mu_bins = _

        p3d_model[ii, isnap, ...] = rebin_model_p3d
        p1d_model[ii, isnap, :] = sim[isnap]["model"].P1D_Mpc(z, k1d_Mpc, parameters=pp)

        pp2 = transform_arinyo_params(pp, sim[isnap]["f_p"])

        params[ii, isnap, 0] = pp["bias"]
        params[ii, isnap, 1] = pp2["bias_eta"]
        params[ii, isnap, 2] = pp["beta"]

# %% [markdown]
# ### Impact of cosmic variance on fit
#
# Difference across z between best-fitting models to central and seed relative to their average

# %%
out = 3
folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"

for iz in range(len(central)):

    if central[iz]["z"] == out:
        pass
    else:
        continue

    jj = 0
    ftsize = 20
    fig, ax = plt.subplots(3, figsize=(8, 9))

    z_grid = np.array([d["z"] for d in central])

    lab = [r"$b_\delta$", r"$b_\eta$"]

    for ii in range(2):
        y = (params[0, :, ii] - params[1, :, ii]) / params[2, :, ii] / np.sqrt(2)
        print("param", ii, np.mean(y) * 100, np.std(y) * 100)
        ax[0].plot(z_grid, y, label=lab[ii], lw=3, alpha=0.8)

    for ii in range(3):
        ax[ii].axhline(0, linestyle=":", color="k")
        ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)
    ax[0].legend(loc="lower left", fontsize=ftsize, ncols=2)
    ax[0].set_xlabel(r"$z$", fontsize=ftsize)
    ax[0].set_ylabel(r"Residual parameter", fontsize=ftsize)
    ax[0].set_ylim(-0.05, 0.05)

    for ii in range(n_mubins):
        if ii == 0:
            lab = str(mu_bins[ii]) + r"$\leq\mu<$" + str(mu_bins[ii + 1])
        else:
            lab = str(mu_bins[ii]) + r"$\leq\mu\leq$" + str(mu_bins[ii + 1])
        col = f"C{ii}"
        x = knew[:, ii]
        _ = np.isfinite(x)
        y = (
            (p3d_model[0, iz, :, ii] - p3d_model[1, iz, :, ii])
            / p3d_model[2, iz, :, ii]
            / np.sqrt(2)
        )
        ax[1].plot(x[_], y[_], col + "-", lw=3, alpha=0.8, label=lab)

    _ = np.isfinite(knew)
    y = (p3d_model[0, iz, _] - p3d_model[1, iz, _]) / p3d_model[2, iz, _] / np.sqrt(2)
    res = np.percentile(y, [50, 16, 84])
    print("p3d", res[0] * 100, 0.5 * (res[2] - res[1]) * 100, np.std(y) * 100)

    x = k1d_Mpc
    y = (p1d_model[0, iz, :] - p1d_model[1, iz, :]) / p1d_model[2, iz, :] / np.sqrt(2)
    ax[2].plot(x, y, "C4-", lw=3)

    res = np.percentile(y, [50, 16, 84])
    print("p1d", res[0] * 100, 0.5 * (res[2] - res[1]) * 100, np.std(y) * 100)

    # ax[0].axhline(0, linestyle=":", color="k")
    # ax[0].axhline(0.1, linestyle="--", color="k")
    # ax[0].axhline(-0.1, linestyle="--", color="k")
    ax[1].axvline(kmax_3d_fit, linestyle="--", color="k")
    # ax[1].axhline(0, linestyle=":", color="k")
    # ax[1].axhline(0.01, linestyle="--", color="k")
    # ax[1].axhline(-0.01, linestyle="--", color="k")
    ax[2].axvline(kmax_1d_fit, linestyle="--", color="k")

    ax[1].set_ylabel(r"Residual $P_\mathrm{3D}$", fontsize=ftsize)
    ax[2].set_ylabel(r"Residual $P_\mathrm{1D}$", fontsize=ftsize)

    ax[1].set_xlabel(r"$k\, [\mathrm{Mpc}^{-1}]$", fontsize=ftsize)
    ax[2].set_xlabel(r"$k_\parallel\, [\mathrm{Mpc}^{-1}]$", fontsize=ftsize)

    ax[1].legend(fontsize=16, ncol=2, loc="lower left")

    if central[iz]["z"] != out:
        ax[0].set_title("z=" + str(central[iz]["z"]))
    ax[2].set_xscale("log")
    ax[1].set_ylim(-0.04, 0.03)
    ax[2].set_ylim(-0.0041, 0.0041)
    for jj in range(1, 3):
        ax[jj].set_xscale("log")
        ax[jj].set_xlim(right=7)

    plt.tight_layout()
    plt.savefig(folder + "cvar_fit_z_" + str(central[iz]["z"]) + ".png")
    plt.savefig(folder + "cvar_fit_z_" + str(central[iz]["z"]) + ".pdf")

# %% [markdown]
# Precision

# %%
kaiser = np.zeros((params.shape[0], params.shape[1], 2))
kaiser[:, :, 0] = params[:, :, 0]**2
kaiser[:, :, 1] = params[:, :, 0]**2*(1+params[:, :, 2])**2

for ii in range(2):
    y = (kaiser[0, :, ii] - kaiser[1, :, ii])/kaiser[2, :, ii]/np.sqrt(2)
    print(np.std(y)*100)

# %% [markdown]
# ### Save data for zenodo

# %%
for ii in range(len(central)):
    if(central[ii]["z"] == 3):
        iz = ii

out = {}

col = ["blue", "orange"]
for ii in range(len(col)):
    y = (params[0, :, ii] - params[1, :, ii])/params[2, :, ii]/np.sqrt(2)
    out["top_" + col[ii] + "_x"] = z_grid
    out["top_" + col[ii] + "_y"] = y

conv = {}
conv["blue"] = 0
conv["orange"] = 1
conv["green"] = 2
conv["red"] = 3
for key in conv.keys():
    ii = conv[key]

    out["center_" + key + "_x"] = knew[:, ii]
    y = (p3d_model[0, iz, :, ii] - p3d_model[1, iz, :, ii])/p3d_model[2, iz, :, ii]/np.sqrt(2)
    out["center_" + key + "_y"] = y

x = k1d_Mpc
y = (p1d_model[0, iz, :] - p1d_model[1, iz, :])/p1d_model[2, iz, :]/np.sqrt(2)
out["bottom_x"] = k1d_Mpc
out["bottom_y"] = y


# %%
import forestflow
path_forestflow = os.path.dirname(forestflow.__path__[0]) + "/"
folder = path_forestflow + "data/figures_machine_readable/"
np.save(folder + "figa2", out)

# %%
res = np.load(folder + "figa2.npy", allow_pickle=True).item()
res.keys()

# %% [markdown]
# ### Goodness of model to average of central and seed

# %%
out = 3
folder = "/home/jchaves/Proyectos/projects/lya/data/forestflow/figures/"


for iz in range(len(central)):

    if(central[iz]["z"] == out):
        pass
    else:
        continue

    jj = 0
    ftsize = 20
    fig, ax = plt.subplots(2, figsize=(8, 6), sharex=True)

    for ii in range(n_mubins):
        col = f"C{ii}"
        x = knew[:, ii]
        _ = np.isfinite(x)
        y = (p3d_measured[2, iz, :, ii] - p3d_model[2, iz, :, ii])/p3d_model[2, iz, :, ii]
        ax[0].plot(x[_], y[_], col+"-", lw=3, alpha=0.8)

    x = k1d_Mpc
    y = (p1d_measured[2, iz, :] - p1d_model[2, iz, :])/p1d_model[2, iz, :]
    ax[1].plot(x, y, "C4-", lw=3)


    ax[0].axhline(0, linestyle=":", color="k")
    ax[0].axhline(0.1, linestyle="--", color="k")
    ax[0].axhline(-0.1, linestyle="--", color="k")
    ax[0].axvline(kmax_3d_fit, linestyle="--", color="k")
    ax[1].axhline(0, linestyle=":", color="k")
    ax[1].axhline(0.01, linestyle="--", color="k")
    ax[1].axhline(-0.01, linestyle="--", color="k")
    ax[1].axvline(kmax_1d_fit, linestyle="--", color="k")

    ax[0].set_ylabel(r"Residual $P_\mathrm{3D}$", fontsize=ftsize)
    ax[1].set_ylabel(r"Residual $P_\mathrm{1D}$", fontsize=ftsize)

    ax[0].set_xlabel(r"$k\, [\mathrm{Mpc}^{-1}]$", fontsize=ftsize)
    ax[1].set_xlabel(r"$k_\parallel\, [\mathrm{Mpc}^{-1}]$", fontsize=ftsize)

    ax[0].tick_params(axis="both", which="major", labelsize=ftsize)
    ax[1].tick_params(axis="both", which="major", labelsize=ftsize)

    if(central[iz]["z"] != out):
        ax[0].set_title("z="+str(central[iz]["z"]))
    ax[0].set_xscale("log")
    ax[0].set_ylim(-0.21, 0.21)
    ax[1].set_ylim(-0.021, 0.021)

    plt.tight_layout()
    plt.savefig(folder + "goodness_fit_z_"+str(central[iz]["z"])+".png")
    plt.savefig(folder + "goodness_fit_z_"+str(central[iz]["z"])+".pdf")

# %% [markdown]
# ## Goodness of corrected Arinyo fits across all training simulations
#
# Each of the 30 MP-Gadget hypercube simulations has 55 averaged snapshots
# (eleven redshifts and five optical-depth rescalings).  The archive attaches
# the completed corrected-postprocessing fits as ``arinyo_fixp3d``.  Below we
# evaluate those saved parameters; this cell does **not** run a minimization.
#
# P3D predictions use the fitter's hybrid finite-volume average, exactly as in
# the fits: sparse large-scale cells are evaluated at their discrete Fourier
# modes and dense cells use the phase-space continuous average.  We then
# combine the native 16 mu cells into four broad bins with their mode counts.

# %%
from forestflow.model_fits import ArinyoFitter
from forestflow.statistics.rebin_p3d import rebin_P3D_Mpc_mode_weighted

list_sims = Archive3D.training_data
simulation_labels = sorted({snapshot["sim_label"] for snapshot in list_sims})
assert len(simulation_labels) == 30
assert all("arinyo_fixp3d" in snapshot for snapshot in list_sims)
print(
    f"Evaluating {len(list_sims)} corrected fits from "
    f"{len(simulation_labels)} training simulations."
)

fitter = ArinyoFitter(
    kmin_3d=0.01,
    kmax_3d=4.5,
    kmin_1d=0.01,
    kmax_1d=6.0,
)
n_mubins = 4
p3d_model_all = []
p3d_measured_all = []
p1d_model_all = []
p1d_measured_all = []
knew = munew = mu_bins = None
k1d_iMpc = None

for index, simulation in enumerate(list_sims):
    if index % 100 == 0:
        print(f"{index}/{len(list_sims)}")

    fitter.prepare_simulation(simulation, is_mpg=True)
    fit_parameters = fitter.params_from_dict(simulation["arinyo_fixp3d"])
    model_p3d, model_p1d = fitter.predict(fit_parameters)

    rebinned_measurement = rebin_P3D_Mpc_mode_weighted(
        fitter.data.k3d,
        fitter.data.mu3d,
        fitter.data.p3d,
        fitter._mpg_k_mu_modes,
        n_mu_bins=n_mubins,
    )
    rebinned_model = rebin_P3D_Mpc_mode_weighted(
        fitter.data.k3d,
        fitter.data.mu3d,
        model_p3d,
        fitter._mpg_k_mu_modes,
        n_mu_bins=n_mubins,
    )
    current_knew, current_munew, measured_p3d, current_mu_bins = rebinned_measurement
    _, _, fitted_p3d, _ = rebinned_model

    if knew is None:
        knew, munew, mu_bins = current_knew, current_munew, current_mu_bins
        k1d_iMpc = fitter.data.k1d.copy()
    else:
        assert np.allclose(knew, current_knew, equal_nan=True)
        assert np.allclose(munew, current_munew, equal_nan=True)
        assert np.allclose(k1d_iMpc, fitter.data.k1d)

    p3d_model_all.append(fitted_p3d)
    p3d_measured_all.append(measured_p3d)
    p1d_model_all.append(model_p1d)
    p1d_measured_all.append(fitter.data.p1d.copy())

p3d_model = np.asarray(p3d_model_all)
p3d_measured = np.asarray(p3d_measured_all)
p1d_model = np.asarray(p1d_model_all)
p1d_measured = np.asarray(p1d_measured_all)

# %%
ftsize = 15
fig, axes = plt.subplots(2, figsize=(8, 6), sharex=False)

for mu_index in range(n_mubins):
    label = (
        rf"${mu_bins[mu_index]:.2f} \leq \mu "
        + (rf"\leq {mu_bins[mu_index + 1]:.2f}$" if mu_index == n_mubins - 1
           else rf"< {mu_bins[mu_index + 1]:.2f}$")
    )
    valid = np.isfinite(knew[:, mu_index])
    residual = np.divide(
        p3d_model[:, :, mu_index],
        p3d_measured[:, :, mu_index],
        out=np.full_like(p3d_model[:, :, mu_index], np.nan),
        where=p3d_measured[:, :, mu_index] != 0,
    ) - 1.0
    percentile = np.nanpercentile(residual, [16, 50, 84], axis=0)
    axes[0].plot(knew[valid, mu_index], percentile[1, valid], lw=2, label=label)
    axes[0].fill_between(
        knew[valid, mu_index], percentile[0, valid], percentile[2, valid], alpha=0.2
    )

p1d_residual = p1d_model / p1d_measured - 1.0
p1d_percentile = np.nanpercentile(p1d_residual, [16, 50, 84], axis=0)
axes[1].plot(k1d_iMpc, p1d_percentile[1], color="C4", lw=2)
axes[1].fill_between(k1d_iMpc, p1d_percentile[0], p1d_percentile[2], color="C4", alpha=0.2)

for axis, scale in zip(axes, (4.5, 6.0)):
    axis.axhline(0.0, color="k", ls=":")
    axis.axvline(scale, color="k", ls="--", label="fit scale cut")
    axis.set_xscale("log")
    axis.grid(alpha=0.25)

axes[0].set(
    ylabel=r"$P_\mathrm{3D}^\mathrm{fit}/P_\mathrm{3D}^\mathrm{data}-1$",
    xlabel=r"$k\,[\mathrm{Mpc}^{-1}]$",
)
axes[1].set(
    ylabel=r"$P_\mathrm{1D}^\mathrm{fit}/P_\mathrm{1D}^\mathrm{data}-1$",
    xlabel=r"$k_\parallel\,[\mathrm{Mpc}^{-1}]$",
)
axes[0].legend(fontsize=10, ncol=2)
for axis in axes:
    axis.tick_params(labelsize=ftsize)
fig.tight_layout()

# %%
# Optional portable summary for the paper-figure workflow.
save_goodness_summary = False
if save_goodness_summary:
    output = Path("corrected_arinyo_goodness_all_training.npz")
    np.savez(
        output,
        k3d_iMpc=knew,
        mu=munew,
        mu_edges=mu_bins,
        P3D_model_Mpc=p3d_model,
        P3D_data_Mpc=p3d_measured,
        k1d_iMpc=k1d_iMpc,
        P1D_model_Mpc=p1d_model,
        P1D_data_Mpc=p1d_measured,
    )
    print(f"Saved {output.resolve()}")

# %% [markdown]
# Precision

# %%
_ = np.isfinite(knew) & (knew > 0.5) & (knew < 5)
rat = p3d_model[:, _]/p3d_measured[:, _]- 1
y = np.nanpercentile(rat, [50, 16, 84])
print(y[0]*100, 0.5*(y[2]-y[1])*100, np.nanstd(rat)*100)

# %%
_ = np.isfinite(k1d_iMpc) & (k1d_iMpc < 4)
rat = p1d_model[:, _]/p1d_measured[:, _] - 1
y = np.nanpercentile(rat, [50, 16, 84])
print(y[0]*100, 0.5*(y[2]-y[1])*100, np.nanstd(rat)*100)

# %% [markdown]
# ### Save data for zenodo

# %%

out = {}

conv = {}
conv["blue"] = 0
conv["orange"] = 1
conv["green"] = 2
conv["red"] = 3
for key in conv.keys():
    ii = conv[key]

    out["top_" + key + "_x"] = knew[:, ii]
    y = np.nanpercentile(p3d_model[:, :, ii]/p3d_measured[:, :, ii], [50, 16, 84], axis=0) - 1
    out["top_" + key + "_y"] = y[0]

x = k1d_iMpc
y = np.nanpercentile(p1d_model/p1d_measured, [50, 16, 84], axis=0) - 1
out["bottom_x"] = k1d_iMpc
out["bottom_y"] = y[0]


# %%
import forestflow
path_forestflow = os.path.dirname(forestflow.__path__[0]) + "/"
folder = path_forestflow + "data/figures_machine_readable/"
np.save(folder + "fig2", out)

# %%
res = np.load(folder + "fig2.npy", allow_pickle=True).item()
res.keys()

# %%
