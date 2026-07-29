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
# # Perform Arinyo fit to Astrid data
#
# Using a dedicated class at the bottom!!

# %%
# %load_ext autoreload
# %autoreload 2

import os
import matplotlib.pyplot as plt
import numpy as np
from forestflow.model_p3d_arinyo import ArinyoModel
from forestflow.archive import GadgetArchive3D
from lace.cosmo import cosmology
from scipy.stats import binned_statistic_2d

# %%
Archive3D = GadgetArchive3D()

# %%
kmin_1d_fit = 0.3
kmax_1d_fit = 7.

kmin_3d_fit = 0.7
kmax_3d_fit = 4.5

zlist = np.arange(2.0, 4.501, 0.25)
zlist


# %%
def get_err_p1d(x, alpha=4, xmin=kmin_1d_fit, xmax=kmax_1d_fit, ymin=1, ymax=4):
    t = (x - xmin) / (xmax - xmin)
    return ymin + ymax * t**alpha
plt.plot(k1d_Mpc, get_err_p1d(k1d_Mpc))


# %%
def get_err_p3d(x, alpha=0.2, xmin=kmin_3d_fit, xmax=kmax_3d_fit,
                ymin=3, ymax=20):
    t = (x - xmin) / (xmax - xmin)
    return ymax - (ymax - ymin) * t**alpha

plt.plot(k3d_Mpc[:,0], get_err_p3d(k3d_Mpc)[:,0])

# %%
sim = Archive3D.training_data[567]

z = sim["z"]
print(z)
k3d_Mpc = sim["k3d_Mpc"]
mu3d = sim["mu3d"]
p3d_Mpc = sim["p3d_Mpc"]
mask_3d = (k3d_Mpc[:, 0] >= kmin_3d_fit) & (k3d_Mpc[:, 0] < kmax_3d_fit)
k3d_Mpc = k3d_Mpc[mask_3d]
mu3d = mu3d[mask_3d]
p3d_Mpc = p3d_Mpc[mask_3d]
std_p3d = get_err_p3d(k3d_Mpc) * 0.01

mask_1d = (sim["k_Mpc"] >= kmin_1d_fit) & (sim["k_Mpc"] < kmax_1d_fit)
k1d_Mpc = sim["k_Mpc"][mask_1d]
p1d_Mpc = sim["p1d_Mpc"][mask_1d]
std_p1d = get_err_p1d(k1d_Mpc) * 0.01


# %%
n_k_bins = 20
k_Mpc_max = 20.0
n_mu_bins = 16
boxsize = 67.5

lnk_max = np.log(k_Mpc_max)
lnk_min = np.log(2.0 * np.pi / boxsize)
lnk_bin_max = lnk_max + (lnk_max - lnk_min) / (n_k_bins - 1)
lnk_bin_edges = np.linspace(lnk_min, lnk_bin_max, n_k_bins + 1)

nrkbin = 12
nrmubin = 8
k_bin_edges = np.exp(lnk_bin_edges)
mu_bin_edges = np.linspace(0.0, 1.0, n_mu_bins + 1)

ind = np.argwhere((k_bin_edges >= kmin_3d_fit) & (k_bin_edges < kmax_3d_fit))[:,0]
k_bin_edges_masked = k_bin_edges[ind[0]-1:ind[-1]+2]

fine_k = np.geomspace(k_bin_edges[ind[0]-1], k_bin_edges[ind[-1]+1], k_bin_edges_masked.shape[0] * nrkbin)
fine_mu = np.linspace(mu_bin_edges.min(), mu_bin_edges.max(), n_mu_bins * nrmubin)

fine2D_k, fine2D_mu = np.meshgrid(fine_k, fine_mu, indexing="ij")

# %%
k_mean, _, _, _ = binned_statistic_2d(
    fine2D_k.ravel(),
    fine2D_mu.ravel(),
    fine2D_k.ravel(),
    statistic="mean",
    bins=[k_bin_edges_masked, mu_bin_edges],
)

mu_mean, _, _, _ = binned_statistic_2d(
    fine2D_k.ravel(),
    fine2D_mu.ravel(),
    fine2D_mu.ravel(),
    statistic="mean",
    bins=[k_bin_edges_masked, mu_bin_edges],
)


# fine_mask_3d = (k_mean[:, 0] >= kmin_3d_fit) & (k_mean[:, 0] < kmax_3d_fit)

# %%
fid_cosmo = cosmology.Cosmology(sim["cosmo_params"])
model_Arinyo = ArinyoModel(fid_cosmo)
linear = model_Arinyo.linear_theory(zlist)

# %%
ari_par = {
    'bias': -0.18,
    'bias_eta': -0.3,
    'q1': 0.4,
    'q2': 0.0,
    'kvav': 0.58,
    'av': 0.29,
    'bv': 1.55,
    'kp': 10.5
}

p3d_model = model_Arinyo.P3D_Mpc_k_mu(linear, z, k3d_Mpc, mu3d, ari_par)
p1d_model = model_Arinyo.P1D_Mpc(linear, z, k1d_Mpc, ari_par)

# %%
min_data = {}

min_data["z"] = z
min_data["linear"] = linear
min_data["k3d_Mpc"] = k3d_Mpc
min_data["mu3d"] = mu3d
min_data["target_p3d_Mpc"] = p3d_Mpc
min_data["std_p3d"] = std_p3d
min_data["k1d_Mpc"] = k1d_Mpc
min_data["target_p1d_Mpc"] = p1d_Mpc
min_data["std_p1d"] = std_p1d
min_data["power_model"] = model_Arinyo

min_data["mod_k3D_Mpc"] = k_mean
min_data["mod_mu3D"] = mu_mean
min_data["mod_fine2D_k"] = fine2D_k
min_data["mod_fine2D_mu"] = fine2D_mu
min_data["edge_k"] = k_bin_edges_masked
min_data["edge_mu"] = mu_bin_edges


def chi2(arr_params, data, err_tol_3D=0.05):

    ari_par = {
        "bias": arr_params[0],
        "bias_eta": arr_params[1],
        "q1": arr_params[2],
        "q2": arr_params[3],
        "kvav": arr_params[4],
        "av": arr_params[5],
        "bv": arr_params[6],
        "kp": arr_params[7],
    }

    fine_p3d_model = data["power_model"].P3D_Mpc_k_mu(
        data["linear"], data["z"], data["mod_fine2D_k"], data["mod_fine2D_mu"], ari_par
    )

    p3d_model, _, _, _ = binned_statistic_2d(
        data["mod_fine2D_k"].ravel(),
        data["mod_fine2D_mu"].ravel(),
        fine_p3d_model.ravel(),
        statistic="mean",
        bins=[data["edge_k"], data["edge_mu"]],
    )

    # p3d_model = data["power_model"].P3D_Mpc_k_mu(
    #     data["linear"], data["z"], data["k3d_Mpc"], data["mu3d"], ari_par
    # )
    p1d_model = data["power_model"].P1D_Mpc(
        data["linear"], data["z"], data["k1d_Mpc"], ari_par
    )

    chi2_3D = np.nanmean(
        (p3d_model / data["target_p3d_Mpc"] - 1) ** 2 / data["std_p3d"] ** 2
    )
    chi2_1D = np.nanmean(
        (p1d_model / data["target_p1d_Mpc"] - 1) ** 2 / data["std_p1d"] ** 2
    )

    chi2_tot = chi2_3D + chi2_1D

    # print(chi2_3D, chi2_1D)

    return chi2_tot


# %%
ari_par = {
    "bias": -0.18,
    "bias_eta": -0.3,
    "q1": 0.4,
    "q2": 0.0,
    "kvav": 0.58,
    "av": 0.29,
    "bv": 1.55,
    "kp": 10.5,
}

arr_par = np.zeros(len(ari_par.keys()))
for ii, param in enumerate(ari_par):
    arr_par[ii] = sim["Arinyo_min"][param]
chi2(arr_par, min_data)

# %%
from scipy.optimize import minimize

bounds = [
    (-1.0, -0.01),   # bias
    (-0.5, -0.01),    # bias_eta
    (0.0, 5.0),    # q1
    (-2.0, 2.0),    # q2
    (0.2, 15.5),  # kvav
    (0., 2.),    # av
    (1.0, 5.0),    # bv
    (4., 50.),   # kp
]


# %%

# %%
# import cma

# sigma0 = 0.1

# x0 = arr_par.copy()

# lower = [b[0] for b in bounds]
# upper = [b[1] for b in bounds]

# es = cma.CMAEvolutionStrategy(
#     x0,
#     sigma0,  # initial step size
#     {
#         "bounds": [lower, upper],
#         "verb_disp": 1,
#     },
# )

# es.optimize(lambda x: chi2(x, min_data))

# best_params = es.result.xbest
# best_chi2 = es.result.fbest

# print(best_params, best_chi2)


# %%

print(best_params, best_chi2)

# %%
x0 = best_params

# %%
ftol = 0.01
for ii in range(20):

    if ii == 0:
        x0 = arr_par.copy()
        method = "L-BFGS-B"
        maxfev = 500
    else:
        x0 = result.x
        method = "Nelder-Mead"
        maxfev = 1000

    result = minimize(
        chi2,
        x0,
        args=(min_data,),
        method=method,
        bounds=bounds,
        options={
            "disp": True,
            "maxiter": maxfev,
            "maxfev": maxfev,
            "fatol": ftol,
            "xatol": 1e-5,
        },
    )

    print(result.success)
    print(result.message)
    print(result.fun)
    print(result.x)

    if ii != 0:
        if chi2_min - result.fun < ftol:
            break
        else:
            chi2_min = result.fun
    else:
        chi2_min = result.fun

    print(ii, chi2_min)
    print()

# %%
sim["Arinyo_min"]

# %%
ari_par_res = {}
for ii, par in enumerate(ari_par):
    ari_par_res[par] = result.x[ii]

# %%
ari_par_res

# %%
fine_p3d_model = min_data["power_model"].P3D_Mpc_k_mu(
    min_data["linear"],
    min_data["z"],
    min_data["mod_fine2D_k"],
    min_data["mod_fine2D_mu"],
    ari_par_res,
)

p3d_model, _, _, _ = binned_statistic_2d(
    min_data["mod_fine2D_k"].ravel(),
    min_data["mod_fine2D_mu"].ravel(),
    fine_p3d_model.ravel(),
    statistic="mean",
    bins=[min_data["edge_k"], min_data["edge_mu"]],
)


p1d_model = min_data["power_model"].P1D_Mpc(
    min_data["linear"], min_data["z"], min_data["k1d_Mpc"], ari_par_res
)

# %%
fig, ax = plt.subplots(2, figsize=(8, 8))

ii0 = 0
for ii in range(0, p3d_Mpc.shape[1],2):
    col = "C" + str(ii0)
    x = k3d_Mpc[:, ii]
    fact = x**3/2./np.pi**2
    # fact = 1
    ax[0].plot(x, fact * p3d_Mpc[:, ii], col)
    ax[0].plot(x, fact * p3d_model[:, ii], col + "--")
    ii0 += 1

fact = k1d_Mpc/np.pi
# fact = 1
ax[1].errorbar(k1d_Mpc, fact * p1d_Mpc)
ax[1].plot(k1d_Mpc, fact * p1d_model)

# ax[0].set_xscale("log")
# ax[0].set_yscale("log")
# ax[1].set_xscale("log")

ax[0].set_ylabel("P3D")
ax[1].set_ylabel("P1D")

# ax[0].set_ylim(1e-4, 6)

ax[1].set_xlabel("k [1/Mpc]")

# %%
fig, ax = plt.subplots(2, figsize=(8, 8))

ii0 = 0
for ii in range(0, p3d_Mpc.shape[1]):
    col = "C" + str(ii0)
    x = k3d_Mpc[:, ii]
    fact = x**3/2./np.pi**2
    # fact = 1
    ax[0].plot(x, p3d_model[:,ii]/p3d_Mpc[:, ii]-1, color=col)
    ii0 += 1

ax[0].fill_between(k3d_Mpc[:, 0], -0.05, 0.05, color="k", alpha=0.2)
ax[0].fill_between(k3d_Mpc[:, 0], -std_p3d[:, 0], std_p3d[:, 0], color="k", alpha=0.2)

fact = k1d_Mpc/np.pi
# fact = 1
ax[1].plot(k1d_Mpc, p1d_model/p1d_Mpc-1)
ax[1].fill_between(k1d_Mpc, -0.01, 0.01, color="k", alpha=0.2)
ax[1].fill_between(k1d_Mpc, -std_p1d, std_p1d, color="k", alpha=0.2)


ax[0].set_xscale("log")
# ax[0].set_yscale("log")
ax[1].set_xscale("log")

ax[0].set_ylabel("Residual P3D")
ax[1].set_ylabel("Residual P1D")

# ax[0].set_ylim(1e-4, 6)

ax[1].set_xlabel("k [1/Mpc]")

# %% [markdown]
# ## Class

# %%
from forestflow.new_fit.ArinyoFitter import ArinyoFitter
import forestflow

# %%
fitter = ArinyoFitter()

# %%
# sim_label = "mpg_hypercube"
sim_label = "mpg_central"

base_folder = os.path.dirname(forestflow.__path__[0])
name_out = "Arinyo_fit_" + sim_label + "_lowk.npy"
file = os.path.join(
    base_folder,
    "data",
    "best_arinyo",
    "minimizer_lowk",
    name_out,
)

if sim_label == "mpg_hypercube":
    data_fit = Archive3D.training_data
else:
    data_fit = Archive3D.get_testing_data(sim_label)

# %%
result_fits = {}
chi2 = np.zeros(len(data_fit))
for par in fitter.PARAM_NAMES:
    result_fits[par] = np.zeros((len(data_fit), len(fitter.PARAM_NAMES)))

# %%
for ii, sim in enumerate(data_fit):
    # if ii != 11:
    #     continue

    # save every 50 steps
    if ii % 50 == 0:
        res = {"chi2": chi2, "Arinyo": result_fits}
        np.save(file, res)

    print("Simulation", ii)
    print()
    print()

    fitter.prepare_simulation(sim)
    result = fitter.fit_iterative()
    # fitter.plot_residuals();
    for jj, par in enumerate(fitter.PARAM_NAMES):
        result_fits[par][ii] = result.x[jj]
    chi2[ii] = result.fun

# save also when finishing the loop
res = {"chi2": chi2, "Arinyo": result_fits}
np.save(file, res)

# %%
fitter.bounds

# %%
result.x

# %%
fitter.plot_fit();

# %%
fitter.plot_residuals();

# %%
# check precision of fits

# sim_label = "mpg_hypercube"
sim_label = "mpg_central"

base_folder = os.path.dirname(forestflow.__path__[0])
name_out = "Arinyo_fit_" + sim_label + "_lowk.npy"
file = os.path.join(
    base_folder,
    "data",
    "best_arinyo",
    "minimizer_lowk",
    name_out,
)

res = np.load(file, allow_pickle=True).item()
plt.hist(res["chi2"], bins=30)

print(np.argwhere(res["chi2"] == 0))

bad_fit = 2.0
ind = np.argwhere(res["chi2"] > bad_fit)[:,0]
print(len(ind))

lab_use = []

for ii in ind:
    print(
        ii,
        Archive3D.training_data[ii]["sim_label"],
        Archive3D.training_data[ii]["z"],
        np.round(Archive3D.training_data[ii]["val_scaling"], 2),
        np.round(res["chi2"][ii], 2),
    )
    sim_label = Archive3D.training_data[ii]["sim_label"]
    if sim_label not in lab_use:
        print(
            "cosmo",
            sim_label,
            Archive3D.training_data[ii]["cosmo_params"]["As"],
            Archive3D.training_data[ii]["cosmo_params"]["ns"],
        )
        lab_use.append(sim_label)

# %%
