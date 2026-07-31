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

# %%
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
import numpy as np
from forestflow.model_p3d_arinyo import ArinyoModel
import h5py


# %%

data = {}
data["cosmo"] = {}
cosmo_labs = ["ombh2", "omch2", "ns", "As", "H0"]

file = "/home/jchaves/Proyectos/projects/lya/P3d+P1d_lya_ASTRID_new.hdf5"

with h5py.File(file, "r") as f:

    def load_dataset(name, obj):
        if isinstance(obj, h5py.Dataset):
            data[name] = obj[()]    # or obj[:] for array datasets

    f.visititems(load_dataset)

    data["z"] = f.attrs["z"]
    for par in cosmo_labs:
        if par == "H0":
            data["cosmo"][par] = f.attrs["hubble"] * 100.
        else:
            data["cosmo"][par] = f.attrs[par]

    for att in f.attrs:
        print(att, f.attrs[att])

data["cosmo"]["w"] = -1.0
data["cosmo"]["mnu"] = 0.0
data["cosmo"]["nrun"] = 0.0
data["cosmo"]["omk"] = 0.0
    

# %%
for ii in range(data["p3d_lya_Mpc"].shape[1]):
    lab = "mu = %.2f" % np.nanmean(data["mu"][:, ii])
    kk = data["k_Mpc"][:, ii]
    mask = kk > 0.1
    sig = kk**2*data["p3d_lya_Mpc"][:, ii]
    err = kk**2*data["p3d_lya_std_Mpc"][:, ii] * np.sqrt(2)
    plt.errorbar(kk[mask], sig[mask], err[mask], label=lab)
    # plt.plot(kk, sig, label=lab)
# plt.legend()
plt.xscale("log")
plt.yscale("log")

# %%
data.keys()

# %%
mask = data["klos_Mpc"] > 0.
plt.plot(data["klos_Mpc"][mask], data["klos_Mpc"][mask] * data["p1d_lya_Mpc"][mask] / np.pi)
plt.xscale("log")
plt.yscale("log")

# %%
kmax_3d_fit = 5
kmax_1d_fit = 5

z = data["z"]
k3d_Mpc = data["k_Mpc"]
mu3d = data["mu"]
p3d_Mpc = data["p3d_lya_Mpc"]
p3d_std_Mpc = data["p3d_lya_std_Mpc"]
mask_3d = (k3d_Mpc[:, 0] > 0.1) & (k3d_Mpc[:, 0] <= kmax_3d_fit)

mask_1d = (data["klos_Mpc"] <= kmax_1d_fit) & (data["klos_Mpc"] > 0.1)
k1d_Mpc = data["klos_Mpc"][mask_1d]
p1d_Mpc = np.real(data["p1d_lya_Mpc"][mask_1d])

# %%
from lace.cosmo import cosmology

fid_cosmo = cosmology.Cosmology(data["cosmo"])
model_Arinyo = ArinyoModel(fid_cosmo)
linear = model_Arinyo.linear_theory(z)

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
min_data["k3d_Mpc"] = k3d_Mpc[mask_3d]
min_data["mu3d"] = mu3d[mask_3d]
min_data["target_p3d_Mpc"] = p3d_Mpc[mask_3d]
min_data["k1d_Mpc"] = k1d_Mpc
min_data["target_p1d_Mpc"] = p1d_Mpc
min_data["power_model"] = model_Arinyo
# min_data["fact3D"] = min_data["k3d_Mpc"] ** 3/2./np.pi**2
# min_data["fact1D"] = min_data["k1d_Mpc"] /np.pi

def chi2(arr_params, data, err_tol_3D=0.05, err_tol_1D=0.02):

    ari_par = {
        'bias': arr_params[0],
        'bias_eta': arr_params[1],
        'q1': arr_params[2],
        'q2': arr_params[3],
        'kvav': arr_params[4],
        'av': arr_params[5],
        'bv': arr_params[6],
        'kp': arr_params[7]
    }

    p3d_model = data["power_model"].P3D_Mpc_k_mu(data["linear"], data["z"], data["k3d_Mpc"], data["mu3d"], ari_par)
    p1d_model = data["power_model"].P1D_Mpc(data["linear"], data["z"], data["k1d_Mpc"], ari_par)

    chi2_3D = np.nanmean((p3d_model/data["target_p3d_Mpc"]-1) ** 2)/err_tol_3D**2
    chi2_1D = np.nanmean((p1d_model/data["target_p1d_Mpc"]-1) ** 2)/err_tol_1D**2

    # print(chi2_3D, chi2_1D)

    return chi2_3D + chi2_1D


# %%
arr_par = np.zeros(len(ari_par.keys()))
for ii, param in enumerate(ari_par):
    arr_par[ii] = ari_par[param]
chi2(arr_par, min_data)

# %%
nelem3d = p3d_Mpc.shape[0] * p3d_Mpc.shape[1]
nelem1d = p1d_Mpc.shape[0]

print(nelem3d, nelem1d)

# %%
from scipy.optimize import minimize


# bias                -0.157      0.053
# beta                 1.428      0.186
# q1                   0.348      0.125
# kvav                 0.589      0.147
# av                   0.408      0.149
# bv                   1.690      0.069
# kp                  14.592      2.565
# q2                   0.253      0.111
# bias_eta            -0.221      0.050

# Initial guess
x0 = np.array([
    -0.21,  # bias
    -0.35,    # bias_eta
    0.28,    # q1
    0.5,    # q2
    0.56,   # kvav
    0.,   # av
    1.6,   # bv
    6.6,   # kp
])

# [-0.23292651 -0.51011834  0.71462909  0.          0.17221933  0.
#   1.8        20.        ]

bounds = [
    (-0.4, -0.1),   # bias
    (-0.7, -0.17),    # bias_eta
    (0.1, 1.2),    # q1
    (-0.8, 0.8),    # q2
    (0.2, 0.8),  # kvav
    (-0.5, 1.),    # av
    (1.0, 2.4),    # bv
    (4., 20.),   # kp
]


# %%
result = minimize(
    chi2,
    x0,
    args=(min_data,),
    method="L-BFGS-B",
    bounds=bounds,
    options={
        "disp": True,
        "maxiter": 500,
    },
)

print(result.success)
print(result.message)
print(result.fun)
print(result.x)

# %%
ari_par_res = {}
for ii, par in enumerate(ari_par):
    ari_par_res[par] = result.x[ii]

# %%
ari_par_res

# %%
chi2(result.x, min_data)

# %%
p3d_model = min_data["power_model"].P3D_Mpc_k_mu(min_data["linear"], min_data["z"], min_data["k3d_Mpc"], min_data["mu3d"], ari_par_res)
p1d_model = min_data["power_model"].P1D_Mpc(min_data["linear"], min_data["z"], min_data["k1d_Mpc"], ari_par_res)

# %%
fig, ax = plt.subplots(2, figsize=(8, 8))

ii0 = 0
for ii in range(0, p3d_Mpc.shape[1],2):
    col = "C" + str(ii0)
    x = k3d_Mpc[mask_3d, ii]
    fact = x**3/2./np.pi**2
    # fact = 1
    ax[0].plot(x, fact * p3d_Mpc[mask_3d, ii], col)
    ax[0].plot(x, fact * p3d_model[:, ii], col + "--")
    ii0 += 1

fact = k1d_Mpc/np.pi
# fact = 1
ax[1].errorbar(k1d_Mpc, fact * p1d_Mpc)
ax[1].plot(k1d_Mpc, fact * p1d_model)

ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[1].set_xscale("log")

ax[0].set_ylabel("P3D")
ax[1].set_ylabel("P1D")

# ax[0].set_ylim(1e-4, 6)

ax[1].set_xlabel("k [1/Mpc]")

# %%
fig, ax = plt.subplots(2, figsize=(8, 8))

ii0 = 0
for ii in range(0, p3d_Mpc.shape[1], 3):
    col = "C" + str(ii0)
    x = k3d_Mpc[mask_3d, ii]
    fact = x**3/2./np.pi**2
    # fact = 1
    ax[0].plot(x, p3d_model[:,ii]/p3d_Mpc[mask_3d, ii]-1, color=col)
    ii0 += 1

ax[0].fill_between(k3d_Mpc[mask_3d, 0], -0.05, 0.05, color="k", alpha=0.2)

fact = k1d_Mpc/np.pi
# fact = 1
ax[1].plot(k1d_Mpc, p1d_model/p1d_Mpc-1)
ax[1].fill_between(k1d_Mpc, -0.02, 0.02, color="k", alpha=0.2)

ax[0].set_xscale("log")
# ax[0].set_yscale("log")
ax[1].set_xscale("log")

ax[0].set_ylabel("Residual P3D")
ax[1].set_ylabel("Residual P1D")

# ax[0].set_ylim(1e-4, 6)

ax[1].set_xlabel("k [1/Mpc]")

# %%

# %%
