# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: lace
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Compute l1O cov matrix for emulator

# %%
# %load_ext autoreload
# %autoreload 2

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

import forestflow
from forestflow.P3D_cINN import P3DEmulator


from forestflow.model_p3d_arinyo import ArinyoModel
from lace.cosmo import cosmology

# %% [markdown]
# ## Load emulator
#
# Here to directly load the emulator

# %%
full_emulator = P3DEmulator(
    model_path=os.path.join(
        os.path.dirname(forestflow.__path__[0]), "data", "emulator_models", "forest_mpg"
    )
)

# %% [markdown]
# ### Train l1O emulators

# %%
from forestflow.archive import GadgetArchive3D
Archive3D = GadgetArchive3D(addcentral=True)

# %%
sim_mpg_central = Archive3D.get_testing_data("mpg_central")

ztar = 3.0
for ii, sim in enumerate(sim_mpg_central):
    if sim["z"] == ztar:
        ind_z3 = ii


# %%
# compute power for Arinyo model
def get_arinyo_power(
    sim,
    n3d=50,
    n1d=100,
    kmax_1d_fit=4,
    kmax_3d_fit=5,
    noise={"n_noise": 0, "keep_all_noise": False, "Lbox_Mpc": 100},
):

    data = {}

    mask_1d = (sim["k_Mpc"] <= kmax_1d_fit) & (sim["k_Mpc"] > 0)
    k1d_Mpc = sim["k_Mpc"][mask_1d]
    p1d_Mpc = sim["p1d_Mpc"][mask_1d]
    data["sim_k1d_Mpc"] = k1d_Mpc
    data["sim_p1d_Mpc"] = p1d_Mpc

    mask_3d = (sim["k3d_Mpc"] <= kmax_3d_fit) & np.isfinite(sim["p3d_Mpc"])
    k3d_Mpc = sim["k3d_Mpc"][mask_3d]
    p3d_Mpc = sim["p3d_Mpc"][mask_3d]
    mu3d = sim["mu3d"][mask_3d]
    data["sim_k3d_Mpc"] = k3d_Mpc
    data["sim_p3d_Mpc"] = p3d_Mpc
    data["sim_mu3d"] = mu3d

    # mu3d
    ari_mu = np.zeros((n3d, 2))
    ari_mu[:, 1] = 1
    ari_mu = ari_mu.T.reshape(-1)
    data["model_mu3d"] = ari_mu

    # k3d
    # min_k3d = 0.01
    # max_k3d = 0.3
    min_k3d = k3d_Mpc.min()
    max_k3d = k3d_Mpc.max()
    _ari_k3d_Mpc = np.geomspace(min_k3d, max_k3d, n3d)
    ari_k3d_Mpc = np.zeros((n3d, 2))
    ari_k3d_Mpc[:, 0] = _ari_k3d_Mpc
    ari_k3d_Mpc[:, 1] = _ari_k3d_Mpc
    ari_k3d_Mpc = ari_k3d_Mpc.T.reshape(-1)
    data["model_k3d_Mpc"] = ari_k3d_Mpc

    # k1d
    ari_k1d_Mpc = np.linspace(k1d_Mpc.min(), k1d_Mpc.max(), n1d)
    data["model_k1d_Mpc"] = ari_k1d_Mpc

    # set Arinyo at z3
    sim = sim_mpg_central[ind_z3]
    cosmo_params_dict = {}
    for par in sim["cosmo_params"]:
        if par != "omk":
            cosmo_params_dict[par] = sim["cosmo_params"][par]
        else:
            cosmo_params_dict[par] = 0.0

    fid_cosmo = cosmology.Cosmology(cosmo_params_dict=cosmo_params_dict)
    model_Arinyo = ArinyoModel(fid_cosmo)

    sim = sim_mpg_central[ind_z3]
    plin_central_z3 = model_Arinyo.linP_Mpc(sim["z"], ari_k3d_Mpc[:n3d])
    data["Plin_Mpc"] = plin_central_z3

    pars_use = {}
    for par in sim["Arinyo_min"]:
        pars_use[par] = sim["Arinyo_min"][par]

    p3d_central_z3 = model_Arinyo.P3D_Mpc_k_mu(sim["z"], ari_k3d_Mpc, ari_mu, pars_use)
    data["ari_P3D_Mpc"] = p3d_central_z3
    p1d_central_z3 = model_Arinyo.P1D_Mpc(sim["z"], ari_k1d_Mpc, pars_use)
    data["ari_P1D_Mpc"] = p1d_central_z3

    # get kaiser
    pars_use = {}
    for par in sim["Arinyo_min"]:
        if par in ["q1", "q2"]:
            pars_use[par] = 0
        elif par == "kp":
            pars_use[par] = 1e6
        else:
            pars_use[par] = sim["Arinyo_min"][par]

    p3d_central_z3_kai = model_Arinyo.P3D_Mpc_k_mu(
        sim["z"], ari_k3d_Mpc, ari_mu, pars_use
    )
    p1d_central_z3_kai = model_Arinyo.P1D_Mpc(sim["z"], ari_k1d_Mpc, pars_use)
    data["kai_P3D_Mpc"] = p3d_central_z3_kai
    data["kai_P1D_Mpc"] = p1d_central_z3_kai

    if noise["n_noise"] > 0:
        ari_noise_P1D_Mpc = np.zeros((noise["n_noise"], ari_k1d_Mpc.shape[0]))
        for ii in range(noise["n_noise"]):
            ari_noise_P1D_Mpc[ii] = model_Arinyo.P1D_Mpc_Gaussian_noise(
                sim["z"],
                ari_k1d_Mpc,
                sim["Arinyo_min"],
                seed=ii,
                Lbox_Mpc=noise["Lbox_Mpc"],
            )

        if noise["keep_all_noise"]:
            data["ari_noise_P1D_Mpc"] = ari_noise_P1D_Mpc

        data["ari_std_P1D_Mpc"] = np.std(ari_noise_P1D_Mpc, axis=0)

    return data


# %%
sim = sim_mpg_central[ind_z3]

# noise = {"n_noise": 1000, "keep_all_noise": False, "Lbox_Mpc": 100}
# power = get_arinyo_power(sim, noise=noise)

noise = {"n_noise": 1000, "keep_all_noise": False, "Lbox_Mpc": 1000}
power2 = get_arinyo_power(sim, noise=noise)
power2.keys()

# %%
n3d = int(power["model_k3d_Mpc"].shape[0]/2.)

plt.plot(
    power["model_k3d_Mpc"][:n3d],
    power["ari_P3D_Mpc"][:n3d] / power["kai_P3D_Mpc"][:n3d],
    label=r"$\mu=0$",
)

plt.plot(
    power["model_k3d_Mpc"][n3d:],
    power["ari_P3D_Mpc"][n3d:] / power["kai_P3D_Mpc"][n3d:],
    label=r"$\mu=1$",
)

plt.legend()
plt.xlabel(r"$k$ [1/Mpc]")
plt.ylabel(r"$P_\mathrm{Arinyo}/P_\mathrm{Kaiser}$")
plt.xscale("log")

# %%
k1d = power["model_k1d_Mpc"]
plt.errorbar(
    k1d,
    k1d * power["ari_P1D_Mpc"],
    k1d * power["ari_std_P1D_Mpc"],
    alpha=0.5
)

# %%

plt.errorbar(
    k1d,
    power["ari_P1D_Mpc"]/power["ari_P1D_Mpc"],
    power["ari_std_P1D_Mpc"]/power["ari_P1D_Mpc"],
    alpha=0.5
)


# %%
Lbox_Mpc2 = 1000.
Lbox_Mpc = 100.
fact = (Lbox_Mpc2/Lbox_Mpc)**(3/2)

plt.plot(
    k1d,
    power["ari_std_P1D_Mpc"],
    alpha=0.5,
)


plt.plot(
    k1d,
    power2["ari_std_P1D_Mpc"],
    alpha=0.5
)

plt.plot(
    k1d,
    power2["ari_std_P1D_Mpc"]*fact,
    alpha=0.5
)

plt.yscale("log")

# %%
cov = np.cov(power["ari_noise_P1D_Mpc"], rowvar=False)
cov.shape

# %%
# uncorrelated noise from cosmic variance

corr = np.zeros_like(cov)
for ii in range(cov.shape[0]):
    for jj in range(cov.shape[0]):
        corr[ii, jj] = cov[ii, jj] / np.sqrt(cov[ii, ii] * cov[jj, jj])

# %%
plt.imshow(corr)
plt.colorbar()

# %%
plt.plot(ari_k3d_Mpc[:n3d], p3d_central_z3[:n3d]/plin_central_z3)
plt.plot(ari_k3d_Mpc[n3d:], p3d_central_z3[n3d:]/plin_central_z3)

plt.xscale("log")


# %%
def fisher_standarize(params, derivatives, covariance)


# %%
Covariance P1D and P3D

# %%
# Get power
kmax_1d_fit = 4
kmax_3d_fit = 5

sim = Archive3D.training_data[0]
mask_1d = (sim['k_Mpc'] <= kmax_1d_fit) & (sim['k_Mpc'] > 0)
k1d_Mpc = sim['k_Mpc'][mask_1d]
p1d_Mpc = sim['p1d_Mpc'][mask_1d]

mask_3d = (sim['k3d_Mpc'] <= kmax_3d_fit) & np.isfinite(sim['p3d_Mpc'])
k3d_Mpc = sim['k3d_Mpc'][mask_3d]
p3d_Mpc = sim['p3d_Mpc'][mask_3d]
mu3d = sim['mu3d'][mask_3d]

nsims = len(Archive3D.training_data)
all_p3d = np.zeros((nsims, k3d_Mpc.shape[0]))
all_p1d = np.zeros((nsims, k1d_Mpc.shape[0]))
all_both  = np.zeros((nsims, k3d_Mpc.shape[0] + k1d_Mpc.shape[0]))

for ii, sim in enumerate(Archive3D.training_data):

    all_p3d[ii] = sim['p3d_Mpc'][mask_3d]
    all_p1d[ii] = sim['p1d_Mpc'][mask_1d]
    all_both[ii, :k3d_Mpc.shape[0]] = all_p3d[ii]
    all_both[ii, k3d_Mpc.shape[0]:] = all_p1d[ii]

# %%
n3d = 50
n1d = 100

# mu3d
ari_mu = np.zeros((n3d, 2))
ari_mu[:, 1] = 1
ari_mu = ari_mu.T.reshape(-1)

# k3d
_ari_k3d_Mpc = np.geomspace(k3d_Mpc.min(), k3d_Mpc.max(), n3d)
ari_k3d_Mpc = np.zeros((n3d, 2))
ari_k3d_Mpc[:, 0] = _ari_k3d_Mpc
ari_k3d_Mpc[:, 1] = _ari_k3d_Mpc
ari_k3d_Mpc = ari_k3d_Mpc.T.reshape(-1)

# k1d
ari_k1d_Mpc = np.linspace(k1d_Mpc.min(), k1d_Mpc.max(), n1d)

nsims = len(Archive3D.training_data)
all_p3d = np.zeros((nsims, ari_k3d_Mpc.shape[0]))
all_p1d = np.zeros((nsims, ari_k1d_Mpc.shape[0]))
all_both = np.zeros((nsims, ari_k3d_Mpc.shape[0] + ari_k1d_Mpc.shape[0]))

pars_ari = {}
for par in sim["Arinyo_min"]:
    pars_ari[par] = np.zeros((nsims))

# set Arinyo for fiducial cosmo
sim = Archive3D.training_data[0]
cosmo_params_dict = {}
for par in sim["cosmo_params"]:
    if par != "omk":
        cosmo_params_dict[par] = sim["cosmo_params"][par]
    else:
        cosmo_params_dict[par] = 0.0

fid_cosmo = cosmology.Cosmology(cosmo_params_dict=cosmo_params_dict)
model_Arinyo = ArinyoModel(fid_cosmo)

for ii, sim in enumerate(Archive3D.training_data):

    for par in sim["Arinyo_min"]:
        pars_ari[par][ii] = sim["Arinyo_min"][par]

    new_cosmo_params = {}
    for par in ["As", "ns"]:
        new_cosmo_params[par] = sim["cosmo_params"][par]

    all_p3d[ii] = model_Arinyo.P3D_Mpc_k_mu(
        sim["z"], ari_k3d_Mpc, ari_mu, sim["Arinyo_min"], new_cosmo_params=new_cosmo_params
    )

    all_p1d[ii] = model_Arinyo.P1D_Mpc(
        sim["z"], ari_k1d_Mpc, sim["Arinyo_min"], new_cosmo_params=new_cosmo_params
    )

    all_both[ii, :ari_k3d_Mpc.shape[0]] = all_p3d[ii]
    all_both[ii, ari_k3d_Mpc.shape[0]:] = all_p1d[ii]


# %%
cov_p1d = np.cov(all_p1d.T)
cov_p3d = np.cov(all_p3d.T)
cov_both = np.cov(all_both.T)

icov_both = np.linalg.inv(cov_both)

# %%
corr_both = np.zeros_like(cov_both)
for ii in range(cov_both.shape[0]):
    for jj in range(cov_both.shape[0]):
        corr_both[ii, jj] = cov_both[ii, jj]/np.sqrt(cov_both[ii, ii] * cov_both[jj, jj])

icorr_both = np.zeros_like(icov_both)
for ii in range(icov_both.shape[0]):
    for jj in range(icov_both.shape[0]):
        icorr_both[ii, jj] = icov_both[ii, jj]/np.sqrt(icov_both[ii, ii] * icov_both[jj, jj])

# %%
plt.imshow(corr_both)

# %%
plt.imshow(icov_both)

# %%

for ii in range(2):
    _ = (ari_mu == ii)

    plt.plot(ari_k3d_Mpc[_], np.diag(cov_p3d)[_], label="P1D mu=" + str(np.mean(ari_mu[_])))

plt.plot(ari_k1d_Mpc, np.diag(cov_p1d), label="P1D")
plt.legend()
plt.yscale("log")
plt.xscale("log")

# %%
Derivatives

# %%
sim_label = "mpg_central"
testing_data = []
for sim in Archive3D.training_data:
    if (sim["sim_label"] == sim_label) & (sim["z"] == 3):
        sim_der = sim

new_cosmo_params = {}
for par in ["As", "ns"]:
    new_cosmo_params[par] = sim_der["cosmo_params"][par]


# %%
diff_pars_ari = {}

for par in sim["Arinyo_min"]:
    diff_pars_ari[par] = pars_ari[par].max() - pars_ari[par].min()

diff_pars_ari

# %%
par_ari = {}
par_ari_var_top = {}
par_ari_var_bot = {}
for par in sim["Arinyo_min"]:
    par_ari[par] = sim["Arinyo_min"][par]

all_p3d_der = {}
all_p1d_der = {}
all_both_der = {}


for par1 in par_ari:
    all_both_der[par1] = np.zeros((ari_k3d_Mpc.shape[0] + ari_k1d_Mpc.shape[0]))

    for par2 in par_ari:
        if par1 == par2:
            hh = diff_pars_ari[par1]
        else:
            hh = 0
        par_ari_var_top[par2] = par_ari[par2] + 0.001 * hh
        par_ari_var_bot[par2] = par_ari[par2] - 0.001 * hh

    hh = diff_pars_ari[par1]

    all_p3d_der_top = model_Arinyo.P3D_Mpc_k_mu(
        sim["z"], ari_k3d_Mpc, ari_mu, par_ari_var_top, new_cosmo_params=new_cosmo_params
    )

    all_p3d_der_bot = model_Arinyo.P3D_Mpc_k_mu(
        sim["z"], ari_k3d_Mpc, ari_mu, par_ari_var_bot, new_cosmo_params=new_cosmo_params
    )

    all_p3d_der[par1] = (all_p3d_der_top - all_p3d_der_bot)/2/hh

    all_p1d_der_top = model_Arinyo.P1D_Mpc(
        sim["z"], ari_k1d_Mpc, par_ari_var_top, new_cosmo_params=new_cosmo_params
    )

    all_p1d_der_bot = model_Arinyo.P1D_Mpc(
        sim["z"], ari_k1d_Mpc, par_ari_var_bot, new_cosmo_params=new_cosmo_params
    )

    all_p1d_der[par1] = (all_p1d_der_top - all_p1d_der_bot)/2/hh

    all_both_der[par1][:ari_k3d_Mpc.shape[0]] = all_p3d_der[par1]
    all_both_der[par1][ari_k3d_Mpc.shape[0]:] = all_p1d_der[par1]


# %%
fig, ax = plt.subplots(len(par_ari)-1, sharex=True, figsize=(8, 20))

ii = 0
for jj, par in enumerate(par_ari):
    if par == "beta":
        continue
    ax[ii].plot(ari_k3d_Mpc[:50], all_p3d_der[par][:50], label=par)
    ax[ii].plot(ari_k3d_Mpc[50:], all_p3d_der[par][50:])
    ax[ii].plot(ari_k1d_Mpc, all_p1d_der[par])
    ax[ii].legend()
    ii += 1
plt.xscale("log")

# %%
fig, ax = plt.subplots(len(par_ari)-1, sharex=True, figsize=(8, 20))

ii = 0
for jj, par in enumerate(par_ari):
    if par == "beta":
        continue
    ax[ii].plot(ari_k1d_Mpc, all_p1d_der[par], label=par)
    ax[ii].legend()
    ii += 1
plt.xscale("log")

# %%
fisher = {}

for par1 in par_ari:

    if par1 == "beta":
        continue
    fisher[par1] = {}

    for par2 in par_ari:
        if par2 == "beta":
            continue

        # prod = np.dot(all_both_der[par1], np.dot(icov_both, all_both_der[par2]))
        prod = np.sum(all_both_der[par1]*all_both_der[par2]/np.diag(cov_both))
        fisher[par1][par2] = prod
        # print(par1, par2, np.sqrt(np.abs(fisher[par1][par2])))


# %%
arinyo_params = []
for par in par_ari:
    if par == "beta":
        continue
    arinyo_params.append(par)
arinyo_params

# %%
nari = len(arinyo_params)
fisher_matrix = np.zeros((nari, nari))
for ii, par1 in enumerate(arinyo_params): 
    for jj, par2 in enumerate(arinyo_params):
        fisher_matrix[ii, jj] = fisher[par1][par2]



# %%

# %%
F_reg = F_std + 1e-6 * np.trace(F_std)/F_std.shape[0] * np.eye(F_std.shape[0])
L = np.linalg.cholesky(F_reg)

np.ones((nari)) @ L.T

# %%
arinyo_params

# %%
eigval = np.linalg.eigvalsh(L)

print("min =", eigval.min())
print("max =", eigval.max())
print("condition number =", eigval.max()/eigval.min())

# %%
# standarize
F_std = fisher_matrix * 1

# Eigen decomposition
eigval, eigvec = np.linalg.eigh(F_std)

# Regularize tiny eigenvalues
eigval = np.maximum(eigval, 1e-8)

# Whitening matrix
W = np.diag(np.sqrt(eigval)) @ eigvec.T

# %%
np.ones((nari)) @ W.T


# %%
def whiten(theta, mu, sigma, W):
    theta_std = (theta - mu) / sigma
    return theta_std @ W.T

def unwhiten(theta_white, mu, sigma, W):
    theta_std = theta_white @ np.linalg.inv(W).T
    return theta_std * sigma + mu


# %%
# withen fisher
import numpy as np

# Mean and standard deviation from training set
mu = theta.mean(axis=0)
sigma = theta.std(axis=0)

# Standardize parameters
theta_std = (theta - mu) / sigma

# Fisher matrix in standardized coordinates
D = np.diag(sigma)
F_std = D @ F @ D

# Eigen decomposition
eigval, eigvec = np.linalg.eigh(F_std)

# Regularize tiny eigenvalues
eigval = np.maximum(eigval, 1e-8)

# Whitening matrix
W = np.diag(np.sqrt(eigval)) @ eigvec.T

# Whitened parameters
theta_white = theta_std @ W.T

# %%
par_ari_var_bot

# %%
save_path = os.path.join(
    os.path.dirname(forestflow.__path__[0]), "data", "emulator_models", "test"
)
emulator = P3DEmulator(
    training_data=Archive3D.training_data,
    emu_input_names=Archive3D.emu_params,
    training_type="Arinyo_min",
    train=True,
    nLayers_inn=5,
    nepochs=1250,
    batch_size=16,
    lr=5e-4,
    dims_int=16,
    use_val_set=True,
    # use_val_set=False,
    Nrealizations=10000,
    save_path=save_path,
)

# %%
# takes forever
4 layers, batch 32 went to -24 for 1000, go longer!
4 layers, batch 16 went to -24.5 for 1000, go longer!
    
# sweet spot
dims_int=16
5 layers, batch 16 went to -30, 1000 is good
# anything better??? stop a little bit longer than when using
# the validation sample
dims_int=32
way worse
dims_int=8
worse

# bad
6 layers, batch 16 went to -24.84, stop

# %%
n = 25

plt.plot(-np.array(emulator.loss_arr)[n:])
plt.plot(-np.array(emulator.val_loss_arr)[n:])

# %%
emulator = P3DEmulator(
    model_path=os.path.join(
        os.path.dirname(forestflow.__path__[0]), "data", "emulator_models", "test"
    )
)

# %%

# %%
len(Archive3D.training_data)

# %%
for isim, sim in enumerate(Archive3D.list_sim_cube):
    if isim != 0:
        continue
    print(sim)
    print()
    save_path = os.path.join(
        os.path.dirname(forestflow.__path__[0]), "data", "emulator_models", "l1O", sim
    )
    emulator = P3DEmulator(
        training_data=Archive3D.training_data,
        emu_input_names=Archive3D.emu_params,
        training_type="Arinyo_min",
        train=True,
        drop_sim=sim,
        # nepochs=4000,
        nepochs=10,
        batch_size=20,
        step_size=200,
        weight_decay=0.01,
        Nrealizations=6000,
        save_path=save_path,
    )
    # break

# %%
plt.plot(np.log(-np.array(emulator.loss_arr)))

# %%
emulator.Arinyo_params

# %% [markdown]
# ## Stop

# %%
p1d_Mpc_sm from evaluating ForestFlow 
for the best-fitting parameters of the 
model, this is in the training data?

apply binning before comparing data?

out = data_for_l10(Archive3D)

# %%
# load data of the simulation we removed
sim_label = "mpg_central"
testing_data = []
for sim in Archive3D.training_data:
    if sim["sim_label"] == sim_label:
        testing_data.append(sim)
# evaluate emulator for this simulation

# store the results from the emulator, 
# the smooth result from Arinyo, and the actual
# result from the simulation



# %%
add central to training data?

# %%

# %%

# %%
sim['cosmo_params']

# %%
# Get power
kmax_1d_fit = 4
kmax_1d_plot = kmax_1d_fit + 1
kp_Mpc = 0.7

sim = Archive3D.training_data[0]
mask_1d = (sim['k_Mpc'] <= kmax_1d_plot) & (sim['k_Mpc'] > 0)
k1d_Mpc = sim['k_Mpc'][mask_1d]
p1d_Mpc = sim['p1d_Mpc'][mask_1d]

# %%
cosmo_params_dict = {}
for par in sim['cosmo_params']:
    if par != "omk":
        cosmo_params_dict[par] = sim['cosmo_params'][par]
    else:
        cosmo_params_dict[par] = 0.0

fid_cosmo = cosmology.Cosmology(cosmo_params_dict=cosmo_params_dict)
model_Arinyo = ArinyoModel(fid_cosmo)



# %%
# emulator = P3DEmulator(
#     model_path=os.path.join(
#         os.path.dirname(forestflow.__path__[0]), "data", "emulator_models", "l1O", "mpg_0"
#     )
# )

# %%
# emulator = full_emulator



for sim in testing_data:
    emu_params = {}
    for param in Archive3D.emu_params:
        emu_params[param] = sim[param]
    model3d_coeffs = emulator.predict_Arinyos(emu_params=emu_params)

    new_cosmo_params = {}
    for par in ["As", "ns"]:
        new_cosmo_params[par] = sim["cosmo_params"][par]

    p1d_emu = model_Arinyo.P1D_Mpc(
        sim["z"], k1d_Mpc, model3d_coeffs, new_cosmo_params=new_cosmo_params
    )
    # p1d_emu_l10 = model_Arinyo.P1D_Mpc(
    #     sim["z"], k1d_Mpc, model3d_coeffs, new_cosmo_params=new_cosmo_params
    # )

    p1d_fit = model_Arinyo.P1D_Mpc(
        sim["z"], k1d_Mpc, sim["Arinyo_min"], new_cosmo_params=new_cosmo_params
    )

    _mask_1d = (sim['k_Mpc'] <= kmax_1d_plot) & (sim['k_Mpc'] > 0)
    p1d_sim = sim["p1d_Mpc"][_mask_1d]

    plt.plot(k1d_Mpc, p1d_emu/p1d_fit, label=np.round(sim["z"],2))

    if sim["z"] == 3:
        break

plt.legend()

# %% [markdown]
#

# %%
plt.plot(k1d_Mpc, p1d_emu/p1d_fit)
# plt.plot(k1d_Mpc, p1d_emu_l10/p1d_fit, ":")
plt.plot(k1d_Mpc, p1d_sim/p1d_fit, "--")
plt.axhline(1, color="black", ls=":")
plt.xlabel(r'$k$ [Mpc]')
plt.ylabel(r'$P_{\rm 1D}(k)$')
plt.xscale('log')

# %%
# emulator = full_emulator

model3d_coeffs = emulator.predict_Arinyos(emu_params=emu_params)
model3d_coeffs

# %%
sim["Arinyo_min"]

# %% [markdown]
#
