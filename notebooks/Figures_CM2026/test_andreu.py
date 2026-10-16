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
# # Compare bias-beta with observations
#
# For random points of the DESI DR1 P1D chain, get the ForestFlow inputs at z=2.33,
# evaluate `forest_mpg_gp`, and compute the combinations of bias, beta, sigma_8 and
# f sigma_8 measured by the BAO analysis.
#
# The chain-derived quantities (inputs to ForestFlow, sigma_8(z=2.33), f sigma_8(z=2.33))
# are read from `int_data_figs/desi_priors_cosmo_IGM_from_p1d.npy`, produced by
# `forestflow.priors.paper.set_samples.set_process_p1d_chain` from the cup1d chain
# (`out_DESI_DR1/DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7`).

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt

from forestflow.emulator import load_emulator

# %%
chain = np.load(
    "int_data_figs/desi_priors_cosmo_IGM_from_p1d.npy", allow_pickle=True
).item()
zeff = 2.33
iz = np.argmin(np.abs(chain["zs"] - zeff))
print(chain["zs"][iz])


# %%
# For random points in the P1D chain, collect the emulator parameters at z=2.33
def get_params_from_chain(chain, iz, nn=100, seed=0):
    rng = np.random.default_rng(seed)
    ind = rng.permutation(chain["As"].shape[0])[:nn]
    new_chain = []
    for ii in ind:
        params = {
            "sigma_8": chain["sig_8"][ii],
            "f_sigma_8": chain["f_sig_8"][ii],
        }
        for key in ["Delta2_p", "n_p", "mF", "gamma", "sigT_Mpc", "kF_Mpc"]:
            params[key] = chain[key][ii, iz]
        new_chain.append(params)
    return new_chain


# %%
new_chain = get_params_from_chain(chain, iz, nn=100)
new_chain[0]

# %%
sig8 = np.array([sample["sigma_8"] for sample in new_chain])
print(np.mean(sig8))
print(np.sqrt(np.var(sig8)))

# %%
f_sig8 = np.array([sample["f_sigma_8"] for sample in new_chain])
print(np.mean(f_sig8))
print(np.sqrt(np.var(f_sig8)))

# %%
DL2 = np.array([sample["Delta2_p"] for sample in new_chain])
print(np.mean(DL2))
print(np.sqrt(np.var(DL2)))

# %%
plt.scatter(sig8, DL2)
plt.xlabel(r"$\sigma_8(z=2.33)$")
plt.ylabel(r"$\Delta^2_p(z=2.33)$")

# %% [markdown]
# ## Load ForestFlow

# %%
emulator = load_emulator("forest_mpg_gp")

# %%
# the GP is deterministic, and it evaluates all samples at once
info = emulator.evaluate(
    [{key: sample[key] for key in emulator.input_labels} for sample in new_chain]
)
for jj, sample in enumerate(new_chain):
    for key in emulator.output_labels:
        sample[key] = info[key][jj]

# %%
new_chain[1]

# %%
bias = np.array([sample["bias"] for sample in new_chain])
print(np.mean(bias))
print(np.sqrt(np.var(bias)))

# %%
# beta = bias_eta * f(z) / bias, with f = f_sigma_8 / sigma_8
bias_eta = np.array([sample["bias_eta"] for sample in new_chain])
beta = bias_eta * (f_sig8 / sig8) / bias
print(np.mean(beta))
print(np.sqrt(np.var(beta)))

# %%
b_eta = beta * bias * sig8 / f_sig8
print(np.mean(b_eta))
print(np.sqrt(np.var(b_eta)))

# %%
b_sig_8 = bias * sig8
print(np.mean(b_sig_8))
print(np.sqrt(np.var(b_sig_8)))

# %%
b_eta_f_sig_8 = b_eta * f_sig8
print(np.mean(b_eta_f_sig_8))
print(np.sqrt(np.var(b_eta_f_sig_8)))

# %%
