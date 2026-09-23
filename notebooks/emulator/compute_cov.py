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
# # Compute the ForestFlow emulator covariance

# %%
# %load_ext autoreload
# %autoreload 2

from pathlib import Path

import numpy as np

import forestflow
from forestflow.P3D_cINN import P3DEmulator
from forestflow.archive import GadgetArchive3D
from forestflow.covariance import data_for_l10_forest
from forestflow.plots import plot_l1o_correlation, plot_l1o_errors

# %% [markdown]
# ## Load the emulator and corresponding archive

# %%
emulator_label = "forest_mpg"

archive = GadgetArchive3D(addcentral=True)
emulator = P3DEmulator(key=emulator_label)

# %% [markdown]
# ## Run the leave-one-out calculation

# %%
zz, k_Mpc, p1d_Mpc_orig, p1d_Mpc_sm, p1d_Mpc_emu, mask = data_for_l10_forest(
    archive,
    emulator_label,
)

rel_diff = p1d_Mpc_emu / p1d_Mpc_sm - 1
rel_diff[~mask] = 0
rel_diff[~np.isfinite(rel_diff)] = 0

rel_diff_zk = rel_diff.reshape(rel_diff.shape[0], -1)
rel_diff_k = rel_diff.reshape(-1, rel_diff.shape[-1])
cov_zk = np.cov(rel_diff_zk.T)
cov_k = np.cov(rel_diff_k.T)

# %% [markdown]
# ## Plot the covariance diagnostics

# %%
plot_l1o_correlation(cov_zk)

# %%
plot_l1o_errors(zz, k_Mpc, rel_diff, cov_zk)

# %% [markdown]
# ## Optionally store the covariance data

# %%
save_data = False

if save_data:
    output_path = (
        Path(forestflow.__file__).resolve().parents[1]
        / "data"
        / "covariance"
        / f"l1O_cov_{emulator_label}.npz"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        emulator_label=emulator_label,
        zz=zz,
        k_Mpc=k_Mpc,
        cov_k=cov_k,
        cov_zk=cov_zk,
        p1d_Mpc_orig=p1d_Mpc_orig,
        p1d_Mpc_sm=p1d_Mpc_sm,
        p1d_Mpc_emu=p1d_Mpc_emu,
        rel_diff=rel_diff,
        mask=mask,
    )
    print(f"Saved {output_path}")
