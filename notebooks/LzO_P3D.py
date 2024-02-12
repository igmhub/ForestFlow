# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: emulators
#     language: python
#     name: emulators
# ---

# %% [markdown]
# # NOTEBOOK TO REPRODUCE THE LEAVE-REDSHIFT-OUT TEST OF forestflow

# %%
import numpy as np
import os
import sys

# %%
from forestflow.model_p3d_arinyo import ArinyoModel
from forestflow import model_p3d_arinyo
from forestflow.archive import GadgetArchive3D
from forestflow.P3D_cINN import P3DEmulator
from forestflow.likelihood import Likelihood
from forestflow.utils import sigma68
from forestflow.plots_v0 import plot_p1d_LzO, plot_p3d_LzO


# %%

import matplotlib.pyplot as plt
import matplotlib

font = {"size": 22}
matplotlib.rc("font", **font)
plt.rc("text", usetex=False)
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rc("xtick", labelsize=15)
matplotlib.rc("ytick", labelsize=15)


# %% [markdown]
#
# ## DEFINE FUNCTIONS


# %%
def ls_level(folder, nlevels):
    for ii in range(nlevels):
        folder = os.path.dirname(folder)
    folder += "/"
    return folder


path_program = ls_level(os.getcwd(), 1)
print(path_program)
sys.path.append(path_program)

# %% [markdown]
# # LOAD DATA

# %%
# %%time
folder_interp = path_program + "/data/plin_interp/"
folder_lya_data = path_program + "/data/best_arinyo/"

Archive3D = GadgetArchive3D(
    base_folder=path_program,
    folder_data=folder_lya_data,
    force_recompute_plin=False,
    average="both",
)
print(len(Archive3D.training_data))


# %%
Nrealizations = 100
Nsim = 30

test_sim = central = Archive3D.get_testing_data(
    "mpg_central", force_recompute_plin=True
)
z_grid = [d["z"] for d in test_sim]
Nz = len(z_grid)


k_Mpc = Archive3D.training_data[0]["k3d_Mpc"]
mu = Archive3D.training_data[0]["mu3d"]

k_mask = (k_Mpc < 4) & (k_Mpc > 0)

k_Mpc = k_Mpc[k_mask]
mu = mu[k_mask]

# %% [markdown]
# ## LEAVE REDSHIFT OUT TEST

# %% [markdown]
# #### Define redshifts to test
#

# %%
z_test = [2.5, 3.5]

# %%
p3ds_pred = np.zeros(shape=(Nsim, len(z_test), 148))
p1ds_pred = np.zeros(shape=(Nsim, len(z_test), 53))

p3ds_arinyo = np.zeros(shape=(Nsim, len(z_test), 148))
p1ds_arinyo = np.zeros(shape=(Nsim, len(z_test), 53))

p1ds_sims = np.zeros(shape=(Nsim, len(z_test), 53))
p3ds_sims = np.zeros(shape=(Nsim, len(z_test), 148))


for iz, zdrop in enumerate(z_test):
    print(f"Dropping redshift {z_test}")

    training_data = [d for d in Archive3D.training_data if d["z"] != zdrop]

    p3d_emu = P3DEmulator(
        training_data,
        Archive3D.emu_params,
        nepochs=300,
        lr=0.001,  # 0.005
        batch_size=20,
        step_size=200,
        gamma=0.1,
        weight_decay=0,
        adamw=True,
        nLayers_inn=12,  # 15
        Archive=Archive3D,
        use_chains=False,
        chain_samp=100_000,
        folder_chains="/data/desi/scratch/jchavesm/p3d_fits_new/",
    )

    for s in range(Nsim):
        # load arinyo module
        flag = f"Plin_interp_sim{s}.npy"
        file_plin_inter = folder_interp + flag
        pk_interp = np.load(file_plin_inter, allow_pickle=True).all()
        model_Arinyo = model_p3d_arinyo.ArinyoModel(camb_pk_interp=pk_interp)

        # define test sim
        dict_sim = [
            d
            for d in Archive3D.training_data
            if d["z"] == zdrop
            and d["sim_label"] == f"mpg_{s}"
            and d["val_scaling"] == 1
        ]

        # p1d from sim
        like = Likelihood(
            dict_sim[0], Archive3D.rel_err_p3d, Archive3D.rel_err_p1d
        )
        k1d_mask = like.like.ind_fit1d.copy()
        p1d_sim = like.like.data["p1d"][k1d_mask]

        # p3d from sim
        p3d_sim = dict_sim[0]["p3d_Mpc"][p3d_emu.k_mask]
        p3d_sim = np.array(p3d_sim)

        p1ds_sims[s, iz] = p1d_sim
        p3ds_sims[s, iz] = p3d_sim

        # load BF Arinyo and estimated the p3d and p1d from BF arinyo parameters
        BF_arinyo = dict_sim[0]["Arinyo_minin"]

        p3d_arinyo = model_Arinyo.P3D_Mpc(zdrop, k_Mpc, mu, BF_arinyo)
        p3ds_arinyo[s, iz] = p3d_arinyo

        p1d_arinyo = like.like.get_model_1d(parameters=BF_arinyo)
        p1d_arinyo = p1d_arinyo[k1d_mask]
        p1ds_arinyo[s, iz] = p1d_arinyo

        # predict p3d and p1d from predicted arinyo parameters
        p3d_pred_median = p3d_emu.predict_P3D_Mpc(
            sim_label=f"mpg_{s}", z=zdrop, test_sim=dict_sim, return_cov=False
        )

        p1d_pred_median = p3d_emu.predict_P1D_Mpc(
            sim_label=f"mpg_{s}", z=zdrop, test_sim=dict_sim, return_cov=False
        )

        p3ds_pred[s, iz] = p3d_pred_median
        p1ds_pred[s, iz] = p1d_pred_median

    print(
        "Mean fractional error P3D pred to Arinyo",
        ((p3ds_pred[:, iz] / p3ds_arinyo[:, iz] - 1) * 100).mean(),
    )
    print(
        "Std fractional error P3D pre to Arinyo",
        ((p3ds_pred[:, iz] / p3ds_arinyo[:, iz] - 1) * 100).std(),
    )

    print(
        "Mean fractional error P3D Arinyo model",
        ((p3ds_arinyo[:, iz] / p3ds_sims[:, iz] - 1) * 100).mean(),
    )
    print(
        "Std fractional error P3D Arinyo model",
        ((p3ds_arinyo[:, iz] / p3ds_sims[:, iz] - 1) * 100).std(),
    )

    print(
        "Mean fractional error P3D pred to sim",
        ((p3ds_pred[:, iz] / p3ds_sims[:, iz] - 1) * 100).mean(),
    )
    print(
        "Std fractional error P3D pred to sim",
        ((p3ds_pred[:, iz] / p3ds_sims[:, iz] - 1) * 100).std(),
    )

    print(
        "Mean fractional error P1D pred to Arinyo",
        ((p1ds_pred[:, iz] / p1ds_arinyo[:, iz] - 1) * 100).mean(),
    )
    print(
        "Std fractional error P1D pred to Arinyo",
        ((p1ds_pred[:, iz] / p1ds_arinyo[:, iz] - 1) * 100).std(),
    )

    print(
        "Mean fractional error P1D Arinyo model",
        ((p1ds_arinyo[:, iz] / p1ds_sims[:, iz] - 1) * 100).mean(),
    )
    print(
        "Std fractional error P1D Arinyo model",
        ((p1ds_arinyo[:, iz] / p1ds_sims[:, iz] - 1) * 100).std(),
    )

    print(
        "Mean fractional error P1D pred to sim",
        ((p1ds_pred[:, iz] / p1ds_sims[:, iz] - 1) * 100).mean(),
    )
    print(
        "Std fractional error P1D pred to sim",
        ((p1ds_pred[:, iz] / p1ds_sims[:, iz] - 1) * 100).std(),
    )


# %% [markdown]
# ## PLOTTING

# %%
fractional_errors_arinyo = (p3ds_pred / p3ds_arinyo - 1) * 100
fractional_errors_sims = (p3ds_pred / p3ds_sims - 1) * 100
fractional_errors_bench = (p3ds_arinyo / p3ds_sims - 1) * 100

# %%
plot_p3d_LzO(Archive3D, fractional_errors_arinyo, z_test)

# %%
fractional_errors_arinyo_p1d = (p1ds_pred / p1ds_arinyo - 1) * 100
fractional_errors_sims_p1d = (p1ds_pred / p1ds_sims - 1) * 100
fractional_errors_bench_p1d = (p1ds_arinyo / p1ds_sims - 1) * 100

# %%
plot_p1d_LzO(Archive3D, fractional_errors_sims_p1d, z_test)
# %%
