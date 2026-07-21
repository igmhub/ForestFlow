# ---
# jupyter:
#   jupytext:
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
# # P1D chain -> ForestFlow -> P3D

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

from forestflow.priors_paper import all_plots, load

np.__version__

# %% [markdown]
# #### Load data

# %%
dict_mapping = load.load_map_igm_p3d()

# %% [markdown]
# Table with output of P1D chain

# %%
all_plots.table_cosmo_igm(dict_mapping)

# %%
bao_data = load.load_BAO_data()

# %%
all_plots.plot_bias_beta_zev(bao_data, dict_mapping)
# plt.gcf().savefig("bias_beta_BAOvsP1D.png")
# plt.gcf().savefig("bias_beta_BAOvsP1D.pdf")

# %%
all_plots.plot_p3d_small_z(dict_mapping)

# %% [markdown]
# Check consistency of P1D from lace and forestflow

# %%

from forestflow.model_p3d_arinyo import ArinyoModel
from lace.cosmo import cosmology, rescale_cosmology

# fid_cosmo_params = {
#     "H0": 67.66,
#     "mnu": 0,
#     "omch2": 0.119,
#     "ombh2": 0.0224,
#     "omk": 0,
#     "As": 2.105e-09,
#     "ns": 0.9665,
#     "nrun": 0.0,
#     "pivot_scalar": 0.05,
#     "w": -1.0,
# }
fid_cosmo = cosmology.Cosmology()
model_Arinyo = ArinyoModel(fid_cosmo)

k_par = np.geomspace(0.1, 4, 100)

# %%
from lace.emulator.emulator_manager import set_emulator
lace_mpg = set_emulator("CH24_mpgcen_gpr")

# %%
pars_list = ["Delta2_p", "n_p", "mF", "gamma", "sigT_Mpc", "kF_Mpc"]
pars_ari = ['bias', 'bias_eta', 'q1', 'kvav', 'av', 'bv', 'kp', 'q2']
cosmo_list = ["As", "ns"]
all_pars = list(dict_mapping["emu_params"].keys())
zz = dict_mapping["emu_params"]["zs"].copy()
nz = len(zz)

# nn = 100
nz = 11

all_P1D_ff = np.zeros((nn, nz, len(k_par)))
all_P1D_lace = np.zeros((nn, nz, len(k_par)))

for jj in range(nn):

    new_cosmo = {}
    for par in cosmo_list:
        new_cosmo[par] = dict_mapping["emu_params"][par][jj]

    for iz in range(nz):
        input_emu = {}
        for par in pars_ari:
            input_emu[par] = dict_mapping["forest_out"][par][jj, iz]

        input_emu["bias"] = -np.abs(input_emu["bias"])

        all_P1D_ff[jj, iz] = model_Arinyo.P1D_Mpc(
            zz[iz], k_par, input_emu, new_cosmo_params=new_cosmo
        )

        # print(new_cosmo)
        # print(input_emu)

        input_emu = {}
        for par in pars_list:
            input_emu[par] = dict_mapping["emu_params"][par][jj, iz]

        
        # print(input_emu)

        all_P1D_lace[jj, iz] = lace_mpg.emulate_p1d_Mpc(input_emu, k_par)

# %%
# jj = 0
# for ii in range(nz-1):
#     col = f"C{ii}"
#     plt.plot(k_par, k_par * all_P1D_lace[jj, ii]/np.pi, col)

#     plt.plot(k_par, k_par * all_P1D_ff[jj, ii]/np.pi, col+"--")
    

# %%
jj = 2
np.mean(all_P1D_ff[:, jj:]/all_P1D_lace[:, jj:])

# %%
jj = 2
np.std(all_P1D_ff[:, jj:]/all_P1D_lace[:, jj:])

# %%
for ii in range(2,nz):
    plt.plot(k_par, np.mean(all_P1D_ff[:, ii]/all_P1D_lace[:, ii], axis=0)-1, label=f"z={zz[ii]:.2f}")
plt.xlabel(r"$k$ [h/Mpc]")
plt.ylabel(r"$P_\mathrm{1D}$ [Mpc$^3$]")
plt.legend()
plt.legend()

# %%
