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
# # P1D chain -> ForestFlow -> P3D

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

from forestflow.priors import loaders, plots

np.__version__

# %% [markdown]
# #### Load data

# %%
dict_mapping = loaders.load_map_igm_p3d(emu_type="legacy")
# model systematic of bias_delta and beta, relative errors:
#   "accel2_offset": common offset from the accel2_priors notebook (floor sigma_mu/3)
#   "fixed": constant 1.0% for bias_delta and 3.1% for bias_eta, independent;
#            beta = bias_eta f / bias_delta gets the propagated 3.26% (loaders.SYS_FIXED)
# SYS_OPTION = "accel2_offset"
SYS_OPTION = "fixed"
sys_shift = loaders.load_sys_shift(SYS_OPTION)

# %% [markdown]
# Table with output of P1D chain

# %%
plots.table_cosmo_igm(dict_mapping)

# %%
bao_data = loaders.load_BAO_data()

# %%
# rescale_bias: plot b_delta [(1+2.33)/(1+z)]^2.9, the evolution assumed for the BAO band
plots.plot_bias_beta_zev(bao_data, dict_mapping, sys_shift=sys_shift, rescale_bias=True)
# plt.gcf().savefig("bias_beta_BAOvsP1D.png")
# plt.gcf().savefig("bias_beta_BAOvsP1D.pdf")

# %%
plots.plot_p3d_small_z(dict_mapping, sys_shift=None)

# %% [markdown]
# The comparison between the P1D from `lace_mpg` and from ForestFlow for points of the
# DR1 P1D chain is in `compare_lace_forestflow_p1d.ipynb`.
