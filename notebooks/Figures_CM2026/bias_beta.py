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

from forestflow.priors.paper import plots, load

np.__version__

# %% [markdown]
# #### Load data

# %%
dict_mapping = load.load_map_igm_p3d()

# %% [markdown]
# Table with output of P1D chain

# %%
plots.table_cosmo_igm(dict_mapping)

# %%
bao_data = load.load_BAO_data()

# %%
plots.plot_bias_beta_zev(bao_data, dict_mapping)
# plt.gcf().savefig("bias_beta_BAOvsP1D.png")
# plt.gcf().savefig("bias_beta_BAOvsP1D.pdf")

# %%
plots.plot_p3d_small_z(dict_mapping)

# %% [markdown]
# The comparison between the P1D from `lace_mpg` and from ForestFlow for points of the
# DR1 P1D chain is in `compare_lace_forestflow_p1d.ipynb`.
