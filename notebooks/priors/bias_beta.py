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
plt.gcf().savefig("bias_beta_BAOvsP1D.png")
plt.gcf().savefig("bias_beta_BAOvsP1D.pdf")

# %%
all_plots.plot_p3d_small_z(dict_mapping)

# %%
