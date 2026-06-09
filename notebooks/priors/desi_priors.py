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
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

from forestflow.priors_paper import all_plots, load

np.__version__

# %% [markdown]
# #### Load data

# %%
dict_mapping = load.load_map_igm_p3d(lab_sample="desi")

# %% [markdown]
# Table with output of P1D chain

# %%
all_plots.table_cosmo_igm(dict_mapping)

# %%
bao_data = load.load_BAO_data()

# %%
all_plots.plot_bias_beta_zev(bao_data, dict_mapping, plot_bias_eta=False)

# %%
BAO DR1 & bias_delta & -0.1318 & 0.0059 \\
BAO DR1 & beta & 1.5429 & 0.0948 \\

BAO DR2 & bias_delta & -0.1302 & 0.0048 \\
BAO DR2 & beta & 1.5017 & 0.0659 \\

bias_delta 2.33: -0.1229 & 0.0061
beta 2.33: 1.4330 & 0.0493

# %%
61/48

# %%
659/493

# %%
all_plots.plot_p3d_small_z(dict_mapping)

# %%
