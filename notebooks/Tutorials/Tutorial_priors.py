# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Extracting priors from ForestFlow
#
# - For the Arinyo parameters based on the DESI DR1 P1D fit (Chaves-Montero+26, https://arxiv.org/abs/2601.21432)
# - For the cosmo+IGM parameters based on the DESI DR1 P1D fit (Chaves-Montero+26, https://arxiv.org/abs/2601.21432)
# - For the Arinyo parameters based on the MP-simulations (Chaves-Montero+25, https://arxiv.org/abs/2409.05682)
# - For cosmo+IGM parameters based on the MP-simulations (Cabayol-Garcia+24, https://arxiv.org/abs/2305.19064)

# +
# %load_ext autoreload
# %autoreload 2

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

import forestflow
from forestflow.archive import GadgetArchive3D
# -

# ## Arinyo parameters based on DESI DR1 P1D fit

from forestflow.priors import get_arinyo_priors
# at a particular z
z = 3.
priors = get_arinyo_priors(z)
print(priors.keys())
priors["mean"]

# ## Cosmo and IGM parameters based on DESI DR1 P1D fit

from forestflow.priors import get_IGM_priors
# at a particular z
z = 3.
priors = get_IGM_priors(z)
print(priors.keys())
priors["mean"]

# ## Arinyo parameters based on MP-Gadget

Archive3D = GadgetArchive3D()

# redshift range
zmin = 2.25
zmax = 2.50
priors = Archive3D.get_Arinyo_priors(zmin, zmax)
priors["mean"]

Archive3D.get_IGM_priors(zmin, zmax)


