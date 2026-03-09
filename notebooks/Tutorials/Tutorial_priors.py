# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: cupix
#     language: python
#     name: cupix
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
arinyo_priors_DR1 = get_arinyo_priors(z)
print(arinyo_priors_DR1.keys())
arinyo_priors_DR1["mean"]

# ## Cosmo and IGM parameters based on DESI DR1 P1D fit

from forestflow.priors import get_IGM_priors
# at a particular z
z = 3.
igm_priors_DR1 = get_IGM_priors(z)
print(igm_priors_DR1.keys())
igm_priors_DR1["mean"]

# ## Arinyo parameters based on MP-Gadget

# load archive
Archive3D = GadgetArchive3D()

# redshift range
zmin = 3.
zmax = 3.
arinyo_priors_mpg = Archive3D.get_Arinyo_priors(zmin, zmax)
arinyo_priors_mpg["mean"]

# redshift range
zmin = 3.
zmax = 3.
igm_priors_mpg = Archive3D.get_IGM_priors(zmin, zmax)
igm_priors_mpg["mean"]

# ## Compare priors
#
# The DR1 analysis provies values of the parameters consistent with P1D observations, while the parameters from the MP-Gadget sims gives a plausible range of values. 
#
# As expected, the first should be included within the second

# It works
print("par     min mpg, central DR1, max mpg")
print("-------------------------------------")
for par in arinyo_priors_DR1["mean"].keys():
    print(
        par, 
        np.round(arinyo_priors_mpg["min"][par], 3), 
        np.round(arinyo_priors_DR1["mean"][par], 3),
        np.round(arinyo_priors_mpg["max"][par], 3),
        sep="\t"
    )

# It works
print("par            min mpg, central DR1, max mpg")
print("--------------------------------------------")
for par in igm_priors_DR1["mean"].keys():
    print(
        f"{par:<12}", 
        np.round(igm_priors_mpg["min"][par], 3), 
        np.round(igm_priors_DR1["mean"][par], 3),
        np.round(igm_priors_mpg["max"][par], 3),
        sep="\t"
    )




