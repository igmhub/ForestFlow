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

# ## Arinyo parameters based on MP-Gadget

Archive3D = GadgetArchive3D()

# redshift range
zmin = 2.25
zmax = 2.50
priors = Archive3D.get_priors_Arinyo(zmin, zmax)
priors["mean"]

Archive3D.get_priors_IGM(zmin, zmax)



params = "Arinyo_min"
val_params = []
for sim in Archive3D.training_data:
    if(sim["z"] >= 2) & (sim["z"] < 2.75):
        val_params.append(sim[params])

# +
nsims = len(val_params)
name_params = val_params[0].keys()
nparams = len(name_params)
arr_val_params = np.zeros((nsims, nparams))

for ii in range(nsims):
    for jj, pname in enumerate(name_params):
        arr_val_params[ii, jj] = val_params[ii][pname]
# -

fig, ax = plt.subplots(4, 2)
ax = ax.reshape(-1)
print("From MP-Gadget sims between z=2 and 2.75")
print("par", "mean", "std", "min", "max")
for jj, pname in enumerate(name_params): 
    ax[jj].hist(arr_val_params[:,jj], bins=20);
    ax[jj].set_xlabel(pname)
    
    ymean = arr_val_params[:,jj].mean()
    ystd = arr_val_params[:,jj].std()
    ymin = arr_val_params[:,jj].min()
    ymax = arr_val_params[:,jj].max()
    print(pname, np.round(ymean, 3), np.round(ystd, 3), np.round(ymin, 3), np.round(ymax, 3)) 
plt.tight_layout()

plt.hist(arr_val_params[:,2], bins=30);
plt.xlabel("q1")
plt.tight_layout()
# plt.savefig("prior_q1_z2_z275.png")

from corner import corner

corner(arr_val_params);


