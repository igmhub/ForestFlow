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

# # Get priors on Arinyo parameters
#
# From the MP-Gadget simulations

# +
# %load_ext autoreload
# %autoreload 2

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

import forestflow
from forestflow.archive import GadgetArchive3D


path_program = os.path.dirname(forestflow.__path__[0]) + '/'
path_program

# +
# %%time
folder_lya_data = path_program + "/data/best_arinyo/"

Archive3D = GadgetArchive3D(
    base_folder=path_program[:-1],
    folder_data=folder_lya_data,
    force_recompute_plin=False,
    average="both",
)
print(len(Archive3D.training_data))
# -

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


