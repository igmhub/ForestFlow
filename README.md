# ForestFlow

Lyman-alpha Cosmology Emulator. This code is a normalising flow emulator for the 3D flux power spectrum of the Lyman-alpha forest.

## Emulator parameters:

These are the parameters that describe each individual P3D(k, mu) power spectrum. We have detached these from redshift and traditional cosmology parameters.

#### Cosmological parameters:

`Delta2_p` is the amplitude of the (dimensionless) linear spectrum at k_p = 0.7 1/Mpc

`n_p` is the slope of the linear power spectrum at k_p

#### IGM parameters:

`mF` is the mean transmitted flux fraction in the box (mean flux)

`sigT_Mpc` is the thermal broadening scale in comoving units, computed from `T_0` in the temperature-density relation

`gamma` is the slope of the temperature-density relation

`kF_Mpc` is the filtering length (or pressure smoothing scale) in inverse comoving units

#### Tutorials:

In the `Notebooks` folder, there are several tutorials one can run to learn how to use
the emulators and archives. 
Archive tutorial:TBD
Emulator tutorial:Tutorial_ForestFlow.ipynb


## Installation
(Last update Dec 9 2023)

- To install on NERSC, you first need to load python module with `module load python`. This is not necessary for personal computers. 

- Create a new conda environment. It is usually better to follow python version one or two behind. In October 2023, latest is 3.11, so we recommend 3.10.

```
conda create -n forestflow python=3.10
conda activate forestflow
```

- First clone the repo into your machine and perform an *editable* installation:

```
git clone git@github.com:igmhub/ForestFlow.git
cd ForestFlow
python setup.py install
``` 

- If you want to use notebooks via JupyterHub, you'll also need to download `ipykernel`:

```
pip install ipykernel
python -m ipykernel install --user --name forestflow --display-name forestflow
```

- REQUIREMENTS:

```numpy==1.24.4
pandas
scipy
h5py
scikit_learn
matplotlib
configobj
camb>=1.1.3
FreIA`
torch
corner
emcee```

```
Ìnstalling ForestFlow also requires installing LaCE (https://github.com/igmhub/LaCE)
```





