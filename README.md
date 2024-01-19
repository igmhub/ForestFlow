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

Archive tutorial: notebooks/Tutorial_archive.ipynb
Emulator tutorial: notebooks/Tutorial_emulator.ipynb


## Installation
(Last update Jan 19 2024)

- Create a new conda environment. It is usually better to follow python version one or two behind. In January 2024, the latest is 3.12, so we recommend 3.11.

```
conda create -n forestflow python=3.11 camb
conda activate forestflow
```
- Install LaCE:

```Follow the instructions from https://github.com/igmhub/LaCE```

- Clone the ForestFlow repo and perform an *editable* installation:

```
git clone https://github.com/igmhub/ForestFlow.git
cd ForestFlow
pip install -e .[jupyter]
``` 

- Generate notebooks:

```
jupytext --to ipynb notebooks/*.py
```

- If you want to use notebooks via JupyterHub, you'll also need to download `ipykernel`:

```
pip install ipykernel
python -m ipykernel install --user --name forestflow --display-name forestflow
```
