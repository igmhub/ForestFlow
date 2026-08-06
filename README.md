# ForestFlow

Lyman-alpha Cosmology Emulator. This code is a normalising flow emulator for the 3D flux power spectrum of the Lyman-alpha forest.

## Documentation

The documentation includes the user guide and complete API reference generated
from NumPy-style docstrings.

- [Online documentation](https://forestflow.readthedocs.io/en/latest/)
- [Documentation source](https://github.com/igmhub/ForestFlow/tree/main/docs)

To build the documentation locally:

```bash
python -m pip install -e ".[docs]"
make docs
```

Open `docs/_build/html/index.html` after the build completes. Read the Docs uses
the repository's `.readthedocs.yaml` file to perform the same build online.

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

- Archive tutorial: notebooks/Tutorial_archive.ipynb
- Emulator tutorial: notebooks/Tutorial_emulator.ipynb


## Installation

ForestFlow requires Python 3.12 or newer. We recommend installing it in a
dedicated environment:

```bash
conda create -n forestflow python=3.12
conda activate forestflow
```

ForestFlow uses [LaCE](https://github.com/igmhub/LaCE) for cosmology and
simulation archive support. Install LaCE by following its installation
instructions, then clone and install ForestFlow:

```bash
git clone https://github.com/igmhub/ForestFlow.git
cd ForestFlow
python -m pip install -e .
```

The editable installation is recommended for development. To install the
documentation and testing tools as well, use:

```bash
python -m pip install -e ".[docs,test]"
```

### Optional features

Install the dependencies required by the cross-power routines with:

```bash
python -m pip install -e ".[px]"
```

To generate notebooks from the Jupytext sources:

```bash
python -m pip install jupytext
jupytext --to ipynb notebooks/*/*.py
```

To expose the environment as a Jupyter kernel:

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name forestflow --display-name forestflow
```
