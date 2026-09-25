# ForestFlow

[![Documentation Status](https://readthedocs.org/projects/igmhubforestflow/badge/?version=latest)](https://igmhubforestflow.readthedocs.io/en/latest/)

Lyman-alpha Cosmology Emulator. This code is a normalising flow emulator for the 3D flux power spectrum of the Lyman-alpha forest.

## Documentation

The documentation includes the user guide and complete API reference generated
from NumPy-style docstrings.

- [Online documentation](https://igmhubforestflow.readthedocs.io/en/latest/)
- [Documentation source](https://github.com/igmhub/ForestFlow/tree/main/docs)

To build the documentation locally:

```bash
python -m pip install -e ".[docs]"
make docs
```

Open `docs/_build/html/index.html` after the build completes. Read the Docs uses
the repository's `.readthedocs.yaml` file to perform the same build online.

### Relationship between the three packages

Install LaCE before ForestFlow; ForestFlow uses its cosmology and simulation
archive interfaces. cup1d is downstream and is only needed by historical or
paper-specific analysis modules. These sibling projects are installed directly
from their IGMHub repositories rather than declared under potentially ambiguous
PyPI package names. The CI workflow follows the same policy.

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

## Tutorials and notebooks

Start with the [end-to-end workflow](https://igmhubforestflow.readthedocs.io/en/latest/workflow.html),
which follows data from a simulation archive through emulator coefficients to
P3D and P1D predictions and explains the available uncertainty products.

- [Archive tutorial](notebooks/Tutorials/Tutorial_archive.ipynb)
- [Emulator tutorial](notebooks/Tutorials/Tutorial_emulator.ipynb)
- [Arinyo-model tutorial](notebooks/Tutorials/Tutorial_Arinyo.ipynb)
- [Cross-power tutorial](notebooks/Tutorials/Tutorial_Pcross.ipynb)
- [P1D covariance tutorial](notebooks/Tutorials/covariance/Tutorial_P1D_cov.ipynb)
- [Training-input tutorial](notebooks/Tutorials/training/Tutorial_cook_input.ipynb)

`notebooks/Tutorials` contains supported user examples;
`notebooks/Figures` reproduces publication figures;
`notebooks/priors` studies priors; `notebooks/emulator` contains emulator
diagnostics; and `notebooks/developers` contains exploratory or legacy work.
See the [notebook guide](https://igmhubforestflow.readthedocs.io/en/latest/tutorials.html)
before choosing an example.


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
documentation tools as well, use:

```bash
python -m pip install -e ".[docs]"
```

### Running tests

Run the complete test suite with:

```bash
make test
```

### Optional features

Install the dependencies required by the cross-power routines with:

```bash
python -m pip install -e ".[px]"
```

After installation, generate or refresh all Jupytext notebooks with:

```bash
python -m pip install jupytext
make notebooks
```

Run this from the ForestFlow repository root. The target searches only `ForestFlow/notebooks/`, recursively, and skips Jupyter checkpoint files.

To expose the environment as a Jupyter kernel:

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name forestflow --display-name forestflow
```

### Versioning

Package versions are derived from Git. Tagged releases use the tag; development builds include the commit distance and short SHA (for example, `1.2.0.dev4+gabc1234`). A dirty working tree adds `.dirty`. Source archives without Git metadata report `0+unknown`.
