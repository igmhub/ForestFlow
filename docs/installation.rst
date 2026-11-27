Installation
============

ForestFlow requires Python 3.12 or newer. Install the package from a checkout
with its runtime dependencies:

.. code-block:: console

   python -m pip install -e .

The package relies on LaCE for cosmology and simulation archive support; it is
declared as a runtime dependency. The minimal installation runs the
Gaussian-process emulator (``forest_mpg_gp``). The cINN emulator
(``forest_mpg_cinn``) needs torch and FrEIA:

.. code-block:: console

   python -m pip install -e ".[cinn]"

The priors analysis (``forestflow.priors``: ``.[priors]``, plus ``vega`` for
the BAO fits), the cross-power routines (``.[px]``) and the MPI scripts
(``.[mpi]``) are optional too; ``.[extended]`` installs all of them:

.. code-block:: console

   python -m pip install -e ".[extended]"

To build this documentation, install the documentation extra and invoke the
Make target:

.. code-block:: console

   python -m pip install -e ".[docs]"
   make docs

The generated site is written to ``docs/_build/html``.
