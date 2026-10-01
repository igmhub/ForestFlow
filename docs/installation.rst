Installation
============

ForestFlow requires Python 3.12 or newer. Install the package from a checkout
with its runtime dependencies:

.. code-block:: console

   python -m pip install -e .

The package relies on LaCE for cosmology and simulation archive support; it is
declared as a runtime dependency. Torch and FrEIA remain mandatory because the
supported cINN emulator uses them. Paper-analysis helpers, the cross-power
routines, and MPI scripts are optional:

.. code-block:: console

   python -m pip install -e ".[extended]"

To build this documentation, install the documentation extra and invoke the
Make target:

.. code-block:: console

   python -m pip install -e ".[docs]"
   make docs

The generated site is written to ``docs/_build/html``.
