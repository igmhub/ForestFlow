Installation
============

ForestFlow requires Python 3.12 or newer. Install the package from a checkout
with its runtime dependencies:

.. code-block:: console

   python -m pip install -e .

The package relies on LaCE for cosmology and simulation archive support. Follow
the LaCE installation instructions before importing archive- or model-related
modules.

To build this documentation, install the documentation extra and invoke the
Make target:

.. code-block:: console

   python -m pip install -e ".[docs]"
   make docs

The generated site is written to ``docs/_build/html``.
