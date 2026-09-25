Notebook guide
==============

ForestFlow notebooks are paired Jupytext Python files and Jupyter notebooks.
Paths are case-sensitive: the supported directory is
``notebooks/Tutorials``.

Start here
----------

``notebooks/Tutorials/Tutorial_archive.ipynb``
   Load ``GadgetArchive3D``, inspect training and testing snapshots, and
   compare measured P3D and P1D with an Arinyo fit.

``notebooks/Tutorials/Tutorial_emulator.ipynb``
   Load the pretrained cINN, predict Arinyo coefficients, and compute spectra.

``notebooks/Tutorials/Tutorial_Arinyo.ipynb``
   Evaluate the analytic Arinyo model and vary its coefficients.

``notebooks/Tutorials/Tutorial_Pcross.ipynb``
   Compute transverse cross-power. Install the ``px`` optional dependencies.

Focused tutorials
-----------------

``notebooks/Tutorials/covariance``
   Finite-volume and leave-one-out covariance exercises.

``notebooks/Tutorials/training``
   Emulator input preparation and leave-one-out validation. These workflows
   require training data and are not needed for ordinary prediction.

Other directories
-----------------

``notebooks/Figures``
   Publication-figure reproduction using paper-specific data products.

``notebooks/priors``
   Arinyo, cosmological, and IGM prior studies.

``notebooks/emulator``
   Emulator covariance and derivative diagnostics.

``notebooks/developers``
   Exploratory, historical, or maintenance workflows; APIs and external paths
   here may not be supported.

Synchronize one pair after editing its Python source with:

.. code-block:: console

   jupytext --sync notebooks/Tutorials/Tutorial_emulator.py

Run ``make notebooks`` from the repository root to synchronize every
non-checkpoint notebook.
