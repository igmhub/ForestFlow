Fitting Arinyo parameters
=========================

Supported API
-------------

ForestFlow supports joint fits of the analytic Arinyo model to one simulation
snapshot's P3D and P1D measurements through :class:`forestflow.model_fits.ArinyoFitter`.
Import it only from ``forestflow.model_fits``; its implementation location is not
part of the public API.

.. code-block:: python

   from forestflow.model_fits import ArinyoFitter

   fitter = ArinyoFitter(kmax_3d=4.5, kmax_1d=7.0)
   fitter.prepare_simulation(simulation, is_mpg=True)
   result = fitter.fit_iterative()
   arinyo = fitter.params_to_dict(fitter.best_params)

``simulation`` is one mapping from ``GadgetArchive3D.training_data`` or
``GadgetArchive3D.get_testing_data``. It must include ``z``, ``cosmo_params``,
``k3d_Mpc``, ``mu3d``, ``p3d_Mpc``, ``k_Mpc``, ``p1d_Mpc``, and an initial
``Arinyo_min`` mapping. Wavenumbers are in inverse Mpc, P3D is in Mpc cubed,
and P1D is in Mpc. The fitter uses P3D in ``(k, mu)`` bins and performs the
P1D projection with the current ``ArinyoModel``.

The parameter-vector order is always ``ArinyoFitter.PARAM_NAMES``:
``bias``, ``bias_eta``, ``q1``, ``q2``, ``kvav``, ``av``, ``bv``, ``kp``.
Use ``params_to_dict`` and ``params_from_dict`` rather than relying on a local
ordering. ``fit`` performs one SciPy optimization and returns an
``OptimizeResult``; ``fit_iterative`` alternates optimizers and returns the
final result. After either call, ``best_params`` and ``best_chi2`` are set.

The ``best_chi2`` name is retained for compatibility, but it is not a
statistical chi-squared. ``ArinyoFitter`` minimizes the sum of the mean
squared, uncertainty-normalized fractional residuals of P3D and P1D. This
deliberately gives the two spectra equal aggregate weight, independently of
their number of bins; it is a fitting objective, not a goodness-of-fit
statistic.

The supplied driver can fit every snapshot of a simulation and save ordinary
parameter names:

.. code-block:: console

   python scripts/fit_p3d/fit_pflux.py mpg_central --output results/mpg_central.npy

Fitting MP-Gadget simulations
-----------------------------

Use ``scripts/fit_p3d/fit_pflux.py`` for reproducible, per-snapshot MPG fits.
It uses the same hybrid finite-volume P3D average as the supported
``ArinyoFitter`` API.  In particular, it does not merely evaluate the model at
mode-weighted bin centres: sparse cells are averaged over their actual Fourier
modes and dense cells use the continuous phase-space approximation.

Corrected Cabayol23 P3D fits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The corrected post-processing is selected explicitly and is the default for
the driver. Save results under the following directory and naming convention:

.. code-block:: console

   python scripts/fit_p3d/fit_pflux.py mpg_0 \
       --postproc Cabayol23_fixp3d \
       --output data/best_arinyo/cabayol23_fixp3d/Arinyo_fit_mpg_0.npy

Repeat for ``mpg_0`` through ``mpg_29``. The result file stores the fit
parameters, fitting objective, optimizer status, redshift, and complete
snapshot identity (snapshot, phase, axis, and optical-depth rescaling). The
parameter names are ordinary API names, never LaTex labels.

When a matching file is present in
``data/best_arinyo/cabayol23_fixp3d/``, ``GadgetArchive3D`` attaches its
per-snapshot parameters automatically as ``snapshot["arinyo_fixp3d"]``. This
keeps corrected fits separate from the historical ``Arinyo_min`` and
``Arinyo_lowk`` fields. Identity matching rather than list position protects
against mixing redshifts, axes, or optical-depth rescalings.

Testing simulations and their combination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fit the two testing realizations individually before fitting their combined
measurement:

.. code-block:: console

   python scripts/fit_p3d/fit_pflux.py mpg_central \
       --output data/best_arinyo/cabayol23_fixp3d/Arinyo_fit_mpg_central.npy

   python scripts/fit_p3d/fit_pflux.py mpg_seed \
       --output data/best_arinyo/cabayol23_fixp3d/Arinyo_fit_mpg_seed.npy

The archive also provides ``get_central_seed_average()``. It pairs central and
seed snapshots by their complete identities, averages ``mF``, ``T0``,
``gamma``, ``sigT_Mpc``, and ``kF_Mpc``, and combines P1D/P3D after converting
to absolute flux power with ``mF**2``. Fit that combined measurement with:

.. code-block:: console

   python scripts/fit_p3d/fit_pflux.py mpg_central_seed \
       --output data/best_arinyo/cabayol23_fixp3d/Arinyo_fit_mpg_central_seed.npy

Once saved there, ``archive.get_central_seed_average()`` likewise attaches the
result as ``arinyo_fixp3d``. The combined data are used in the fit; central and
seed data are never silently substituted for one another.

Interactive example
~~~~~~~~~~~~~~~~~~~

The `MP-Gadget Arinyo fitting notebook <https://github.com/igmhub/ForestFlow/blob/main/notebooks/arinyo_fits/Arinyo_fit_mpg.py>`_
shows one fit, the hybrid prediction and residual plots, and an optional loop
over all snapshots of one selected hypercube simulation. It is the best place
to modify scale cuts or inspect a result before starting a batch run.

Historical implementations
--------------------------

``forestflow.old_code.fits.fit_p3d``, ``forestflow.old_code.fits.fit_p3dz``, and the remaining
files in ``scripts/fit_p3d`` are retained for reproducing older analyses. They
use superseded LaCE/ForestFlow model and data contracts and are not supported
for new work. Their replacement is the API above, with one fit per snapshot.

Training ForestFlow emulators
-----------------------------

Use the batch driver to train reproducible cINN bundles from the corrected
``Cabayol23_fixp3d`` Arinyo fits.  It uses the production network settings
(400 epochs, six cINN layers, hidden dimension 30, batch size 8, and
``z <= 4.6``), records the fitted P3D and P1D cuts (4.5 and 6.0 ``iMpc``),
and writes transformations, weights, metadata, and a manifest together.

Train the full emulator on the 30 hypercube simulations plus ``mpg_central``::

   python scripts/training_l1O/train_emulator.py --full

This saves ``data/emulator_models/forest_mpg_fix``.  The default invocation
trains one leave-one-out bundle for every hypercube simulation::

   python scripts/training_l1O/train_emulator.py

To train the complete leave-one-out suite, including the emulator without
``mpg_central``, run the fixed batch wrapper with no arguments::

   scripts/training_l1O/train_all_l1O.sh

To train a selected leave-one-out model, for example excluding ``mpg_7``::

   python scripts/training_l1O/train_emulator.py --simulations mpg_7

The corresponding bundle is saved below ``data/emulator_models/l1O/``.  To
exclude the central simulation instead, run::

   python scripts/training_l1O/train_emulator.py --simulations mpg_central

which writes ``forest_mpg_fix_l1O_mpg_central``.  Existing complete bundles
are skipped; pass ``--overwrite`` only when deliberately retraining one.
