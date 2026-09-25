Fitting Arinyo parameters
=========================

Supported API
-------------

ForestFlow supports joint fits of the analytic Arinyo model to one simulation
snapshot's P3D and P1D measurements through :class:`forestflow.fitting.ArinyoFitter`.
Import it only from ``forestflow.fitting``; its implementation location is not
part of the public API.

.. code-block:: python

   from forestflow.fitting import ArinyoFitter

   fitter = ArinyoFitter(kmax_3d=4.5, kmax_1d=7.0)
   fitter.prepare_simulation(simulation)
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

The supplied driver can fit every snapshot of a simulation and save ordinary
parameter names:

.. code-block:: console

   python scripts/fit_p3d/fit_pflux.py mpg_central --output results/mpg_central.npz

Historical implementations
--------------------------

``forestflow.fits.fit_p3d``, ``forestflow.fits.fit_p3dz``, and the remaining
files in ``scripts/fit_p3d`` are retained for reproducing older analyses. They
use superseded LaCE/ForestFlow model and data contracts and are not supported
for new work. Their replacement is the API above, with one fit per snapshot.
