Gaussian-process Arinyo emulator
================================

``forestflow.emulator.arinyo_gp`` fits a Gaussian process directly to fitted
Arinyo coefficients in ForestFlow's transformed parameter space. It is a new
emulator of archive fit targets, not a deterministic approximation to a flow.

Use :func:`prepare_arinyo_gp_training_data` with a transform compatible with
``Transf_data``. To reproduce the input/output scaling of ``forest_mpg_fix``,
pass its stored transform. For a strict held-out experiment, build a transform
from the GP training fold only. Split by simulation labels before either option:
splitting individual redshift/rescaling rows leaks closely related data.

.. code-block:: python

   from forestflow.emulator.arinyo_gp import (
       ArinyoGPConfig, prepare_arinyo_gp_training_data, train_arinyo_gp,
   )
   from forestflow.emulator.p3d_cinn import P3DEmulator

   flow = P3DEmulator(key="forest_mpg_fix")
   data = prepare_arinyo_gp_training_data(
       training_records, flow.transf_data, flow.input_labels, flow.output_labels,
       fit_key="arinyo_fixp3d",
   )
   emulator = train_arinyo_gp(data, flow.transf_data, ArinyoGPConfig())
   arinyo = emulator.evaluate(input_mapping)

``predict_transformed(..., return_std=True)`` returns GP posterior standard
deviations in transformed coefficient space. Those values do not automatically
represent emulator or observational uncertainty, and inverse-transforming their
means differs from a physical-space posterior mean when transformations are
nonlinear.

The detailed experiment plan and three executable notebooks live in
``notebooks/arinyo_gp/``. A saved GP bundle uses pickle and must be treated as a
trusted local artifact.
