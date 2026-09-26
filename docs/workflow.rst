End-to-end workflow
===================

This guide follows one ForestFlow prediction from simulation data to P3D and
P1D. Loading the archive is useful for inspection and validation but is not
required merely to evaluate the pretrained emulator.

Data flow
---------

.. graphviz::

   digraph forestflow_workflow {
       graph [rankdir=LR, bgcolor="transparent"];
       node [shape=box, style="rounded,filled", fillcolor="#eef4fb"];
       Archive [label="GadgetArchive3D\nsimulation P3D/P1D"];
       Inputs [label="cosmology + IGM\ninput mapping"];
       Emulator [label="P3DEmulator\ncINN"];
       Coefficients [label="Arinyo coefficient\nmapping"];
       Model [label="ArinyoModel +\nlinear theory"];
       Spectra [label="P3D_Mpc / P1D_Mpc"];
       Validation [label="leave-one-out\nresidual covariance"];
       Archive -> Inputs [label="training/validation"];
       Inputs -> Emulator;
       Emulator -> Coefficients;
       Coefficients -> Model;
       Model -> Spectra;
       Archive -> Validation;
       Spectra -> Validation;
   }

1. Inspect the archive
----------------------

``GadgetArchive3D`` extends the LaCE archive with measured P3D, P1D,
and precomputed Arinyo fits:

.. code-block:: python

   from forestflow.archive import GadgetArchive3D

   archive = GadgetArchive3D()
   training = archive.training_data
   testing = archive.get_testing_data("mpg_central")
   sample = testing[0]
   print(sample["z"], sample["cosmo_params"])

Archive files retain historical keys such as ``k_Mpc`` and
``p1d_Mpc``. Public prediction code uses ``k_iMpc``,
``P1D_Mpc``, and ``P3D_Mpc``; see :doc:`conventions`.

2. Load the emulator
--------------------

.. code-block:: python

   from forestflow.P3D_cINN import P3DEmulator

   emulator = P3DEmulator(key="forest_mpg")
   print(emulator.input_labels)
   print(emulator.output_labels)

``input_labels`` is the authoritative normalized input order. Prefer a
mapping over a positional array.

3. Predict Arinyo coefficients
------------------------------

The compressed cosmology must describe the same cosmology supplied to the
Arinyo model:

.. code-block:: python

   from lace.cosmo import cosmology

   z = 3.0
   cosmo = cosmology.Cosmology()
   linear_parameters = cosmo.get_linP_Mpc_params(z=z, kp_Mpc=0.7)
   emulator_input = {
       "Delta2_p": linear_parameters["Delta2_p"],
       "n_p": linear_parameters["n_p"],
       "mF": 0.23,
       "sigT_Mpc": 0.10,
       "gamma": 1.21,
       "kF_Mpc": 14.2,
   }
   arinyo = emulator.evaluate(emulator_input, seed=0)

The result is keyed by ``emulator.output_labels``. Inputs outside the
training domain are extrapolations; inspect the archive training sample before
interpreting them.

For several redshifts or parameter points, pass a list of dictionaries. The
network then evaluates them in one batch instead of making one Python call per
redshift:

.. code-block:: python

   inputs_by_redshift = [input_at_z2, input_at_z3, input_at_z4]
   arinyo_by_redshift = emulator.evaluate(inputs_by_redshift, seed=0)

Each returned value has one entry per input dictionary. Repeated calls with the
same batch size, ``Nrealizations``, and seed reuse the deterministic latent
sample tensor.

Long minimization or sampling runs can also compile the neural network once:

.. code-block:: python

   emulator = P3DEmulator(key="forest_mpg", compile_model=True)
   emulator.evaluate(inputs_by_redshift)  # compiles this input shape
   arinyo_by_redshift = emulator.evaluate(inputs_by_redshift)  # reuses it

The first call for a new input shape pays the compilation cost. Compilation is
therefore opt-in and is most useful when later calls keep the same number of
redshifts. The portable model saved on disk remains the ordinary PyTorch model.

4. Compute P3D and P1D
----------------------

.. code-block:: python

   import numpy as np
   from forestflow.model_p3d_arinyo import ArinyoModel

   model = ArinyoModel(cosmo)
   linear = model.linear_theory(z)
   k_iMpc = np.geomspace(0.1, 4.0, 80)
   mu = np.linspace(0.0, 1.0, 6)
   k2d_iMpc, mu2d = np.meshgrid(k_iMpc, mu, indexing="ij")

   P3D_Mpc = model.P3D_Mpc_k_mu(
       linear, z, k2d_iMpc, mu2d, arinyo
   )
   P1D_Mpc = model.P1D_Mpc(linear, z, k_iMpc, arinyo)

The outputs match their input grid shapes. For velocity coordinates use
:func:`forestflow.p1d.P1D_kms` with ``k_ikms`` and
``dkms_diMpc = H(z)/(1+z)``.

5. Interpret uncertainty
------------------------

``P3DEmulator.evaluate`` averages cINN latent realizations. Increasing
``Nrealizations`` reduces Monte Carlo noise in that mean; their scatter is
not automatically a calibrated prediction error, and ``evaluate`` does not
return a covariance.

For emulator uncertainty, use leave-one-simulation-out residuals from
``forestflow.covariance``. Their covariance describes prediction residuals
over the validated simulations. Gaussian-noise helpers and the covariance
tutorials instead estimate finite-volume/sample variance. These uncertainties
answer different questions and should not be interchanged without an explicit
analysis model.

Continue with :doc:`tutorials`, consult :doc:`conventions` for array and
unit contracts, and use :doc:`api` for individual functions.

Training a new emulator
-----------------------

With ``train=True`` and ``use_val_set=True``, every improvement in
validation negative log likelihood snapshots the model state. After training,
ForestFlow restores and saves the minimum-validation-loss state instead of the
final epoch. Metadata records the zero-based ``best_epoch`` and
``best_validation_loss``, and loading exposes both as emulator attributes.
Without validation, the final epoch is saved and
``best_validation_loss`` is ``None``.
