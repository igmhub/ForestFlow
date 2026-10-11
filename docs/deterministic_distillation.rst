Deterministic student emulator
==============================

``forestflow.emulator.distillation`` trains a small fully connected network to
reproduce the transformed, latent-averaged output of a frozen ForestFlow model.
It is a speed-oriented approximation to a named teacher bundle. It does not
improve the physical accuracy of that teacher or supply an emulator covariance.

The target is the mean in the teacher's transformed-and-standardized output
space. The student uses the teacher's stored input normalization and output
inverse transform, so it preserves parameter ordering and transformations.

.. code-block:: python

   from forestflow.emulator.p3d_cinn import P3DEmulator
   from forestflow.emulator.distillation.data import (
       TeacherSamplingConfig, generate_teacher_dataset,
   )
   from forestflow.emulator.distillation.network import MLPConfig
   from forestflow.emulator.distillation.training import train_student
   from forestflow.emulator.distillation.inference import DeterministicArinyoEmulator

   teacher = P3DEmulator(key="forest_mpg_fix")
   train = generate_teacher_dataset(
       teacher, training_inputs,
       TeacherSamplingConfig(realizations=4096, sampler="sobol_antithetic"),
   )
   validation = generate_teacher_dataset(teacher, validation_inputs)
   result = train_student(MLPConfig(6, 8, hidden_dim=64, hidden_layers=3),
                          train, validation)
   student = DeterministicArinyoEmulator(
       result.model, teacher.transf_data, teacher.input_labels, teacher.output_labels,
   )
   physical_arinyo_parameters = student.evaluate(input_mapping)

Generate and split unique physical inputs before producing labels. Use
``build_input_design`` only with documented valid bounds, or provide its
``validity`` predicate. Validate against independent, higher-realization teacher
predictions, including physical P1D/P3D fixtures. Do not select architectures
on the held-out test set.

``MLPConfig`` covers the initial 2x32, 3x64, and 3x128 candidates. Select the
smallest network passing predeclared transformed-coefficient and observable
criteria. The 4,096 realization value is a starting point only: establish its
adequacy with independent-scramble convergence checks for the intended domain.
