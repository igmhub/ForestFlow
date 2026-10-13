# Direct GP emulator of ForestFlow's transformed Arinyo training targets

Status: proposed production implementation, with accompanying executable prototype.
Date: 2026-10-11.

## Recommendation and scientific target

Test a GP before investing further in large-scale MLP distillation. Six inputs
and eight fitted outputs are a plausible GP problem, but speed and accuracy must
be measured. Unlike the MLP student, this is a new regression of the simulation
fit targets. It may improve or degrade physical predictions relative to the flow.
The GP predicts Arinyo outputs, not the cosmology/IGM conditioning inputs.

Use the six input labels and eight output labels from `forest_mpg_fix`, with
`Transf_data.transf_stand(..., direct=True)` for inputs and outputs and the same
inverse output transform for inference. The target recipe in the maintained
ForestFlow training tutorial is `Cabayol23_fixp3d`, `arinyo_fixp3d`, z <= 4.6,
with central snapshots included. Audit the actual bundle training provenance
before claiming an exact reproduction: a tutorial recipe alone does not prove
which archive revision or subset produced a checkpoint.

The prediction is inverse_transform(GP transformed posterior mean). It is not
the physical posterior mean under nonlinear inverse transformations. In
particular log-bias/kvav corrections must not be silently added.

## What to reuse from LaCE

The maintained `lace/emulator/gp_emulator_multi.py` uses sklearn
`GaussianProcessRegressor`, anisotropic `Matern(nu=0.5)`, and its public
`optimizer` (L-BFGS-B with maxiter=1000). It also splits large datasets into
overlapping mean-flux experts. Reuse the regression backend and optimizer first;
do not call its P1D-specific constructor, shape fit or mean-flux normalization.

Start with one multi-target GP with shared kernel hyperparameters. This jointly
fits length scales using all outputs but does not learn cross-output covariance.
Compare independent per-output kernels only if residuals justify the additional
training/inference cost. Start with nu=0.5 for a LaCE baseline, then 1.5 and 2.5
for smoother predictions and derivatives. Keep target normalization disabled
because ForestFlow already supplies the scaling.

## Data, scaling and leakage

1. Extract physical inputs, fitted targets, simulation IDs, redshift, rescaling
   IDs and matched linear-theory context from the corrected archive.
2. Check finite values, transform domains, label order and duplicate inputs.
   Report duplicate/conflicting targets; do not hide them with arbitrary jitter.
3. Split by simulation, retaining every redshift/rescaling of a simulation in
   one partition. Use validation simulations for kernel/nugget selection and an
   untouched test group for final reporting. For stronger claims, perform full
   leave-one-simulation-out validation.
4. Primary generalization experiment: fit the same transformation rules using
   training-fold statistics only (Transf_data with compute_fisher=False).
   Separately report exact frozen-bundle-transform results for compatibility.
   Those stored moments may include held-out simulations; label that explicitly.
5. The pretrained flow may have seen the GP test simulations. A comparison to
   it is diagnostic, not a fair held-out contest. A fair contest needs matched
   leave-one-out flow models or simulations held out from both models.

## Noise and capacity

Fit targets have estimation error and parameter degeneracies. Do not assume
perfect interpolation is desirable. Separate numerical jitter from a physical
fit-noise model. Start with explicit alpha variances (1e-6, 1e-4, 1e-2 in
standardized target units) selected on validation groups. Record convergence
warnings and length scales hitting bounds. Later propagate available fit
covariances through the transform Jacobian; per-output noise requires separate
GPs, and correlated fit noise requires a richer model.

Dense training is O(N^3) time/O(N^2) memory; mean inference scales with N and
variance inference costs more. Count rows before training. The prototype uses
a stated subset in smoke mode. Production should reuse/refactor LaCE's expert
strategy if needed, with an explicit one-expert case, overlap continuity tests,
and validation near mean-flux boundaries. Do not copy private methods blindly.

## Production APIs and artifacts

Proposed `forestflow/emulator/arinyo_gp.py`:

- `prepare_gp_training_data(records, fit_key, transform, labels)` with data audit.
- `train_arinyo_gp(data, groups, config)` with group splits and validation report.
- `ArinyoGPEmulator.predict_transformed(inputs, return_std=False)`.
- `ArinyoGPEmulator.evaluate(inputs)` returning the physical coefficient mapping.
- `save/load` with bundle version, transforms, ordered labels, kernel/nugget,
  fit key, z range, archive/input hashes, group split, seed, dependency versions,
  training revision/dirty state and domain metadata.

Default inference returns means only. GP standard deviations live in transformed
space and are not automatically calibrated physical or observational covariance.
Propagate draws through inverse transform and power calculation only for a
separately labelled uncertainty experiment. Retain latent-output independence
limitations. Do not register a production model alias until validation passes.

## Validation and acceptance

Report transformed RMSE, signed bias, p95/p99/max residuals per coefficient,
physical errors (avoid fractional errors near zero), and residuals versus each
input, redshift and simulation. Validate GP -> ArinyoModel P1D/P3D against both
the fitted-target spectra and measured simulation spectra using matched cosmology,
linear spectrum and fixed projection. The former isolates regression error; the
latter includes fit/model error and decides physical usefulness.

Compare GP and converged flow at identical inputs, but label this as disagreement.
Check teacher count/independent-seed convergence for that comparison. Benchmark
warm scalar and redshift-batch means separately from predictive variances, with
fixed CPU threads; then benchmark full P1D/likelihood if available. Measure actual
speedup rather than assuming GP is faster than a small MLP.

Provisional extra-regression budget: P1D RMS 0.2%, p99 0.5% and P3D RMS 0.3%,
p99 1%, relative to spectra from fitted coefficients on declared held-out grids.
These are starting budgets, not measured guarantees. Require simulation-spectrum
accuracy to remain scientifically acceptable. Validate likelihood impact where
data/covariance exist; power error alone is not a guarantee of negligible bias.

Tests: forward/inverse transforms and label order; unequal batch/output axes;
simulation group isolation; serialization prediction equality; all model assets
immutable; one-expert and boundary cases; deterministic repeated means; finite
outputs and meaningful failure messages. Preserve existing pretrained regressions.

## Implementation order and notebooks

1. Audit actual recipe/provenance, data counts and duplicates.
2. Run `01_train`: grouped splits, transform, kernel/nugget comparison and save.
3. Run `02_predict`: scalar/batch coefficients, inverse consistency, timing.
4. Run `03_validate`: untouched fits, physical coefficients, power impact and
   flow disagreement. Inspect failures before selecting a production route.
5. Promote the validated prototype to the package; add expert scaling only if
   dense GP measurements require it. Add full simulation-spectrum validation and
   matched leave-one-out model comparisons before scientific replacement.

The notebooks deliberately keep prototype scope separate from this production
plan. A successful smoke run does not establish any acceptance gate.
