# Execution check

Executed the three maintained notebook Python sources in sequence against the
installed corrected archive with `MPLBACKEND=Agg`. All finished successfully
after fixing the prototype optimizer's serialization. The `.ipynb` files are
synced from these sources; notebook outputs are not pre-populated.

- Selected archive: 1,661 rows, including central snapshots, z <= 4.6.
- Smoke partitions: 96 training, 24 validation, 24 test rows; disjoint simulation
  groups. Exact stored forest_mpg_cinn transformation used.
- No exact duplicate training input rows detected.
- Validation transformed MSE: 0.04874 for Matérn nu=0.5, 0.02240 for nu=1.5,
  both with alpha=1e-4. Selected nu=1.5 using validation only.
- Save/reload predictions, scalar/batch consistency and transform round trip pass.
- Four-input mean timings in this one smoke run: GP ~0.171 ms and flow ~2.146 ms
  with 128 latent realizations. This is a small-training-set coefficient benchmark,
  not a production or likelihood speed claim.
- Power diagnostics ran on two test records with matched cosmologies. Relative
  to fitted-target spectra, aggregate fractional RMS was ~19% P1D and ~14% P3D.
  The second test case (mpg_18, z=4.5) dominates the poor result. Measured-spectrum
  P1D RMS for that case was ~29% for the GP versus ~0.43% for fitted coefficients.

These errors fail the proposed accuracy budgets. The default smoke settings are
only an execution check, not an accepted emulator. Full-data grouped training and
validation are still needed to assess the GP approach. Do not select new kernels
using these test results while retaining the same final-test claim.

No production emulator code, weights, archive files or LaCE APIs were changed.
