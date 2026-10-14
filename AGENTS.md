# AGENTS.md — ForestFlow

## Purpose and active layout

ForestFlow emulates Arinyo flux-power parameters with a conditional invertible neural network and predicts P3D/P1D. LaCE owns cosmology and base archives; cup1d consumes the P1D adapter.

- Emulator/inference: `forestflow/emulator/p3d_cinn.py`, `network.py`, `training.py`, `bundle.py`; downstream P1D adapter: `forestflow/emulator/p1d.py`.
- Physical model and LaCE adapter: `forestflow/model/`; projections/binning/covariances: `forestflow/statistics/`; archive extensions: `forestflow/archive/`; model fitting: `forestflow/model_fits/`.
- Scientific contracts: `forestflow/conventions.py`; supported examples: `notebooks/Tutorials/`; publication reproductions: `notebooks/Figures/`.
- `forestflow/old_code/` is deprecated. `notebooks/developers/` includes exploratory/legacy work; paper-specific `priors/paper/` and plotting modules are not automatically production inference paths.

## Working rules

- Inspect `git status`, the current branch, and applicable nested instructions before editing. The maintained branch is `main`; do not switch branches or discard user changes automatically.
- Read the relevant implementation, tests, and `docs/workflow*` before changing a public interface. Follow active imports rather than assuming every notebook defines supported behavior.
- Make focused changes. Preserve scientific defaults, parameter ordering, serialization, and scalar/batch behavior unless the task explicitly changes them. Document intentional numerical changes and their validation.
- Do not edit `old_code/`, notebook `old/`, `wip/`, or developer experiments by default. Historical/paper modules are not necessarily deprecated: check callers first. Never copy an obsolete API back into the active package without checking it.
- Do not regenerate simulation archives, model weights, covariance products, chains, or publication outputs as part of a routine code change. Use configured external assets and report missing prerequisites. Never replace a scientific regression reference solely to make a test pass.
- Use Python >=3.12 and editable installs. Install sibling IGMHub repositories explicitly from compatible revisions, rather than relying on the PyPI name `lace`. Record the three commit SHAs for cross-package validation.
- Distinguish fast unit checks from model/data-dependent regression tests. A skipped or unavailable regression is not a pass. Prefer a small test of the failing scientific invariant over a test that merely reproduces the implementation.
- Maintain Jupytext `.py` notebook sources; sync only the affected pair with `jupytext --sync path/to/notebook.py`. Avoid generating every notebook for an unrelated change.
- Versions are derived from Git via setuptools-scm; do not hand-edit generated `_version.py` files. Update API docstrings and relevant documentation when behavior changes.
- Report what changed, commands actually run, missing assets/dependencies, and any numerical or scientific limitations.

## Shared scientific contracts

- Consult each package's `conventions.py`. Canonical public names include `k_iMpc`, `k_ikms`, `P1D_Mpc`, `P1D_kms`, `P3D_Mpc`, and `dkms_diMpc`. Existing serialized data and APIs retain legacy spellings; translate at explicit boundaries rather than silently renaming stored products.
- With `M = H(z)/(1+z)` in km/s/Mpc: `k_iMpc = M * k_ikms`, `P1D_kms = M * P1D_Mpc`, and P1D covariance gains two factors of M. P3D has volume units (Mpc^3); do not apply a P1D Jacobian to it.
- Preserve the distinction between comoving Mpc and Mpc/h, thermal broadening length `sigT_Mpc`, and inverse pressure smoothing scale `kF_Mpc`.
- Linear-power defaults distinguish baryon+CDM (`bc`) from total matter (`bcnu`). Check species, pivot, redshift, primordial running convention, and growth convention before comparing predictions.
- Primordial rescaling is only valid when all transfer-function/background parameters are unchanged. Changes in neutrino mass, effective relativistic species, dark energy, curvature, or densities require an appropriate fresh cosmology calculation.
- Treat scalar, redshift, k, batch, and stochastic-sample axes explicitly. Use unequal axis lengths in tests to expose accidental broadcasting; preserve ragged observational k grids.
- Covariance must preserve data ordering and selected cross-bin correlations. Validate symmetry, finite entries, and positive definiteness; do not hide invalid matrices with absolute determinants or arbitrary regularization.

## ForestFlow-specific safeguards

- Preserve input normalization, output transformations, `ARINYO_PARAMETER_NAMES` ordering, latent sampling/seeding, and model-bundle metadata. Compatible-looking array shapes are insufficient to establish checkpoint compatibility.
- Distinguish variation across latent samples from calibrated emulator prediction error and from simulation sample variance. State which uncertainty a covariance represents.
- Training changes must preserve validation isolation and best-validation-checkpoint selection. Do not retrain bundled scientific models during routine API work.
- Preserve the bias/bias_eta growth convention: the large-scale factor is `(bias + bias_eta*f*mu**2)**2`. Do not interchange beta and bias_eta without the required conversion.
- P1D projection uses `integral dln(k_perp) k_perp**2 P3D/(2*pi)`. Check transverse cutoffs and quadrature convergence against analytic examples and high-resolution integration.
- Match bin averaging to the estimator: log-coordinate averages, phase-space weights, and discrete mode averages are different operations. Preserve padded-grid masks and cell-edge conventions.
- Scalar and batched P1D must agree with controlled latent indices. Cosmology caches must key the actually cached cosmology, including an explicit return to fiducial after a perturbed call.
- Keep optional cross-power dependencies lazy; P3D/P1D users should not require `hankl` unless using cross-power features.

## Validation commands

After explicitly installing a compatible LaCE checkout:

```bash
python -m pip install -e ".[test]"
pytest -q -m "not pretrained_model"
pytest -q -m pretrained_model  # requires the pretrained scientific model assets
```

The Makefile provides `test-unit` and `test-regression` equivalents. Use `tests/test_scientific_units.py` for units, projections and binning; also select fitting, compiled inference, checkpoint and manifest tests as appropriate. Run pretrained regression for numerical/model changes when its prerequisites are available. Validate the cup1d adapter for interface changes. For documentation changes install `.[docs]` and run `make docs`.
