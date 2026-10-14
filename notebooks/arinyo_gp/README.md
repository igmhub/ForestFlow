# Gaussian processes for transformed Arinyo parameters

Run `01_train`, `02_predict`, then `03_validate` with the ForestFlow/LaCE Python
environment. Each notebook has a maintained Jupytext `.py` and a generated
`.ipynb` counterpart. Paths work from this folder or the repository root.

These are executable experiments built on `forestflow.emulator.arinyo_gp`.
`experiment.py` contains only notebook plotting/timing helpers. The package
implementation reuses LaCE's GP optimizer and scikit-learn backend. No existing
model is overwritten.
The training notebook creates a fresh timestamped run under `artifacts/`;
the following notebooks read `artifacts/latest.json` (or an explicit run path).
Only load pickle artifacts produced locally by this experiment.

The default smoke run caps the training sample and optimizer budget. Its plots
and timing demonstrate execution, not scientific acceptance. Set `SMOKE=False`
for the full grouped experiment after checking available memory.

See [PLAN.md](PLAN.md) for the proposed production implementation and decision
criteria. The crucial distinction from deterministic distillation is the target:
here it is the archive's **fitted Arinyo coefficients**, not predictions of the
flow. ForestFlow comparison therefore measures model disagreement.
