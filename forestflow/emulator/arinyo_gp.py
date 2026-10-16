"""Gaussian-process emulation of transformed ForestFlow Arinyo fit targets.

This module is deliberately separate from :class:`P3DEmulator`: it learns the
archive's fitted Arinyo parameters, rather than imitating cINN predictions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import pickle
from typing import Any, ClassVar, Mapping, Sequence
from warnings import warn

import numpy as np

@dataclass(frozen=True)
class ArinyoGPConfig:
    """Kernel and numerical settings for an Arinyo target GP.

    The kernel is ``Matern(nu)`` with one length scale per input, optionally
    multiplied by a fitted amplitude (``fit_amplitude``). ``alpha`` is a fixed
    nugget variance in standardized target units; with a fitted amplitude, the
    effective noise-to-signal ratio is ``alpha / amplitude**2``.
    """

    nu: float = 0.5
    alpha: float = 1e-4
    n_restarts_optimizer: int = 0
    optimize_kernel: bool = True
    random_state: int = 0
    fit_amplitude: bool = False
    amplitude_bounds: tuple[float, float] = (1e-2, 1e3)
    length_scale_bounds: tuple[float, float] = (1e-5, 1e5)

    def __post_init__(self):
        if self.nu not in {0.5, 1.5, 2.5}:
            raise ValueError("nu must be one of 0.5, 1.5, or 2.5")
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and non-negative")
        if self.n_restarts_optimizer < 0:
            raise ValueError("n_restarts_optimizer must be non-negative")


@dataclass(frozen=True)
class ArinyoGPTrainingData:
    """Ordered transformed arrays and source metadata for one GP fit."""

    inputs: np.ndarray
    targets: np.ndarray
    input_labels: tuple[str, ...]
    output_labels: tuple[str, ...]
    group_ids: tuple[str, ...]
    fit_key: str

    def __post_init__(self):
        if self.inputs.ndim != 2 or self.targets.ndim != 2:
            raise ValueError("GP inputs and targets must be two-dimensional")
        if len(self.inputs) != len(self.targets) or len(self.inputs) != len(self.group_ids):
            raise ValueError("GP rows and group_ids must have equal length")
        if self.inputs.shape[1] != len(self.input_labels):
            raise ValueError("input labels do not match input matrix width")
        if self.targets.shape[1] != len(self.output_labels):
            raise ValueError("output labels do not match target matrix width")
        if not np.isfinite(self.inputs).all() or not np.isfinite(self.targets).all():
            raise ValueError("GP training arrays must be finite")


def _columns(records: Sequence[Mapping[str, Any]], labels: Sequence[str], fit_key: str | None = None):
    if not records:
        raise ValueError("at least one archive record is required")
    try:
        result = {
            name: np.asarray(
                [record[name] if fit_key is None else record[fit_key][name] for record in records],
                dtype=float,
            )
            for name in labels
        }
    except KeyError as error:
        location = "record" if fit_key is None else f"record[{fit_key!r}]"
        raise KeyError(f"missing {error.args[0]!r} in {location}") from error
    if not all(np.isfinite(values).all() for values in result.values()):
        raise ValueError("archive inputs or fitted targets contain non-finite values")
    return result


def _transform_matrix(transform, columns, labels, kind: str) -> np.ndarray:
    values = transform.transf_stand(columns, type_stand=kind, direct=True)
    result = np.column_stack([values[name] for name in labels])
    if not np.isfinite(result).all():
        raise ValueError("ForestFlow transformation yielded non-finite GP values")
    return result


def prepare_arinyo_gp_training_data(
    records: Sequence[Mapping[str, Any]],
    transform,
    input_labels: Sequence[str],
    output_labels: Sequence[str],
    *,
    fit_key: str = "arinyo_fixp3d",
    group_key: str = "sim_label",
) -> ArinyoGPTrainingData:
    """
    Transform archive inputs and fitted Arinyo targets in ForestFlow space.

    ``transform`` must be a :class:`Transf_data` compatible object. Passing the
    frozen ``forest_mpg_cinn`` transform gives compatibility with that model;
    constructing it from a training fold avoids held-out scaling leakage.
    """
    input_labels, output_labels = tuple(input_labels), tuple(output_labels)
    inputs = _transform_matrix(transform, _columns(records, input_labels), input_labels, "input")
    targets = _transform_matrix(
        transform, _columns(records, output_labels, fit_key), output_labels, "output"
    )
    try:
        groups = tuple(str(record[group_key]) for record in records)
    except KeyError as error:
        raise KeyError(f"missing grouping key {group_key!r}") from error
    return ArinyoGPTrainingData(inputs, targets, input_labels, output_labels, groups, fit_key)


def split_groups(
    group_ids: Sequence[str], *, validation_groups: int, test_groups: int, seed: int = 0
) -> tuple[set[str], set[str], set[str]]:
    """Randomly split complete simulation groups without row-level leakage."""
    groups = np.asarray(sorted(set(map(str, group_ids))))
    if validation_groups < 1 or test_groups < 1 or validation_groups + test_groups >= len(groups):
        raise ValueError("requested group split leaves an empty training partition")
    groups = np.random.default_rng(seed).permutation(groups)
    test = set(groups[:test_groups])
    validation = set(groups[test_groups:test_groups + validation_groups])
    train = set(groups[test_groups + validation_groups:])
    return train, validation, test


def select_groups(data: ArinyoGPTrainingData, groups: set[str]) -> ArinyoGPTrainingData:
    """Return complete rows whose simulation group belongs to ``groups``."""
    mask = np.asarray([group in groups for group in data.group_ids])
    if not mask.any():
        raise ValueError("selected group set has no training rows")
    return ArinyoGPTrainingData(
        data.inputs[mask], data.targets[mask], data.input_labels, data.output_labels,
        tuple(np.asarray(data.group_ids)[mask]), data.fit_key,
    )


@dataclass
class ArinyoGPEmulator:
    """Predict transformed or physical Arinyo coefficients with a multi-target GP."""

    gp: Any
    transform: Any
    input_labels: tuple[str, ...]
    output_labels: tuple[str, ...]
    metadata: dict[str, Any]

    # The GP is deterministic: there are no latent draws to configure, which
    # lets clients such as ``P1DEmulator`` skip the cINN sampling options.
    uses_sampling: ClassVar[bool] = False

    def _domain_value(self, name: str) -> Any:
        try:
            return self.metadata["model_domain"][name]
        except KeyError as error:
            raise AttributeError(
                f"This GP bundle does not define {name!r} domain metadata; "
                "train it with metadata={'model_domain': ...} (see "
                "P3DEmulator.model_domain)"
            ) from error

    @property
    def kp_iMpc(self) -> float:
        """Pivot wavenumber in inverse Mpc."""
        return self._domain_value("kp_iMpc")

    @property
    def kmax_3d_iMpc(self) -> float:
        """Maximum P3D wavenumber used in the fits, in inverse Mpc."""
        return self._domain_value("kmax_3d_iMpc")

    @property
    def kmax_1d_iMpc(self) -> float:
        """Maximum P1D wavenumber used in the fits, in inverse Mpc."""
        return self._domain_value("kmax_1d_iMpc")

    @property
    def zmax(self) -> float:
        """Maximum redshift covered by the training data."""
        return self._domain_value("zmax")

    @property
    def list_sim_cube(self) -> list[str]:
        """Training simulations."""
        return self._domain_value("list_sim_cube")

    def _input_matrix(self, emu_params: Mapping[str, float] | Sequence[Mapping[str, float]]):
        single = isinstance(emu_params, Mapping)
        rows = [emu_params] if single else list(emu_params)
        if not rows:
            raise ValueError("at least one input mapping is required")
        matrix = _transform_matrix(self.transform, _columns(rows, self.input_labels), self.input_labels, "input")
        return matrix, single

    @staticmethod
    def _mapping(values: np.ndarray, labels: tuple[str, ...], single: bool):
        result = {name: values[:, index] for index, name in enumerate(labels)}
        return {name: values[0] for name, values in result.items()} if single else result

    def predict_transformed(self, emu_params, *, return_std: bool = False):
        """Return transformed GP posterior means, optionally standard deviations.

        Returned standard deviations are in transformed ForestFlow output units.
        They are neither simulation-emulator uncertainty nor physical-space
        uncertainty after a nonlinear inverse transformation.
        """
        matrix, single = self._input_matrix(emu_params)
        if return_std:
            mean, std = self.gp.predict(matrix, return_std=True)
            return self._mapping(mean, self.output_labels, single), self._mapping(std, self.output_labels, single)
        return self._mapping(self._posterior_mean(matrix), self.output_labels, single)

    def _posterior_mean(self, matrix: np.ndarray) -> np.ndarray:
        """Posterior mean K(x, X_train) @ alpha, identical to ``gp.predict``.

        scikit-learn's ``predict`` also solves the n_train x n_train triangular
        system needed only for variances, even when they are not requested;
        that solve dominates the cost of mean-only predictions.
        """
        gp = self.gp
        mean = gp.kernel_(matrix, gp.X_train_) @ gp.alpha_
        mean = gp._y_train_std * mean + gp._y_train_mean
        return mean[:, None] if mean.ndim == 1 else mean

    def evaluate(self, emu_params):
        """Return physical Arinyo coefficients using the stored ForestFlow inverse transform."""
        return self.transform.transf_stand(
            self.predict_transformed(emu_params), type_stand="output", direct=False
        )

    def save(self, path: str | Path):
        """Persist a local, trusted Python bundle with transform and provenance."""
        path = Path(path)
        payload = {
            "schema_version": 1,
            "gp": self.gp,
            "transform": self.transform,
            "input_labels": self.input_labels,
            "output_labels": self.output_labels,
            "metadata": self.metadata,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path):
        """Load a bundle created locally by :meth:`save`.

        Pickle can execute code during loading; never load an untrusted artifact.
        """
        with Path(path).open("rb") as stream:
            payload = pickle.load(stream)
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported Arinyo GP bundle schema")
        return cls(
            payload["gp"], payload["transform"], tuple(payload["input_labels"]),
            tuple(payload["output_labels"]), dict(payload["metadata"]),
        )


def train_arinyo_gp(
    data: ArinyoGPTrainingData,
    transform,
    config: ArinyoGPConfig = ArinyoGPConfig(),
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ArinyoGPEmulator:
    """Fit a shared-kernel multi-output GP in transformed ForestFlow space."""
    # Keep these imports local: ordinary ForestFlow users and documentation
    # builds do not need to import the optional GP numerical stack.
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    from lace.emulator.gp_emulator_multi import optimizer as lace_gp_optimizer

    kernel = Matern(
        length_scale=np.ones(data.inputs.shape[1]),
        nu=config.nu,
        length_scale_bounds=tuple(config.length_scale_bounds),
    )
    if config.fit_amplitude:
        kernel = ConstantKernel(1.0, tuple(config.amplitude_bounds)) * kernel
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=config.alpha,
        normalize_y=False,
        optimizer=lace_gp_optimizer if config.optimize_kernel else None,
        random_state=config.random_state,
        n_restarts_optimizer=config.n_restarts_optimizer,
    ).fit(data.inputs, data.targets)
    details = dict(metadata or {})
    details.update({"config": asdict(config), "fit_key": data.fit_key, "groups": sorted(set(data.group_ids))})
    return ArinyoGPEmulator(gp, transform, data.input_labels, data.output_labels, details)


def transformed_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-output residual metrics in transformed ForestFlow space."""
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if prediction.shape != reference.shape or prediction.ndim != 2:
        raise ValueError("prediction and reference must be equal two-dimensional arrays")
    residual = prediction - reference
    if not np.isfinite(residual).all():
        raise ValueError("residuals must be finite")
    absolute = np.abs(residual)
    return {"bias": residual.mean(0), "rmse": np.sqrt((residual**2).mean(0)),
            "p99": np.percentile(absolute, 99, axis=0), "maximum": absolute.max(0)}
