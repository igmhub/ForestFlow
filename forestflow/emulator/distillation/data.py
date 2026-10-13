"""Generation and storage of deterministic-emulator teacher labels."""

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence
from pathlib import Path
import json
import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True)
class TeacherSamplingConfig:
    """Controls for stochastic teacher labels used in distillation."""
    realizations: int = 4096
    seed: int = 0
    sampler: str = "sobol_antithetic"
    statistic: str = "mean"
    draw_policy: str = "nested"
    replicates: int = 1

    def __post_init__(self):
        if self.realizations < 1 or self.replicates < 1 or (self.sampler in {"antithetic", "sobol_antithetic"} and self.realizations % 2):
            raise ValueError("realizations must be positive and even for antithetic samplers")


@dataclass(frozen=True)
class DatasetSplit:
    """Normalized inputs and transformed teacher targets for one data split."""
    inputs: np.ndarray
    targets: np.ndarray
    input_labels: tuple[str, ...]
    output_labels: tuple[str, ...]

    def __post_init__(self):
        if self.inputs.ndim != 2 or self.targets.ndim != 2:
            raise ValueError("inputs and targets must be two-dimensional")
        if len(self.inputs) != len(self.targets):
            raise ValueError("input and target row counts differ")


@dataclass(frozen=True)
class TeacherConvergence:
    """Transformed-space uncertainty across independent teacher scrambles."""

    realizations: int
    replicate_rmse: np.ndarray
    replicate_max_abs: np.ndarray


def build_input_design(
    bounds: Mapping[str, tuple[float, float]], count: int, *, seed: int = 0,
    validity: Callable[[Mapping[str, float]], bool] | None = None,
) -> list[dict[str, float]]:
    """
    Create a reproducible scrambled-Sobol physical input design.

    ``bounds`` is only a proposal domain.  Callers should provide ``validity``
    when combinations in the bounding box are not physically supported.
    """
    if count < 1:
        raise ValueError("count must be positive")
    labels = tuple(bounds)
    lower = np.asarray([bounds[name][0] for name in labels], dtype=float)
    upper = np.asarray([bounds[name][1] for name in labels], dtype=float)
    if not np.all(np.isfinite(lower)) or not np.all(upper > lower):
        raise ValueError("each bound must be finite and increasing")
    # Oversample conservatively so a validity predicate does not correlate the
    # accepted prefix with Sobol ordering.
    engine = qmc.Sobol(len(labels), scramble=True, seed=seed)
    candidates = qmc.scale(engine.random_base2(int(np.ceil(np.log2(max(2 * count, 2))))), lower, upper)
    design = [dict(zip(labels, row)) for row in candidates]
    if validity is not None:
        design = [item for item in design if validity(item)]
    if len(design) < count:
        raise ValueError("validity rule rejected too many Sobol candidates")
    return design[:count]


def split_input_design(inputs: Sequence[Mapping[str, float]], *, validation_fraction: float = .1,
                       test_fraction: float = .1, seed: int = 0):
    """Split unique conditioning points before any teacher labels are generated."""
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must be positive and sum to less than one")
    if len(inputs) < 3:
        raise ValueError("at least three unique inputs are needed")
    indices = np.random.default_rng(seed).permutation(len(inputs))
    nval, ntest = int(round(len(inputs) * validation_fraction)), int(round(len(inputs) * test_fraction))
    if min(nval, ntest, len(inputs) - nval - ntest) < 1:
        raise ValueError("split leaves an empty partition")
    materialized = list(inputs)
    return (tuple(materialized[index] for index in indices[ntest + nval:]),
            tuple(materialized[index] for index in indices[ntest:ntest + nval]),
            tuple(materialized[index] for index in indices[:ntest]))


def normalized_inputs(teacher, inputs: Sequence[Mapping[str, float]]) -> np.ndarray:
    """Return teacher-normalized inputs in authoritative label order."""
    values = []
    for item in inputs:
        transformed = teacher.transf_data.transf_stand(item, type_stand="input", direct=True)
        values.append([transformed[name] for name in teacher.input_labels])
    return np.asarray(values, dtype=np.float32)


def _teacher_prediction(teacher, inputs, sampling: TeacherSamplingConfig, *, batch_size: int, seed: int):
    """Evaluate a deterministic batch partition with a given scramble seed."""
    target_chunks = []
    for start in range(0, len(inputs), batch_size):
        subset = list(inputs[start:start + batch_size])
        groups = list(range(start, start + len(subset)))
        result = teacher.evaluate_transformed(
            subset, Nrealizations=sampling.realizations, seed=seed,
            latent_group_ids=groups, sampler=sampling.sampler,
            statistic=sampling.statistic, draw_policy=sampling.draw_policy,
        )
        target_chunks.append(np.column_stack([result[name] for name in teacher.output_labels]))
    return np.concatenate(target_chunks).astype(np.float32) if target_chunks else np.empty((0, len(teacher.output_labels)), dtype=np.float32)


def generate_teacher_dataset(teacher, inputs: Sequence[Mapping[str, float]], sampling: TeacherSamplingConfig = TeacherSamplingConfig(), *, batch_size: int = 64) -> DatasetSplit:
    """
    Generate transformed-mean labels from a frozen ForestFlow teacher.

    ``inputs`` must already be a physically valid, split-isolated design.  The
    function uses stable group IDs so labels do not depend on batching.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    predictions = [_teacher_prediction(teacher, inputs, sampling, batch_size=batch_size,
                                       seed=sampling.seed + replicate)
                   for replicate in range(sampling.replicates)]
    targets = np.mean(predictions, axis=0, dtype=np.float32)
    if not np.isfinite(targets).all():
        raise ValueError("teacher generated non-finite transformed labels")
    return DatasetSplit(normalized_inputs(teacher, inputs), targets,
                        tuple(teacher.input_labels), tuple(teacher.output_labels))


def calibrate_teacher_sampling(teacher, inputs: Sequence[Mapping[str, float]],
                               configurations: Sequence[TeacherSamplingConfig], *, batch_size: int = 64):
    """
    Measure independent-scramble transformed-label scatter for each count.

    The reported scatter is a direct numerical diagnostic, not a calibrated
    physical emulator uncertainty.  Supply at least two replicates in every
    configuration used for a convergence decision.
    """
    results = []
    for config in configurations:
        if config.replicates < 2:
            raise ValueError("sampling calibration requires at least two independent replicates")
        predictions = np.stack([
            _teacher_prediction(teacher, inputs, config, batch_size=batch_size,
                                seed=config.seed + replicate)
            for replicate in range(config.replicates)
        ])
        residual = predictions - predictions.mean(axis=0, keepdims=True)
        results.append(TeacherConvergence(
            config.realizations,
            np.sqrt(np.mean(residual**2, axis=(0, 1))),
            np.max(np.abs(residual), axis=(0, 1)),
        ))
    return tuple(results)


def save_teacher_dataset(path: str | Path, dataset: DatasetSplit, *, metadata: Mapping[str, object] | None = None) -> None:
    """Write arrays and label ordering; metadata remains explicit and inspectable."""
    np.savez_compressed(path, inputs=dataset.inputs, targets=dataset.targets,
                        input_labels=np.asarray(dataset.input_labels),
                        output_labels=np.asarray(dataset.output_labels),
                        metadata=np.asarray(json.dumps(dict(metadata or {}), sort_keys=True)))


def load_teacher_dataset(path: str | Path) -> DatasetSplit:
    """Load a dataset written by :func:`save_teacher_dataset`."""
    with np.load(path, allow_pickle=False) as stored:
        return DatasetSplit(stored["inputs"], stored["targets"],
                            tuple(stored["input_labels"].tolist()),
                            tuple(stored["output_labels"].tolist()))
