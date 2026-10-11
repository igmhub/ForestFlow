"""Numerical diagnostics for a deterministic ForestFlow student."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CoefficientValidation:
    bias: np.ndarray
    rmse: np.ndarray
    mae: np.ndarray
    percentile_95_abs: np.ndarray
    percentile_99_abs: np.ndarray


def validate_coefficients(prediction: np.ndarray, reference: np.ndarray) -> CoefficientValidation:
    """Summarize transformed-space prediction residuals by output coefficient."""
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if prediction.shape != reference.shape or prediction.ndim != 2:
        raise ValueError("prediction and reference must have equal two-dimensional shapes")
    residual = prediction - reference
    absolute = np.abs(residual)
    return CoefficientValidation(residual.mean(0), np.sqrt((residual ** 2).mean(0)),
                                 absolute.mean(0), np.percentile(absolute, 95, axis=0),
                                 np.percentile(absolute, 99, axis=0))


@dataclass(frozen=True)
class ObservableValidation:
    rms_fractional: float
    percentile_99_abs_fractional: float
    maximum_abs_fractional: float


def validate_observable(student, teacher, fixtures, observable) -> ObservableValidation:
    """Compare an observable evaluated with student and teacher coefficients.

    ``fixtures`` supplies the physically consistent inputs and any linear-theory
    context required by ``observable(coefficients, fixture)``.  This deliberately
    avoids constructing a linear spectrum from a compressed conditioning vector.
    """
    residuals = []
    for fixture in fixtures:
        inputs = fixture["emu_params"]
        reference = np.asarray(observable(teacher.evaluate(inputs), fixture))
        prediction = np.asarray(observable(student.evaluate(inputs), fixture))
        if reference.shape != prediction.shape or not np.all(np.isfinite(reference)):
            raise ValueError("observable outputs must be finite and share a shape")
        scale = np.maximum(np.abs(reference), np.finfo(float).tiny)
        residuals.append(((prediction - reference) / scale).ravel())
    if not residuals:
        raise ValueError("at least one physical fixture is required")
    values = np.concatenate(residuals)
    absolute = np.abs(values)
    return ObservableValidation(float(np.sqrt(np.mean(values**2))),
                                float(np.percentile(absolute, 99)), float(absolute.max()))
