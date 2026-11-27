"""
Scientific names, parameter ordering, and array contracts for ForestFlow.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from lace.conventions import validate_wavenumber

ARINYO_PARAMETER_NAMES = (
    "bias",
    "bias_eta",
    "q1",
    "q2",
    "kvav",
    "av",
    "bv",
    "kp",
)

__all__ = [
    "ARINYO_PARAMETER_NAMES",
    "validate_finite_array",
    "validate_mu",
    "validate_wavenumber",
]


def validate_finite_array(
    values: Any, name: str, minimum: float | None = None
) -> np.ndarray:
    """
    Validate a finite numeric array with an optional lower bound.

    Parameters
    ----------
    values : array_like
        Candidate numerical values.
    name : str
        Field name included in validation errors.
    minimum : float, optional
        Inclusive lower physical bound.

    Returns
    -------
    ndarray
        Floating-point array with its original shape.

    Raises
    ------
    ValueError
        If values are non-finite or lie below ``minimum``.
    """
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)) or (
        minimum is not None and np.any(array < minimum)
    ):
        qualifier = "finite" if minimum is None else f"finite and >= {minimum}"
        raise ValueError(f"{name} must contain {qualifier} values")
    return array


def validate_mu(values: Any) -> np.ndarray:
    """
    Validate finite direction cosines in the interval ``[0, 1]``.

    Parameters
    ----------
    values : array_like
        Candidate direction cosines.

    Returns
    -------
    ndarray
        Floating-point direction cosines.

    Raises
    ------
    ValueError
        If values are non-finite or outside the closed interval.
    """
    array = validate_finite_array(values, "mu")
    if np.any((array < 0) | (array > 1)):
        raise ValueError("mu must lie in [0, 1]")
    return array
