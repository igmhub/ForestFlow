"""
Scientific names, parameter ordering, and array contracts for ForestFlow.
"""

from __future__ import annotations

from typing import Any
from warnings import warn

import numpy as np

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

LEGACY_UNIT_KEYS = {
    "k_Mpc": "k_iMpc",
    "k_kms": "k_ikms",
    "p1d_Mpc": "P1D_Mpc",
    "p3d_Mpc": "P3D_Mpc",
    "Pk_kms": "P1D_kms",
    "dkms_dMpc": "dkms_diMpc",
}


def canonicalize_unit_keys(values: dict[str, Any]) -> dict[str, Any]:
    """
    Temporarily add canonical names for legacy serialized mapping keys.

    Parameters
    ----------
    values : mapping
        Serialized or legacy mapping potentially using old field names.

    Returns
    -------
    dict
        Shallow copy containing canonical aliases for any recognized legacy
        fields.

    Warns
    -----
    FutureWarning
        New public mappings must use canonical unit-qualified names directly.
    directly. This helper will be removed in the next major release.
    """
    warn(
        "canonicalize_unit_keys is deprecated; provide canonical "
        "unit-qualified keys directly.",
        FutureWarning,
        stacklevel=2,
    )
    result = dict(values)
    for old, new in LEGACY_UNIT_KEYS.items():
        if new not in result and old in result:
            result[new] = result[old]
    return result


def validate_wavenumber(values: Any, *, name: str) -> np.ndarray:
    """
    Validate a finite, positive one-dimensional wavenumber array.

    Parameters
    ----------
    values : array_like
        Candidate wavenumbers.
    name : str
        Field name included in validation errors.

    Returns
    -------
    ndarray
        Floating-point one-dimensional wavenumbers.

    Raises
    ------
    ValueError
        If the array is not one-dimensional, finite, and strictly positive.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(f"{name} must be a finite, positive 1D array")
    return array


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
