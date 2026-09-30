"""Scientific names, parameter ordering, and array contracts for ForestFlow."""

from __future__ import annotations

from typing import Any

import numpy as np

ARINYO_PARAMETER_NAMES = (
    "bias",
    "beta",
    "q1",
    "kvav",
    "av",
    "bv",
    "kp",
    "q2",
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
    """Return a shallow copy containing canonical aliases for legacy keys."""
    result = dict(values)
    for old, new in LEGACY_UNIT_KEYS.items():
        if new not in result and old in result:
            result[new] = result[old]
    return result


def validate_wavenumber(values: Any, *, name: str) -> np.ndarray:
    """Return a finite, positive one-dimensional wavenumber array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(f"{name} must be a finite, positive 1D array")
    return array


def validate_finite_array(
    values: Any, name: str, minimum: float | None = None
) -> np.ndarray:
    """Return a finite array, optionally enforcing a lower physical bound."""
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)) or (
        minimum is not None and np.any(array < minimum)
    ):
        qualifier = "finite" if minimum is None else f"finite and >= {minimum}"
        raise ValueError(f"{name} must contain {qualifier} values")
    return array


def validate_mu(values: Any) -> np.ndarray:
    """Return finite direction cosines in the physical interval [-1, 1]."""
    array = validate_finite_array(values, "mu")
    if np.any(np.abs(array) > 1):
        raise ValueError("mu must lie in [-1, 1]")
    return array
