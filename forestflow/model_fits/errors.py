"""
Effective fitting-error prescriptions.
"""
from typing import Any
from numpy.typing import NDArray
import numpy as np


def gaussian_p3d_relative_error(mode_counts, count_convention="full", fractional_floor=0.0):
    """
    Return Gaussian fractional P3D errors from Fourier-mode counts.

    Parameters
    ----------
    mode_counts : array_like
        Positive number of modes contributing to each P3D bin. With the
        default convention this includes both members of conjugate Fourier
        pairs.
    count_convention : {"full", "independent"}, default="full"
        Whether ``mode_counts`` includes conjugate pairs (``"full"``) or
        counts statistically independent complex modes directly.
    fractional_floor : float, default=0.0
        Non-negative diagonal fractional error combined in quadrature with
        the Gaussian mode-count uncertainty.

    Returns
    -------
    numpy.ndarray
        Fractional standard deviations with the shape of ``mode_counts``.

    Raises
    ------
    ValueError
        If counts are non-finite or non-positive, or if an option is invalid.

    Notes
    -----
    A real field has half as many independent complex modes as a full Fourier
    grid. Therefore the default error is ``sqrt(2 / N_full)``.
    """
    counts = np.asarray(mode_counts, dtype=float)
    if np.any(~np.isfinite(counts)) or np.any(counts <= 0):
        raise ValueError("mode_counts must be finite and strictly positive")
    if count_convention == "full":
        variance = 2.0 / counts
    elif count_convention == "independent":
        variance = 1.0 / counts
    else:
        raise ValueError("count_convention must be 'full' or 'independent'")
    if not np.isfinite(fractional_floor) or fractional_floor < 0:
        raise ValueError("fractional_floor must be finite and non-negative")
    return np.sqrt(variance + float(fractional_floor) ** 2)

def _get_err_p1d(x: Any, alpha: int | None=4, xmin: float | None=0.1, xmax: int | None=5, ymin: int | None=1, ymax: int | None=4) -> NDArray[Any]:
    """
    Evaluate the empirical relative P1D fitting-error curve.

    Parameters
    ----------
    x : array_like
        Wavenumber-like coordinate used by the empirical prescription.
    alpha : float, default: 4
        Power-law exponent above ``xmin``.
    xmin, xmax : float, default: 0.1, 5
        Lower reference and scale range of the prescription.
    ymin, ymax : float, default: 1, 4
        Baseline and amplitude coefficients.

    Returns
    -------
    ndarray
        Empirical non-negative relative-error values with shape of ``x``.
    """
    t = np.clip((x - xmin) / (xmax - xmin), 0.0, None)
    return ymin + ymax * t**alpha

def _get_err_p3d(x: Any, alpha: float | None=0.2, xmin: float | None=0.1, xmax: int | None=5, ymin: int | None=3, ymax: int | None=20) -> NDArray[Any]:
    """
    Evaluate the empirical relative P3D fitting-error curve.

    Parameters
    ----------
    x : array_like
        Wavenumber-like coordinate used by the empirical prescription.
    alpha : float, default: 0.2
        Power-law exponent above ``xmin``.
    xmin, xmax : float, default: 0.1, 5
        Lower reference and scale range of the prescription.
    ymin, ymax : float, default: 3, 20
        Small- and large-coordinate relative-error coefficients.

    Returns
    -------
    ndarray
        Empirical relative-error values with shape of ``x``.
    """
    t = np.clip((x - xmin) / (xmax - xmin), 0.0, None)
    return ymax - (ymax - ymin) * t**alpha
