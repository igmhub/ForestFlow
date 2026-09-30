"""Analytic covariance statistics."""
from typing import Any
from numpy.typing import ArrayLike
import numpy as np

def compute_Gaussian_cov(kpar: ArrayLike, kperp: ArrayLike, P3D: ArrayLike, vol: Any) -> Any:

    # linear
    """
    Compute Gaussian covariance.

    Parameters
    ----------
    kpar : numpy.ndarray
        Kpar used by the calculation.
    kperp : numpy.ndarray
        Kperp used by the calculation.
    P3D : numpy.ndarray
        P3d used by the calculation.
    vol : object
        Vol used by the calculation.

    Returns
    -------
    object
        Result produced when the function is used to compute gaussian covariance.
    """
    dkpar = kpar[1, 0] - kpar[0, 0]

    # logarithmic
    dkperp = kperp[0, 1:] - kperp[0, :-1]
    dkperp = np.append(dkperp, dkperp[-1] ** 2 / dkperp[-2])

    # get Gaussian covariance
    Nmodes = (vol / (2 * np.pi) ** 2) * kperp * dkperp[np.newaxis, :] * dkpar
    sigma = np.sqrt(2.0 * P3D**2 / Nmodes.reshape(-1))

    return sigma
