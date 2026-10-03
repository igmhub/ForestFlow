"""
Analytic covariance statistics.
"""
from typing import Any
from numpy.typing import ArrayLike
import numpy as np

def compute_Gaussian_cov(kpar: ArrayLike, kperp: ArrayLike, P3D: ArrayLike, vol: Any) -> Any:
    """
    Compute diagonal Gaussian P3D standard deviations from mode counts.

    Parameters
    ----------
    kpar, kperp : ndarray, shape (n_kpar, n_kperp)
        Cartesian comoving wavenumber grid in ``1 / Mpc``.
    P3D : array_like
        P3D values flattened in the same cell order as the k grids, in
        ``Mpc**3``.
    vol : float
        Simulation volume in ``Mpc**3``.

    Returns
    -------
    ndarray
        Per-cell Gaussian standard deviations in ``Mpc**3``.
    """
    dkpar = kpar[1, 0] - kpar[0, 0]

    # logarithmic
    dkperp = kperp[0, 1:] - kperp[0, :-1]
    dkperp = np.append(dkperp, dkperp[-1] ** 2 / dkperp[-2])

    # get Gaussian covariance
    Nmodes = (vol / (2 * np.pi) ** 2) * kperp * dkperp[np.newaxis, :] * dkpar
    sigma = np.sqrt(2.0 * P3D**2 / Nmodes.reshape(-1))

    return sigma
