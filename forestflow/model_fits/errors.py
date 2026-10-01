"""Effective fitting-error prescriptions."""
from typing import Any
from numpy.typing import NDArray
import numpy as np

def _get_err_p1d(x: Any, alpha: int | None=4, xmin: float | None=0.1, xmax: int | None=5, ymin: int | None=1, ymax: int | None=4) -> NDArray[Any]:
    """
    Return err one-dimensional power spectrum.

    Parameters
    ----------
    x : object
        X used by the calculation.
    alpha : int, optional
        Alpha used by the calculation.
    xmin : float, optional
        Xmin used by the calculation.
    xmax : int, optional
        Xmax used by the calculation.
    ymin : int, optional
        Ymin used by the calculation.
    ymax : int, optional
        Ymax used by the calculation.

    Returns
    -------
    object
        Result produced when the function is used to return err one-dimensional power spectrum.
    """
    t = np.clip((x - xmin) / (xmax - xmin), 0.0, None)
    return ymin + ymax * t**alpha

def _get_err_p3d(x: Any, alpha: float | None=0.2, xmin: float | None=0.1, xmax: int | None=5, ymin: int | None=3, ymax: int | None=20) -> NDArray[Any]:
    """
    Return err three-dimensional power spectrum.

    Parameters
    ----------
    x : object
        X used by the calculation.
    alpha : float, optional
        Alpha used by the calculation.
    xmin : float, optional
        Xmin used by the calculation.
    xmax : int, optional
        Xmax used by the calculation.
    ymin : int, optional
        Ymin used by the calculation.
    ymax : int, optional
        Ymax used by the calculation.

    Returns
    -------
    object
        Result produced when the function is used to return err three-dimensional power spectrum.
    """
    t = np.clip((x - xmin) / (xmax - xmin), 0.0, None)
    return ymax - (ymax - ymin) * t**alpha
