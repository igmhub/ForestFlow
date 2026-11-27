"""
Array helpers shared across ForestFlow models.
"""

import numpy as np


def broadcast_leading_dimensions(values, target):
    """
    Add trailing singleton axes so values broadcast over a target array.

    Parameters
    ----------
    values : array_like
        Scalar or leading-axis values.
    target : array_like
        Array whose dimensionality determines the number of added axes.

    Returns
    -------
    ndarray
        Reshaped ``values`` with enough trailing singleton dimensions to
        broadcast over ``target``.
    """
    values = np.asarray(values)
    return values.reshape(values.shape + (1,) * (np.ndim(target) - values.ndim))


def sigma68(data):
    """
    Return half the central 68-percent interval along the sample axis.

    Parameters
    ----------
    data : array_like
        Samples on axis zero; NaNs are ignored by the percentile calculation.

    Returns
    -------
    ndarray
        ``0.5 * (q84 - q16)`` with all non-sample axes retained.
    """
    return 0.5 * (
        np.nanquantile(data, q=0.84, axis=0) - np.nanquantile(data, q=0.16, axis=0)
    )
