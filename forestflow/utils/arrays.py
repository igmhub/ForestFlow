"""
Array-shape helpers shared across ForestFlow models.
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
