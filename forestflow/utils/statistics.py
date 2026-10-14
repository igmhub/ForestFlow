"""
Shared utility helpers.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from numpy.typing import ArrayLike, NDArray
import numpy as np

def sort_dict(dct: Sequence[Any], keys: Sequence[Any]) -> list[Any]:
    """
    Reorder every mapping in place to a specified key order.

    Parameters
    ----------
    dct : sequence of dict
        Mutable mappings to reorder.
    keys : sequence
        Required key order. Every key must exist in every mapping.

    Returns
    -------
    list
        The same list of mutated mappings.
    """
    for d in dct:
        sorted_d = {
            k: d[k] for k in keys
        }  # create a new dictionary with only the specified keys
        d.clear()  # remove all items from the original dictionary
        d.update(sorted_d)  # update the original dictionary with the sorted dictionary
    return dct

def get_covariance(x: Any, y: Any, return_corr: bool | None=False) -> NDArray[Any]:
    """
    Estimate a clipped covariance around a supplied reference vector.

    Parameters
    ----------
    x : ndarray, shape (n_samples, n_parameters)
        Sample matrix. Rows farther than three column standard deviations are
        removed before covariance estimation.
    y : array_like, shape (n_parameters,)
        Reference vector subtracted from retained samples.
    return_corr : bool, default: False
        Whether to return the correlation matrix with the covariance.

    Returns
    -------
    ndarray or tuple of ndarray
        Sample covariance; when requested, also the correlation coefficient
        matrix computed from that covariance.
    """
    mean_x = np.mean(x, axis=0)
    std_dev_x = np.std(x, axis=0)
    # Create a mask indicating elements within one standard deviation from the mean
    mask_within_sigma = np.abs(x - mean_x) <= 3 * std_dev_x
    # Apply the mask along each column to preserve the shape
    x = x[mask_within_sigma.all(axis=1)]

    cov = (
        1 / (len(x) - 1) * np.einsum("ij,jk ->ik", (x - y[None, :]).T, (x - y[None, :]))
    )
    corr = np.corrcoef(cov)
    if return_corr:
        return cov, corr
    else:
        return cov

def sigma68(data: ArrayLike) -> NDArray[Any]:
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
