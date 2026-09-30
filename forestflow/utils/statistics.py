"""Shared utility helpers."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from numpy.typing import ArrayLike, NDArray
import numpy as np

def sort_dict(dct: Sequence[Any], keys: Sequence[Any]) -> list[Any]:
    """
    Sort a list of dictionaries based on specified keys.

    Args:
        dct (list): List of dictionaries to be sorted.
        keys (list): List of keys to sort the dictionaries by.

    Returns:
        list: The sorted list of dictionaries.
    """
    for d in dct:
        sorted_d = {
            k: d[k] for k in keys
        }  # create a new dictionary with only the specified keys
        d.clear()  # remove all items from the original dictionary
        d.update(sorted_d)  # update the original dictionary with the sorted dictionary
    return dct

def get_covariance(x: Any, y: Any, return_corr: bool | None=False) -> NDArray[Any]:
    # Calculate the mean and standard deviation along each column
    """
    Return covariance.

    Parameters
    ----------
    x : object
        X used by the calculation.
    y : object
        Y used by the calculation.
    return_corr : bool, optional
        Whether to return the correlation matrix with the covariance.

    Returns
    -------
    object
        Result produced when the function is used to return covariance.
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
    Compute sigma68.

    Parameters
    ----------
    data : numpy.ndarray
        Input data.

    Returns
    -------
    object
        Result produced when the function is used to compute sigma68.
    """
    return 0.5 * (
        np.nanquantile(data, q=0.84, axis=0) - np.nanquantile(data, q=0.16, axis=0)
    )
