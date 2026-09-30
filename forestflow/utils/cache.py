"""Shared utility helpers."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any
import numpy as np
import torch
import functools

def memorize(func: Callable[..., Any]) -> Callable[..., Any]:
    # Initialize a dictionary to store the previous input parameters and result
    """
    Memoize the requested values.

    Parameters
    ----------
    func : callable
        Function to wrap.

    Returns
    -------
    object
        Result produced when the function is used to memoize the requested values.
    """
    cache = {}

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Convert arguments and keyword arguments to a tuple of their values
        """
        Call the wrapped function, reusing a cached result when available.

        Parameters
        ----------
        args : object
            Args used by the calculation.
        kwargs : object
            Kwargs used by the calculation.

        Returns
        -------
        object
            Result produced when the function is used to call the wrapped function, reusing a cached result when available.
        """
        key = (args, tuple(kwargs.items()))

        # Check if the same input parameters have been seen before
        if key in cache:
            # If yes, return the cached result
            return cache[key]
        else:
            # If not, call the inner function and cache the result
            result = func(*args, **kwargs)
            cache[key] = result
            return result

    return wrapper

def memoize_numpy_arrays(func: Callable[..., Any], max_history: int | None=2) -> Callable[..., Any]:
    # Initialize a dictionary to store the previous results for each key
    """
    Memoize numpy arrays.

    Parameters
    ----------
    func : callable
        Function to wrap.
    max_history : int, optional
        Maximum number of results retained in the cache.

    Returns
    -------
    object
        Result produced when the function is used to memoize numpy arrays.
    """
    cache = {}

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Convert NumPy arrays to a tuple of their shapes and contents
        """
        Call the wrapped function, reusing a cached result when available.

        Parameters
        ----------
        args : object
            Args used by the calculation.
        kwargs : object
            Kwargs used by the calculation.

        Returns
        -------
        object
            Result produced when the function is used to call the wrapped function, reusing a cached result when available.
        """
        key = tuple(
            (a.shape, tuple(a.flat)) if isinstance(a, np.ndarray) else a for a in args
        )

        # Check if the key is in the cache
        if key in cache:
            # If yes, return the cached result
            return cache[key]
        else:
            # If not, call the inner function and cache the result
            result = func(*args, **kwargs)
            cache[key] = result
            # Trim the history to the specified maximum
            list_keys = list(cache.keys())
            if len(list_keys) > max_history:
                del cache[list_keys[0]]
            return result

    return wrapper

def memoize_pytorch(func: Callable[..., Any]) -> Callable[..., Any]:
    # Initialize a dictionary to store the previous input tensors and result
    """
    Memoize pytorch.

    Parameters
    ----------
    func : callable
        Function to wrap.

    Returns
    -------
    object
        Result produced when the function is used to memoize pytorch.
    """
    cache = {}

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Convert PyTorch tensors to tuples of their shapes and contents
        """
        Call the wrapped function, reusing a cached result when available.

        Parameters
        ----------
        args : object
            Args used by the calculation.
        kwargs : object
            Kwargs used by the calculation.

        Returns
        -------
        object
            Result produced when the function is used to call the wrapped function, reusing a cached result when available.
        """
        args_key = tuple(
            (a.shape, tuple(a.flatten().tolist())) if isinstance(a, torch.Tensor) else a
            for a in args
        )
        kwargs_key = tuple(
            (
                (key, value.shape, tuple(value.flatten().tolist()))
                if isinstance(value, torch.Tensor)
                else (key, value)
            )
            for key, value in kwargs.items()
        )

        # Combine args and kwargs keys into a single key
        key = (args_key, kwargs_key)

        # Check if the same input parameters have been seen before
        if key in cache:
            # If yes, return the cached result
            return cache[key]
        else:
            # If not, call the inner function and cache the result
            result = func(*args, **kwargs)
            cache[key] = result
            return result

    return wrapper
