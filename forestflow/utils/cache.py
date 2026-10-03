"""
Shared utility helpers.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any
import numpy as np
import torch
import functools

def memorize(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Memoize hashable positional and keyword arguments without eviction.

    Parameters
    ----------
    func : callable
        Function whose arguments and keyword values are hashable.

    Returns
    -------
    callable
        Wrapped function returning cached object identities for repeated calls.
    """
    cache = {}

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """
        Call the wrapped function and reuse an exact argument cache entry.
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
    """
    Memoize positional NumPy-array calls with bounded insertion history.

    Parameters
    ----------
    func : callable
        Function to wrap.
    max_history : int, default: 2
        Maximum number of entries retained; ``None`` is not supported by the
        current eviction comparison.

    Returns
    -------
    callable
        Wrapped function keyed by positional array shapes and values. Keyword
        arguments are forwarded but are not part of the cache key.
    """
    cache = {}

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """
        Call the wrapped function and reuse a positional-array cache entry.
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
    """
    Memoize PyTorch tensor calls using shape and host-value snapshots.

    Parameters
    ----------
    func : callable
        Function to wrap.

    Returns
    -------
    callable
        Wrapped function with unbounded exact-value tensor cache.
    """
    cache = {}

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """
        Call the wrapped function and reuse an exact tensor cache entry.
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
