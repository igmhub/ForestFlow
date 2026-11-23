from .arrays import broadcast_leading_dimensions
"""Small, domain-neutral helpers."""
from .chains import init_chains, load_Arinyo_chains, purge_chains
from .parameters import (params_numpy2dict, params_numpy2dict_minimizer, params_numpy2dict_minimizerz, transform_arinyo_params)
from .statistics import get_covariance, sigma68, sort_dict

__all__ = ["broadcast_leading_dimensions", "get_covariance", "init_chains", "load_Arinyo_chains", "memorize", "memoize_numpy_arrays", "memoize_pytorch", "params_numpy2dict", "params_numpy2dict_minimizer", "params_numpy2dict_minimizerz", "print_memory_usage", "purge_chains", "sigma68", "sort_dict", "transform_arinyo_params"]

# cache.py and system.py import torch; load them only when used.
_LAZY = {
    "memorize": ".cache",
    "memoize_numpy_arrays": ".cache",
    "memoize_pytorch": ".cache",
    "print_memory_usage": ".system",
}


def __getattr__(name):
    """Import torch-dependent helpers on first access."""
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
