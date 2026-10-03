"""
Neural-network construction helpers for the P3D emulator.
"""
from collections.abc import Mapping
from typing import Any
import hashlib
import numpy as np
import torch

def _training_data_fingerprint(training_data: Mapping[str, Mapping[str, Any]]) -> str:
    """
    Return a stable digest of exact input/output training-array bytes.

    Parameters
    ----------
    training_data : mapping
        ``input_par`` and ``output_par`` mappings in deterministic insertion
        order.

    Returns
    -------
    str
        SHA-256 digest including field names, dtypes, shapes, and bytes.
    """
    digest = hashlib.sha256()
    for group in ("input_par", "output_par"):
        digest.update(group.encode("utf-8"))
        for name, values in training_data[group].items():
            array = np.ascontiguousarray(np.asarray(values))
            digest.update(name.encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
    return digest.hexdigest()

def init_xavier(m: torch.nn.Module) -> None:
    """
    Initialize neural network weights using Xavier uniform initialization.

    This function applies Xavier initialization to Linear layers, setting weights
    from a uniform distribution with variance based on the number of input units.
    Biases are initialized to a small constant value.

    Parameters
    ----------
    m : torch.nn.Module
        The neural network module to initialize. Only Linear layers are modified.

    Returns
    -------
    None
        The module is modified in-place.

    Examples
    --------
    >>> model = torch.nn.Sequential(torch.nn.Linear(10, 5))
    >>> model.apply(init_xavier)
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
