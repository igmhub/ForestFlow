"""Validated containers for Arinyo fitting data."""
from dataclasses import dataclass
from typing import Any
import numpy as np
from forestflow.model.arinyo import ArinyoModel

@dataclass(slots=True)
class FitData:

    # cosmology
    """
    Store measured power spectra and uncertainties.
    """
    z: float
    linear: Any

    # model
    power_model: ArinyoModel

    # params
    ini_params: dict

    # 1D
    k1d: np.ndarray
    p1d: np.ndarray
    std_p1d: np.ndarray

    # 3D
    k3d: np.ndarray
    mu3d: np.ndarray
    p3d: np.ndarray
    std_p3d: np.ndarray
