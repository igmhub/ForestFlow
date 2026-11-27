"""
Validated containers for Arinyo fitting data.
"""
from dataclasses import dataclass
from typing import Any
import numpy as np
from forestflow.model.arinyo import ArinyoModel

@dataclass(slots=True)
class FitData:
    """
    Store one Arinyo fitting data set and its model dependencies.

    Parameters
    ----------
    z : float
        Measurement redshift.
    linear : object
        Linear-theory data evaluated at ``z``.
    power_model : ArinyoModel
        Arinyo model coupled to the measurement cosmology.
    ini_params : dict
        Initial Arinyo parameter mapping.
    k1d, p1d, std_p1d : ndarray
        P1D wavenumbers, values, and fractional uncertainties.
    k3d, mu3d, p3d, std_p3d : ndarray
        P3D wavenumbers, angle cosines, values, and fractional uncertainties.
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
