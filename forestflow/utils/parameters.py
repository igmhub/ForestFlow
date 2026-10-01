"""Shared utility helpers."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from numpy.typing import ArrayLike
from forestflow.conventions import ARINYO_PARAMETER_NAMES

def params_numpy2dict(params: ArrayLike) -> dict[str, Any]:
    """
    Convert a NumPy array of parameters to a dictionary.

    Args:
        params (numpy.ndarray): Array of parameters.

    Returns:
        dict: Dictionary containing the parameters with their corresponding names.
    """
    param_names = ARINYO_PARAMETER_NAMES
    dict_param = {}
    for ii in range(params.shape[0]):
        dict_param[param_names[ii]] = params[ii]
    return dict_param

def params_numpy2dict_minimizer(params: ArrayLike) -> dict[str, Any]:
    """
    Convert a NumPy array of parameters to a dictionary.

    Args:
        params (numpy.ndarray): Array of parameters.

    Returns:
        dict: Dictionary containing the parameters with their corresponding names.
    """
    param_names = ARINYO_PARAMETER_NAMES
    dict_param = {}
    for ii in range(params.shape[0]):
        dict_param[param_names[ii]] = params[ii]

    if "q2" in dict_param.keys():
        q1 = 0.5 * (dict_param["q1"] + dict_param["q2"])
        q2 = 0.5 * (dict_param["q1"] - dict_param["q2"])
        dict_param["q1"] = q1
        dict_param["q2"] = q2
    return dict_param

def params_numpy2dict_minimizerz(params: ArrayLike) -> dict[str, Any]:
    """
    Convert a NumPy array of parameters to a dictionary.

    Args:
        params (numpy.ndarray): Array of parameters.

    Returns:
        dict: Dictionary containing the parameters with their corresponding names.
    """
    dict_param = {}
    for key in params:
        if key == "q1":
            dict_param["q1"] = 0.5 * params[key]
        else:
            dict_param[key] = params[key]
    dict_param["q2"] = dict_param["q1"]

    return dict_param

def transform_arinyo_params(dict_arinyo_params: Mapping[str, Any], fcosmo: Any) -> Any:
    """
    Transform Arinyo parameters.

    Parameters
    ----------
    dict_arinyo_params : dict
        Arinyo parameter mapping.
    fcosmo : object
        Cosmological growth rate.

    Returns
    -------
    object
        Result produced when the function is used to transform arinyo parameters.
    """
    dict_arinyo_params_out = {}
    for key in dict_arinyo_params.keys():
        if key == "beta":
            dict_arinyo_params_out["bias_eta"] = (
                dict_arinyo_params["bias"] * dict_arinyo_params["beta"] / fcosmo
            )
        elif key == "kvav":
            dict_arinyo_params_out["kv"] = dict_arinyo_params["kvav"] ** (
                1 / dict_arinyo_params["av"]
            )
        else:
            dict_arinyo_params_out[key] = dict_arinyo_params[key]
    return dict_arinyo_params_out
