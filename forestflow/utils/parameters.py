"""
Shared utility helpers.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from numpy.typing import ArrayLike
from forestflow.conventions import ARINYO_PARAMETER_NAMES

def params_numpy2dict(params: ArrayLike) -> dict[str, Any]:
    """
    Map ordered Arinyo parameters to their canonical names.

    Parameters
    ----------
    params : array_like
        One-dimensional values in ``ARINYO_PARAMETER_NAMES`` order.

    Returns
    -------
    dict
        Parameter mapping preserving individual array/scalar values.
    """
    param_names = ARINYO_PARAMETER_NAMES
    dict_param = {}
    for ii in range(params.shape[0]):
        dict_param[param_names[ii]] = params[ii]
    return dict_param

def params_numpy2dict_minimizer(params: ArrayLike) -> dict[str, Any]:
    """
    Map minimizer coordinates to physical Arinyo parameters.

    Parameters
    ----------
    params : array_like
        Values in canonical order; ``q1`` and ``q2`` are interpreted as their
        stored sum/difference coordinates when both are present.

    Returns
    -------
    dict
        Physical Arinyo parameter mapping.
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
    Map redshift-fit Arinyo output to its symmetric q1/q2 representation.

    Parameters
    ----------
    params : mapping
        Redshift-dependent minimizer parameter mapping.

    Returns
    -------
    dict
        Parameter mapping with ``q1`` and ``q2`` set to half the stored q1.
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
    Convert beta/kvav parameterization to Arinyo model parameters.

    Parameters
    ----------
    dict_arinyo_params : mapping
        Arinyo mapping potentially containing ``beta`` and ``kvav``.
    fcosmo : float or array_like
        Dimensionless cosmological growth rate used for ``bias_eta``.

    Returns
    -------
    dict
        Mapping with ``bias_eta = bias * beta / fcosmo`` and
        ``kv = kvav**(1 / av)`` where applicable.
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
