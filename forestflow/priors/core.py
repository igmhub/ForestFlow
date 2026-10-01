"""
Define priors for Arinyo and intergalactic-medium parameters.
"""

from typing import Any

import os
import numpy as np
import forestflow


def get_arinyo_priors(z: int | float, tag: str | None="DESI_DR1_P1D", return_all: bool | None=False) -> Any:
    """
    Return Arinyo priors.

    Parameters
    ----------
    z : int or float
        Redshift.
    tag : str, optional
        Tag used by the calculation.
    return_all : bool, optional
        Return all used by the calculation.

    Returns
    -------
    object
        Result produced when the function is used to return arinyo priors.
    """
    if tag == "DESI_DR1_P1D":
        fname = "priors_arinyo_from_p1d.npy"
    else:
        raise ValueError("tag not recognized, only implemented for DESI_DR1_P1D")

    folder = os.path.join(os.path.dirname(forestflow.__path__[0]), "data", "priors")
    data_priors = np.load(os.path.join(folder, fname), allow_pickle=True).item()

    if (z > np.max(data_priors["zs"])) | (z < np.min(data_priors["zs"])):
        raise ValueError(
            "Priors only computed between",
            np.min(data_priors["zs"]),
            "and",
            np.max(data_priors["zs"]),
            "use z within this range",
        )

    out_priors = {}
    out_priors["mean"] = {}
    out_priors["std"] = {}
    out_priors["percen_5"] = {}
    out_priors["percen_95"] = {}
    for par in data_priors:
        if par == "zs":
            continue

        if par == "bias":
            use_dat = -np.abs(data_priors[par])
        else:
            use_dat = data_priors[par]

        mean = np.mean(use_dat, axis=0)
        std = np.std(use_dat, axis=0)
        val_min = np.percentile(use_dat, 5, axis=0)
        val_max = np.percentile(use_dat, 95, axis=0)

        out_priors["mean"][par] = np.interp(z, data_priors["zs"], mean)
        out_priors["std"][par] = np.interp(z, data_priors["zs"], std)
        out_priors["percen_5"][par] = np.interp(z, data_priors["zs"], val_min)
        out_priors["percen_95"][par] = np.interp(z, data_priors["zs"], val_max)

    if return_all:
        return out_priors, data_priors
    else:
        return out_priors


def get_IGM_priors(z: int | float, tag: str | None="DESI_DR1_P1D", return_all: bool | None=False) -> Any:
    """
    Return intergalactic-medium priors.

    Parameters
    ----------
    z : int or float
        Redshift.
    tag : str, optional
        Tag used by the calculation.
    return_all : bool, optional
        Return all used by the calculation.

    Returns
    -------
    object
        Result produced when the function is used to return intergalactic-medium priors.
    """
    if tag == "DESI_DR1_P1D":
        fname = "priors_cosmo_IGM_from_p1d.npy"
    else:
        raise ValueError("tag not recognized, only implemented for DESI_DR1_P1D")

    folder = os.path.join(os.path.dirname(forestflow.__path__[0]), "data", "priors")
    data_priors = np.load(os.path.join(folder, fname), allow_pickle=True).item()

    if (z > np.max(data_priors["zs"])) | (z < np.min(data_priors["zs"])):
        raise ValueError(
            "Priors only computed between",
            np.min(data_priors["zs"]),
            "and",
            np.max(data_priors["zs"]),
            "use z within this range",
        )

    out_priors = {}
    out_priors["mean"] = {}
    out_priors["std"] = {}
    out_priors["percen_5"] = {}
    out_priors["percen_95"] = {}
    for par in data_priors:
        if par == "zs":
            continue

        mean = np.mean(data_priors[par], axis=0)
        std = np.std(data_priors[par], axis=0)
        val_min = np.percentile(data_priors[par], 5, axis=0)
        val_max = np.percentile(data_priors[par], 95, axis=0)

        out_priors["mean"][par] = np.interp(z, data_priors["zs"], mean)
        out_priors["std"][par] = np.interp(z, data_priors["zs"], std)
        out_priors["percen_5"][par] = np.interp(z, data_priors["zs"], val_min)
        out_priors["percen_95"][par] = np.interp(z, data_priors["zs"], val_max)

    if return_all:
        return out_priors, data_priors
    else:
        return out_priors
