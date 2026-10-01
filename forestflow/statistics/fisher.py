"""Numerical derivatives and Fisher-information helpers."""
from collections.abc import Mapping
from typing import Any
from numpy.typing import ArrayLike
import numpy as np
from forestflow.statistics.mock_power import make_arinyo_mock_power

def compute_arinyo_derivatives(trans_data: ArrayLike, data_model: ArrayLike, model_Arinyo: Any, hh: float | None=1e-6) -> Any:

    """
    Compute Arinyo derivatives.

    Parameters
    ----------
    trans_data : numpy.ndarray
        Trans data used by the calculation.
    data_model : numpy.ndarray
        Data model used by the calculation.
    model_Arinyo : object
        Model arinyo used by the calculation.
    hh : float, optional
        Hh used by the calculation.

    Returns
    -------
    object
        Result produced when the function is used to compute arinyo derivatives.
    """
    data = {}
    data["P3D_der"] = {}
    data["P1D_der"] = {}
    linear = data_model.get("linear")
    if linear is None:
        linear = model_Arinyo.linear.get_linear_theory(data_model["z"])

    tranf_Arinyo = trans_data.transf_stand(
        data_model["Arinyo"], type_stand="output", direct=True
    )

    for par in data_model["Arinyo"]:

        transf_top_par = {}
        transf_bot_par = {}

        # copy all other parameters
        for par1 in data_model["Arinyo"]:
            if par != par1:
                transf_top_par[par1] = tranf_Arinyo[par1]
                transf_bot_par[par1] = tranf_Arinyo[par1]
            else:
                transf_top_par[par1] = tranf_Arinyo[par1] + hh
                transf_bot_par[par1] = tranf_Arinyo[par1] - hh

        # go back to original space
        top_par = trans_data.transf_stand(
            transf_top_par, type_stand="output", direct=False
        )
        bot_par = trans_data.transf_stand(
            transf_bot_par, type_stand="output", direct=False
        )

        # print("")
        # print(par)
        # print(top_par)
        # print(bot_par)

        # 3D
        p3d_der_top = model_Arinyo.P3D_Mpc_kpar_kperp(
            linear,
            data_model["z"],
            data_model["k_par_iMpc"],
            data_model["k_perp_iMpc"],
            top_par,
        )

        p3d_der_bot = model_Arinyo.P3D_Mpc_kpar_kperp(
            linear,
            data_model["z"],
            data_model["k_par_iMpc"],
            data_model["k_perp_iMpc"],
            bot_par,
        )

        data["P3D_der"][par] = (p3d_der_top - p3d_der_bot) / 2 / hh

        p1d_der_top = model_Arinyo.P1D_Mpc(
            linear,
            data_model["z"],
            data_model["k_1d_iMpc"],
            top_par,
        )

        p1d_der_bot = model_Arinyo.P1D_Mpc(
            linear,
            data_model["z"],
            data_model["k_1d_iMpc"],
            bot_par,
        )

        data["P1D_der"][par] = (p1d_der_top - p1d_der_bot) / 2 / hh

    return data

def compute_fisher(data_model: ArrayLike, weight_3d: float | None=1.0, weight_1d: float | None=1.0) -> Any:

    """
    Compute Fisher matrix.

    Parameters
    ----------
    data_model : numpy.ndarray
        Data model used by the calculation.
    weight_3d : float, optional
        Weight 3d used by the calculation.
    weight_1d : float, optional
        Weight 1d used by the calculation.

    Returns
    -------
    object
        Result produced when the function is used to compute fisher matrix.
    """
    fisher = {}

    for par1 in data_model["Arinyo"]:

        if par1 == "beta":
            continue

        fisher[par1] = {}

        for par2 in data_model["Arinyo"]:
            if par2 == "beta":
                continue

            x = data_model["P3D_der"][par1]
            y = data_model["P3D_der"][par2]
            icov = 1 / data_model["std_P3D_Mpc"] ** 2
            res3d = np.sum(x * icov * y)

            x = data_model["P1D_der"][par1]
            y = data_model["P1D_der"][par2]
            icov = 1 / data_model["std_P1D_Mpc"] ** 2
            res1d = np.sum(x * icov * y)

            fisher[par1][par2] = weight_3d * res3d + weight_1d * res1d

    return fisher

def get_fisher(
    transf_data: ArrayLike,
    pars_model: Any,
    model_Arinyo: Any,
    weight_3d: float | None=1.0,
    weight_1d: float | None=1.0,
    noise: Mapping[str, Any] | None={"n_noise": 10000, "keep_all_noise": False, "Lbox_Mpc": 1000},
) -> Any:

    """
    Return Fisher matrix.

    Parameters
    ----------
    transf_data : numpy.ndarray
        Transf data used by the calculation.
    pars_model : object
        Pars model used by the calculation.
    model_Arinyo : object
        Model arinyo used by the calculation.
    weight_3d : float, optional
        Weight 3d used by the calculation.
    weight_1d : float, optional
        Weight 1d used by the calculation.
    noise : dict, optional
        Noise used by the calculation.

    Returns
    -------
    object
        Result produced when the function is used to return fisher matrix.
    """
    power = make_arinyo_mock_power(pars_model, model_Arinyo, noise=noise)
    pars_model["linear"] = power["linear"]
    pars_model["k_par_iMpc"] = power["model_k_par_iMpc"]
    pars_model["k_perp_iMpc"] = power["model_k_perp_iMpc"]
    pars_model["P3D_Mpc"] = power["ari_P3D_Mpc"]
    pars_model["std_P3D_Mpc"] = power["ari_std_P3D_Mpc"]

    pars_model["k_1d_iMpc"] = power["model_k_1d_iMpc"]
    pars_model["P1D_Mpc"] = power["ari_P1D_Mpc"]
    pars_model["std_P1D_Mpc"] = power["ari_std_P1D_Mpc"]

    der_data = compute_arinyo_derivatives(transf_data, pars_model, model_Arinyo)
    pars_model["P3D_der"] = der_data["P3D_der"]
    pars_model["P1D_der"] = der_data["P1D_der"]

    fisher = compute_fisher(pars_model, weight_3d=weight_3d, weight_1d=weight_1d)

    return fisher
