"""
Explore power spectra, derivatives, and Fisher information.
"""

from collections.abc import Mapping
from typing import Any
from numpy.typing import ArrayLike

import numpy as np
from forestflow.model_p3d_arinyo import compute_Gaussian_cov


def get_sim_power(sim: Any, kmax_1d_Mpc: int | None=4, kmax_3d_Mpc: int | None=5) -> Any:

    """
    Return simulation power.

    Parameters
    ----------
    sim : object
        Sim used by the calculation.
    kmax_1d_Mpc : int, optional
        Kmax 1d mpc used by the calculation.
    kmax_3d_Mpc : int, optional
        Kmax 3d mpc used by the calculation.

    Returns
    -------
    object
        Result produced when the function is used to return simulation power.
    """
    data = {}

    mask_1d = (sim["k_Mpc"] <= kmax_1d_Mpc) & (sim["k_Mpc"] > 0)
    k1d_Mpc = sim["k_Mpc"][mask_1d]
    p1d_Mpc = sim["p1d_Mpc"][mask_1d]
    data["sim_k1d_Mpc"] = k1d_Mpc
    data["sim_p1d_Mpc"] = p1d_Mpc

    mask_3d = (sim["k3d_Mpc"] <= kmax_3d_Mpc) & np.isfinite(sim["p3d_Mpc"])
    k3d_Mpc = sim["k3d_Mpc"][mask_3d]
    p3d_Mpc = sim["p3d_Mpc"][mask_3d]
    mu3d = sim["mu3d"][mask_3d]
    data["sim_k3d_Mpc"] = k3d_Mpc
    data["sim_p3d_Mpc"] = p3d_Mpc
    data["sim_mu3d"] = mu3d

    return data


# compute power for Arinyo model
def compute_arinyo_power(
    pars_model: Any,
    model_Arinyo: Any,
    n3d: int | None=100,
    n1d: int | None=100,
    kmin_1d_Mpc: float | None=0.1,
    kmax_1d_Mpc: float | None=5.0,
    kmin_3d_Mpc: float | None=0.1,
    kmax_3d_Mpc: float | None=5.0,
    noise: Mapping[str, Any] | None={"n_noise": 0, "keep_all_noise": False, "Lbox_Mpc": 100},
) -> Any:

    # get kaiser params
    """
    Compute Arinyo power.

    Parameters
    ----------
    pars_model : object
        Pars model used by the calculation.
    model_Arinyo : object
        Model arinyo used by the calculation.
    n3d : int, optional
        N3d used by the calculation.
    n1d : int, optional
        N1d used by the calculation.
    kmin_1d_Mpc : float, optional
        Kmin 1d mpc used by the calculation.
    kmax_1d_Mpc : float, optional
        Kmax 1d mpc used by the calculation.
    kmin_3d_Mpc : float, optional
        Kmin 3d mpc used by the calculation.
    kmax_3d_Mpc : float, optional
        Kmax 3d mpc used by the calculation.
    noise : dict, optional
        Noise used by the calculation.

    Returns
    -------
    object
        Result produced when the function is used to compute arinyo power.
    """
    pars_kai = {}
    for par in pars_model["Arinyo"]:
        if par in ["q1", "q2"]:
            pars_kai[par] = 0
        elif par == "kp":
            pars_kai[par] = 1e6
        else:
            pars_kai[par] = pars_model["Arinyo"][par]

    data = {}

    # get 3D
    kpar = np.linspace(kmin_3d_Mpc, kmax_3d_Mpc, n3d)
    kper = np.linspace(kmin_3d_Mpc, kmax_3d_Mpc, n3d)
    kpar2d, kperp2d = np.meshgrid(kpar, kper, indexing="ij")
    data["model_kpar_Mpc"] = kpar2d
    data["model_kper_Mpc"] = kperp2d

    data["ari_P3D_Mpc"] = model_Arinyo.P3D_Mpc_kpar_kperp(
        pars_model["z"], kpar2d, kperp2d, pars_model["Arinyo"]
    )
    data["kai_P3D_Mpc"] = model_Arinyo.P3D_Mpc_kpar_kperp(
        pars_model["z"], kpar2d, kperp2d, pars_kai
    )
    k3d = np.sqrt(kpar2d**2 + kperp2d**2)
    data["Plin_Mpc"] = model_Arinyo.linP_Mpc(pars_model["z"], k3d)

    # get 1D
    k1d_Mpc = np.linspace(kmin_1d_Mpc, kmax_1d_Mpc, n1d)
    data["model_k1d_Mpc"] = k1d_Mpc
    data["ari_P1D_Mpc"] = model_Arinyo.P1D_Mpc(
        pars_model["z"], k1d_Mpc, pars_model["Arinyo"]
    )

    if noise["n_noise"] > 0:

        vol = noise["Lbox_Mpc"] ** 3
        data["ari_std_P3D_Mpc"] = compute_Gaussian_cov(
            kpar2d, kperp2d, data["ari_P3D_Mpc"].reshape(-1), vol
        ).reshape(n3d, n3d)

        ari_noise_P1D_Mpc = np.zeros((noise["n_noise"], k1d_Mpc.shape[0]))
        for ii in range(noise["n_noise"]):
            ari_noise_P1D_Mpc[ii] = model_Arinyo.P1D_Mpc_Gaussian_noise(
                pars_model["z"],
                k1d_Mpc,
                pars_model["Arinyo"],
                seed=ii,
                Lbox_Mpc=noise["Lbox_Mpc"],
            )

        if noise["keep_all_noise"]:
            data["ari_noise_P1D_Mpc"] = ari_noise_P1D_Mpc

        data["ari_std_P1D_Mpc"] = np.std(ari_noise_P1D_Mpc, axis=0)

    return data


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
            data_model["z"],
            data_model["kpar_Mpc"],
            data_model["kper_Mpc"],
            top_par,
        )

        p3d_der_bot = model_Arinyo.P3D_Mpc_kpar_kperp(
            data_model["z"],
            data_model["kpar_Mpc"],
            data_model["kper_Mpc"],
            bot_par,
        )

        data["P3D_der"][par] = (p3d_der_top - p3d_der_bot) / 2 / hh

        p1d_der_top = model_Arinyo.P1D_Mpc(
            data_model["z"],
            data_model["k1D_Mpc"],
            top_par,
        )

        p1d_der_bot = model_Arinyo.P1D_Mpc(
            data_model["z"],
            data_model["k1D_Mpc"],
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
    power = compute_arinyo_power(pars_model, model_Arinyo, noise=noise)
    pars_model["kpar_Mpc"] = power["model_kpar_Mpc"]
    pars_model["kper_Mpc"] = power["model_kper_Mpc"]
    pars_model["P3D_Mpc"] = power["ari_P3D_Mpc"]
    pars_model["std_P3D_Mpc"] = power["ari_std_P3D_Mpc"]

    pars_model["k1D_Mpc"] = power["model_k1d_Mpc"]
    pars_model["P1D_Mpc"] = power["ari_P1D_Mpc"]
    pars_model["std_P1D_Mpc"] = power["ari_std_P1D_Mpc"]

    der_data = compute_arinyo_derivatives(transf_data, pars_model, model_Arinyo)
    pars_model["P3D_der"] = der_data["P3D_der"]
    pars_model["P1D_der"] = der_data["P1D_der"]

    fisher = compute_fisher(pars_model, weight_3d=weight_3d, weight_1d=weight_1d)

    return fisher
