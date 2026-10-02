"""Synthetic Arinyo power measurements for forecasts and Fisher studies."""

import numpy as np

from forestflow.statistics.covariance import compute_Gaussian_cov


def make_arinyo_mock_power(
    model_parameters,
    arinyo_model,
    n_3d=100,
    n_1d=100,
    k_min_1d_iMpc=0.1,
    k_max_1d_iMpc=5.0,
    k_min_3d_iMpc=0.1,
    k_max_3d_iMpc=5.0,
    noise=None,
):
    """
    Build deterministic or finite-volume synthetic Arinyo P3D and P1D data.

    ``model_parameters`` must contain ``z`` and ``Arinyo``. All input and
    output wavenumbers use inverse Mpc, and powers use the corresponding Mpc
    convention in their key names.
    """
    for name, value in {
        "n_3d": n_3d,
        "n_1d": n_1d,
    }.items():
        if not isinstance(value, (int, np.integer)) or value < 2:
            raise ValueError(f"{name} must be an integer of at least two")
    for minimum, maximum, label in (
        (k_min_1d_iMpc, k_max_1d_iMpc, "1D"),
        (k_min_3d_iMpc, k_max_3d_iMpc, "3D"),
    ):
        if (
            not np.isfinite(minimum)
            or not np.isfinite(maximum)
            or not 0 < minimum < maximum
        ):
            raise ValueError(
                f"{label} k range must satisfy 0 < k_min_iMpc < k_max_iMpc"
            )
    noise = {"n_realizations": 0, "keep_realizations": False, "Lbox_Mpc": 100.0} | (
        noise or {}
    )
    if noise["n_realizations"] < 0 or noise["Lbox_Mpc"] <= 0:
        raise ValueError(
            "noise requires non-negative n_realizations and positive Lbox_Mpc"
        )

    arinyo_parameters = model_parameters["Arinyo"]
    kaiser_parameters = {
        name: (0.0 if name in {"q1", "q2"} else 1.0e6 if name == "kp" else value)
        for name, value in arinyo_parameters.items()
    }
    z = model_parameters["z"]
    linear = arinyo_model.linear.get_linear_theory(z)
    k_par_iMpc = np.linspace(k_min_3d_iMpc, k_max_3d_iMpc, n_3d)
    k_perp_iMpc = np.linspace(k_min_3d_iMpc, k_max_3d_iMpc, n_3d)
    k_par_grid_iMpc, k_perp_grid_iMpc = np.meshgrid(
        k_par_iMpc, k_perp_iMpc, indexing="ij"
    )
    k_iMpc = np.hypot(k_par_grid_iMpc, k_perp_grid_iMpc)
    k_1d_iMpc = np.linspace(k_min_1d_iMpc, k_max_1d_iMpc, n_1d)
    result = {
        "linear": linear,
        "model_k_par_iMpc": k_par_grid_iMpc,
        "model_k_perp_iMpc": k_perp_grid_iMpc,
        "model_k_1d_iMpc": k_1d_iMpc,
        "ari_P3D_Mpc": arinyo_model.P3D_Mpc_kpar_kperp(
            linear, z, k_par_grid_iMpc, k_perp_grid_iMpc, arinyo_parameters
        ),
        "kai_P3D_Mpc": arinyo_model.P3D_Mpc_kpar_kperp(
            linear, z, k_par_grid_iMpc, k_perp_grid_iMpc, kaiser_parameters
        ),
        "Plin_Mpc": arinyo_model.linear.get_linP_Mpc(linear, z, k_iMpc),
        "ari_P1D_Mpc": arinyo_model.P1D_Mpc(linear, z, k_1d_iMpc, arinyo_parameters),
    }
    if noise["n_realizations"]:
        result["ari_std_P3D_Mpc"] = compute_Gaussian_cov(
            k_par_grid_iMpc,
            k_perp_grid_iMpc,
            result["ari_P3D_Mpc"].ravel(),
            noise["Lbox_Mpc"] ** 3,
        ).reshape(n_3d, n_3d)
        realizations = np.asarray(
            [
                arinyo_model.P1D_Mpc_Gaussian_noise(
                    linear,
                    z,
                    k_1d_iMpc,
                    arinyo_parameters,
                    seed=index,
                    Lbox_Mpc=noise["Lbox_Mpc"],
                )
                for index in range(noise["n_realizations"])
            ]
        )
        result["ari_std_P1D_Mpc"] = np.std(realizations, axis=0)
        if noise["keep_realizations"]:
            result["ari_P1D_Mpc_realizations"] = realizations
    return result
