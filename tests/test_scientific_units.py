import numpy as np

from forestflow.p1d import P1D_Mpc
from forestflow.set_training import Transf_data


def _assert_parameter_dicts_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-12)


def test_input_and_output_transformations_round_trip():
    input_parameters = {
        "Delta2_p": np.array([0.25, 0.35, 0.45]),
        "n_p": np.array([-2.4, -2.3, -2.2]),
        "mF": np.array([0.60, 0.70, 0.80]),
        "sigT_Mpc": np.array([0.10, 0.13, 0.16]),
        "gamma": np.array([1.2, 1.4, 1.6]),
        "kF_Mpc": np.array([8.0, 10.0, 12.0]),
    }
    output_parameters = {
        "bias": np.array([-0.30, -0.24, -0.18]),
        "bias_eta": np.array([-0.35, -0.29, -0.23]),
        "q1": np.array([0.25, 0.40, 0.55]),
        "kvav": np.array([0.45, 0.65, 0.85]),
        "av": np.array([0.35, 0.50, 0.65]),
        "bv": np.array([1.4, 1.8, 2.2]),
        "kp": np.array([12.0, 15.0, 18.0]),
        "q2": np.array([0.10, 0.25, 0.40]),
    }
    transform = Transf_data(
        dict_all_params={
            "input_par": input_parameters,
            "output_par": output_parameters,
        }
    )

    for kind, parameters in (
        ("input", input_parameters),
        ("output", output_parameters),
    ):
        standardized = transform.transf_stand(
            parameters, direct=True, type_stand=kind
        )
        restored = transform.transf_stand(
            standardized, direct=False, type_stand=kind
        )
        _assert_parameter_dicts_equal(restored, parameters)


def _gaussian_P3D_Mpc_k_mu(
    z, k_Mpc, mu, parameters, new_cosmo_params=None
):
    del z, new_cosmo_params
    k_perp_iMpc = k_Mpc * np.sqrt(1 - mu**2)
    return np.exp(-parameters["alpha_Mpc2"] * k_perp_iMpc**2)


_gaussian_P3D_Mpc_k_mu.coordinates = "k_mu"


def _gaussian_P3D_Mpc_kpar_kperp(
    z, k_par_iMpc, k_perp_iMpc, parameters, new_cosmo_params=None
):
    del z, k_par_iMpc, new_cosmo_params
    return np.exp(-parameters["alpha_Mpc2"] * k_perp_iMpc**2)


_gaussian_P3D_Mpc_kpar_kperp.coordinates = "kpar_kperp"


def test_p3d_to_p1d_integration_matches_analytic_result_in_both_coordinates():
    alpha_Mpc2 = 0.7
    k_perp_min_iMpc = 1.0e-3
    k_perp_max_iMpc = 8.0
    k_par_iMpc = np.array([0.2, 0.8])
    expected_P1D_Mpc = np.full(
        k_par_iMpc.shape,
        (
            np.exp(-alpha_Mpc2 * k_perp_min_iMpc**2)
            - np.exp(-alpha_Mpc2 * k_perp_max_iMpc**2)
        )
        / (4 * np.pi * alpha_Mpc2),
    )

    predictions = []
    for p3d_function in (
        _gaussian_P3D_Mpc_k_mu,
        _gaussian_P3D_Mpc_kpar_kperp,
    ):
        predictions.append(
            P1D_Mpc(
                3.0,
                k_par_iMpc,
                p3d_function,
                {"alpha_Mpc2": alpha_Mpc2},
                k_perp_min_iMpc=k_perp_min_iMpc,
                k_perp_max_iMpc=k_perp_max_iMpc,
                n_k_perp=1001,
            )
        )

    np.testing.assert_allclose(predictions[0], expected_P1D_Mpc, rtol=1e-8)
    np.testing.assert_allclose(predictions[1], expected_P1D_Mpc, rtol=1e-8)
    np.testing.assert_allclose(predictions[0], predictions[1], rtol=1e-12)
