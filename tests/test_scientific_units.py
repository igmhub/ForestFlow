import numpy as np

from forestflow.statistics.p1d import P1DIntegrator, P1D_Mpc_bin_averaged, p1d_from_p3d
from forestflow.statistics.mock_power import make_arinyo_mock_power
from forestflow.statistics.p1d import P1D_Mpc
from forestflow.emulator.training import Transf_data


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
        standardized = transform.transf_stand(parameters, direct=True, type_stand=kind)
        restored = transform.transf_stand(standardized, direct=False, type_stand=kind)
        _assert_parameter_dicts_equal(restored, parameters)


def _gaussian_P3D_Mpc_kpar_kperp(
    linear, z, k_par_iMpc, k_perp_iMpc, parameters, new_cosmo_params=None
):
    del linear, z, k_par_iMpc, new_cosmo_params
    return np.exp(-parameters["alpha_Mpc2"] * k_perp_iMpc**2)


def test_p3d_to_p1d_integration_matches_analytic_result():
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

    integrator = P1DIntegrator(
        k_perp_min_iMpc=k_perp_min_iMpc,
        k_perp_max_iMpc=k_perp_max_iMpc,
        n_k_perp=1001,
    )
    prediction = P1D_Mpc(
        None,
        3.0,
        k_par_iMpc,
        _gaussian_P3D_Mpc_kpar_kperp,
        {"alpha_Mpc2": alpha_Mpc2},
        integrator,
    )

    np.testing.assert_allclose(prediction, expected_P1D_Mpc, rtol=1e-8)


def test_gauss_legendre_integrator_matches_analytic_result_and_caches_geometry():
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

    def gaussian_p3d(linear, z, k_par_iMpc, k_perp_iMpc, parameters):
        del linear, z, k_par_iMpc, parameters
        return np.exp(-alpha_Mpc2 * k_perp_iMpc**2)

    integrator = P1DIntegrator(
        k_perp_min_iMpc=k_perp_min_iMpc,
        k_perp_max_iMpc=k_perp_max_iMpc,
        n_k_perp=32,
        method="gauss_legendre",
    )
    prediction = integrator(None, 3.0, k_par_iMpc, gaussian_p3d, {})
    np.testing.assert_allclose(prediction, expected_P1D_Mpc[None, :], rtol=2e-7)
    assert len(integrator._geometry_cache) == 1
    repeat = integrator(None, 3.0, k_par_iMpc, gaussian_p3d, {})
    np.testing.assert_allclose(repeat, prediction, rtol=0)
    assert len(integrator._geometry_cache) == 1


def test_p1d_bin_average_uses_explicit_or_inferred_logarithmic_edges():
    def quadratic_p1d(linear, z, k_par_iMpc, parameters):
        del linear, z, parameters
        return np.asarray(k_par_iMpc) ** 2

    centres = np.array([1.0, 2.0, 4.0])
    inferred = P1D_Mpc_bin_averaged(
        None, 3.0, centres, quadratic_p1d, {}, fine_factor=8
    )
    log_centres = np.log(centres)
    log_edges = np.r_[
        log_centres[0] - 0.5 * np.diff(log_centres)[0],
        0.5 * (log_centres[:-1] + log_centres[1:]),
        log_centres[-1] + 0.5 * np.diff(log_centres)[-1],
    ]
    explicit = P1D_Mpc_bin_averaged(
        None,
        3.0,
        centres,
        quadratic_p1d,
        {},
        k_par_edges=np.exp(log_edges),
        fine_factor=8,
    )
    np.testing.assert_allclose(inferred, explicit)


def test_p1d_integrator_accepts_zero_parallel_wavenumber():
    def gaussian_p3d(linear, z, k_par_iMpc, k_perp_iMpc, parameters):
        del linear, z, k_par_iMpc, parameters
        return np.exp(-(k_perp_iMpc**2))

    result = P1DIntegrator(n_k_perp=17)(
        None, 3.0, np.array([0.0, 0.2]), gaussian_p3d, {}
    )
    assert result.shape == (1, 2)
    assert np.all(np.isfinite(result))


def test_diagnostic_p1d_projection_reuses_supplied_integrator():
    def constant_p3d(linear, z, k_par_iMpc, k_perp_iMpc, parameters):
        del linear, z, k_par_iMpc, parameters
        return np.ones_like(k_perp_iMpc)

    integrator = P1DIntegrator(
        k_perp_min_iMpc=0.01,
        k_perp_max_iMpc=2.0,
        n_k_perp=31,
    )
    k_par_iMpc = np.array([0.0, 0.2])
    diagnostic = p1d_from_p3d(
        None, k_par_iMpc, constant_p3d, 3.0, integrator=integrator
    )
    direct = integrator(None, 3.0, k_par_iMpc, constant_p3d)
    np.testing.assert_allclose(diagnostic["P1D_Mpc"], direct[0])
    assert diagnostic["P3D_Mpc"].shape[-1] == integrator.n_k_perp


def test_mock_power_uses_canonical_wavenumber_keys():
    class Linear:
        def get_linear_theory(self, z):
            return z

        def get_linP_Mpc(self, linear, z, k_iMpc):
            return np.ones_like(k_iMpc)

    class Model:
        linear = Linear()

        @staticmethod
        def P3D_Mpc_kpar_kperp(linear, z, k_par_iMpc, k_perp_iMpc, parameters):
            return np.ones_like(k_par_iMpc)

        @staticmethod
        def P1D_Mpc(linear, z, k_par_iMpc, parameters):
            return np.ones_like(k_par_iMpc)

    result = make_arinyo_mock_power(
        {"z": 3.0, "Arinyo": {"q1": 1.0, "q2": 0.0, "kp": 1.0}},
        Model(),
        n_1d=3,
        n_3d=3,
    )
    assert {"model_k_par_iMpc", "model_k_perp_iMpc", "model_k_1d_iMpc"} <= result.keys()
    assert "model_kpar_Mpc" not in result


def test_p3d_accepts_zero_wavenumber_and_preserves_padded_nan_cells():
    from forestflow.model.arinyo import ArinyoModel
    from forestflow.model.linear import LinearTheoryGrid

    model = ArinyoModel()
    linear = LinearTheoryGrid(
        z=np.array([3.0]),
        logk_iMpc=np.log(np.array([1.0e-3, 1.0, 10.0])),
        loglinP_Mpc=np.log(np.array([[1.0, 1.0, 1.0]])),
        fz=np.array([1.0]),
    )
    parameters = {
        "bias": -0.2,
        "bias_eta": -0.2,
        "q1": 0.0,
        "q2": 0.0,
        "av": 1.0,
        "kvav": 1.0,
        "bv": 1.0,
        "kp": 1.0,
    }
    result = model.P3D_Mpc_k_mu(
        linear,
        3.0,
        np.array([0.0, np.nan, 1.0]),
        np.array([0.0, np.nan, 0.5]),
        parameters,
    )
    assert np.isfinite(result[0]) and np.isnan(result[1]) and np.isfinite(result[2])


def test_mode_weighted_p3d_rebinning_uses_discrete_mode_counts():
    from forestflow.statistics.rebin_p3d import rebin_P3D_Mpc_mode_weighted

    k_iMpc = np.array([[1.0, 1.0, 1.0, 1.0]])
    mu = np.array([[0.1, 0.3, 0.6, 0.9]])
    P3D_Mpc = np.array([[1.0, 3.0, 10.0, 14.0]])
    k_mu_modes = {
        "0_0_k": np.empty(1),
        "0_1_k": np.empty(3),
        "0_2_k": np.empty(2),
        "0_3_k": np.empty(2),
    }
    _, _, rebinned, mu_edges, mode_counts = rebin_P3D_Mpc_mode_weighted(
        k_iMpc, mu, P3D_Mpc, k_mu_modes, n_mu_bins=2, return_mode_counts=True
    )
    np.testing.assert_allclose(mu_edges, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(rebinned, [[2.5, 12.0]])
    np.testing.assert_allclose(mode_counts, [[4.0, 4.0]])


def test_generic_p3d_bin_averages_accept_explicit_or_inferred_edges():
    from forestflow.statistics.p3d import (
        P3D_Mpc_k_mu_bin_averaged,
        P3D_Mpc_kpar_kperp_bin_averaged,
    )

    def P3D_model(linear, z, k_iMpc, mu, parameters):
        del linear, z, parameters
        return k_iMpc**2 + mu

    k_iMpc = np.array([1.0, 2.0, 4.0])
    mu = np.array([0.25, 0.75])
    inferred = P3D_Mpc_k_mu_bin_averaged(None, 3.0, k_iMpc, mu, P3D_model, {})
    explicit = P3D_Mpc_k_mu_bin_averaged(
        None, 3.0, k_iMpc, mu, P3D_model, {},
        k_iMpc_edges=np.array([2**-0.5, 2**0.5, 2**1.5, 2**2.5]),
        mu_edges=np.array([0.0, 0.5, 1.0]),
    )
    np.testing.assert_allclose(inferred, explicit)
    edge_only = P3D_Mpc_k_mu_bin_averaged(
        None, 3.0, None, None, P3D_model, {},
        k_iMpc_edges=np.array([2**-0.5, 2**0.5, 2**1.5, 2**2.5]),
        mu_edges=np.array([0.0, 0.5, 1.0]),
    )
    np.testing.assert_allclose(edge_only, explicit)
    ignored_centres = P3D_Mpc_k_mu_bin_averaged(
        None, 3.0, np.array([-99.0]), np.array([99.0]), P3D_model, {},
        k_iMpc_edges=np.array([2**-0.5, 2**0.5, 2**1.5, 2**2.5]),
        mu_edges=np.array([0.0, 0.5, 1.0]),
    )
    np.testing.assert_allclose(ignored_centres, explicit)

    def p3d_kpar_kperp(linear, z, k_par_iMpc, k_perp_iMpc, parameters):
        del linear, z, parameters
        return k_par_iMpc + k_perp_iMpc**2

    cartesian = P3D_Mpc_kpar_kperp_bin_averaged(
        None, 3.0, np.array([-1.0, 1.0]), np.array([1.0, 2.0]),
        p3d_kpar_kperp, {},
        k_par_edges=np.array([-2.0, 0.0, 2.0]),
        k_perp_edges=np.array([2**-0.5, 2**0.5, 2**1.5]),
    )
    assert cartesian.shape == (2, 2)
    assert np.all(np.isfinite(cartesian))


def test_p3d_exact_mode_average_preserves_padded_grid_cells():
    from forestflow.statistics.p3d import P3D_Mpc_k_mu_mode_averaged

    def P3D_model(linear, z, k_iMpc, mu, parameters):
        del linear, z, parameters
        return k_iMpc + 2.0 * mu

    k_iMpc = np.array([[1.0, np.nan], [2.0, 2.0]])
    mu = np.array([[0.1, np.nan], [0.2, 0.8]])
    modes = {
        "0_0_k": np.array([1.0, 3.0]),
        "0_0_mu": np.array([0.0, 0.5]),
        "1_0_k": np.array([2.0]),
        "1_0_mu": np.array([0.2]),
    }
    result = P3D_Mpc_k_mu_mode_averaged(
        None, 3.0, P3D_model, {}, k_mu_modes=modes, k_iMpc=k_iMpc, mu=mu
    )
    np.testing.assert_allclose(result[0, 0], 2.5)
    np.testing.assert_allclose(result[1, 0], 2.4)
    assert np.isnan(result[0, 1]) and np.isnan(result[1, 1])
    edge_only = P3D_Mpc_k_mu_mode_averaged(
        None, 3.0, P3D_model, {}, k_mu_modes=modes,
        k_iMpc_edges=np.array([0.5, 1.5, 2.5]),
        mu_edges=np.array([0.0, 0.5, 1.0]),
    )
    np.testing.assert_allclose(edge_only, result, equal_nan=True)


def test_mpg_p3d_bin_edges_start_at_the_fundamental_mode():
    from forestflow.statistics.rebin_p3d import get_P3D_k_mu_bin_edges

    k_edges, mu_edges = get_P3D_k_mu_bin_edges(4.0)
    np.testing.assert_allclose(k_edges[0], 2.0 * np.pi / 67.5)
    assert len(k_edges) - 1 == 14
    np.testing.assert_allclose(mu_edges, np.linspace(0.0, 1.0, 17))


def test_continuous_p3d_average_uses_three_dimensional_phase_space_weights():
    from forestflow.statistics.p3d import P3D_Mpc_k_mu_bin_averaged

    def P3D_model(linear, z, k_iMpc, mu, parameters):
        del linear, z, mu, parameters
        return k_iMpc

    result = P3D_Mpc_k_mu_bin_averaged(
        None,
        3.0,
        P3D_model=P3D_model,
        k_iMpc_edges=np.array([1.0, 2.0]),
        mu_edges=np.array([0.0, 1.0]),
        fine_factor=2,
    )
    fine_k = 2.0 ** np.array([0.25, 0.75])
    expected = np.sum(fine_k**4) / np.sum(fine_k**3)
    np.testing.assert_allclose(result, [[expected]])
    assert not np.isclose(result[0, 0], np.mean(fine_k))


def test_hybrid_p3d_uses_exact_sparse_cells_and_continuous_dense_cells():
    from forestflow.statistics.p3d import (
        P3D_Mpc_k_mu_bin_averaged,
        P3D_Mpc_k_mu_hybrid_averaged,
        P3D_Mpc_k_mu_mode_averaged,
    )

    def P3D_model(linear, z, k_iMpc, mu, parameters):
        del linear, z, parameters
        return k_iMpc + mu

    k_edges = np.array([1.0, 2.0, 4.0])
    mu_edges = np.array([0.0, 1.0])
    modes = {
        "0_0_k": np.array([1.1, 1.9]),
        "0_0_mu": np.array([0.1, 0.9]),
        "1_0_k": np.array([2.1, 2.5, 3.0]),
        "1_0_mu": np.array([0.2, 0.4, 0.8]),
    }
    hybrid = P3D_Mpc_k_mu_hybrid_averaged(
        None, 3.0, P3D_model, k_mu_modes=modes,
        k_iMpc_edges=k_edges, mu_edges=mu_edges, max_discrete_modes=2,
        fine_factor=4,
    )
    exact = P3D_Mpc_k_mu_mode_averaged(
        None, 3.0, P3D_model, k_mu_modes=modes,
        k_iMpc_edges=k_edges, mu_edges=mu_edges,
    )
    continuous = P3D_Mpc_k_mu_bin_averaged(
        None, 3.0, P3D_model=P3D_model,
        k_iMpc_edges=k_edges, mu_edges=mu_edges, fine_factor=4,
    )
    np.testing.assert_allclose(hybrid[0, 0], exact[0, 0])
    np.testing.assert_allclose(hybrid[1, 0], continuous[1, 0])


def test_p3d_mode_builder_uses_half_open_mu_cells():
    from forestflow.statistics.rebin_p3d import get_P3D_k_mu_modes

    n_mu_bins = 4
    modes = get_P3D_k_mu_modes(1.0, n_mu_bins=n_mu_bins)
    mu_edges = np.linspace(0.0, 1.0, n_mu_bins + 1)
    for key, values in modes.items():
        if not key.endswith("_mu"):
            continue
        mu_index = int(key.split("_")[1])
        assert np.all(values >= mu_edges[mu_index])
        if mu_index == n_mu_bins - 1:
            assert np.all(values <= mu_edges[mu_index + 1])
        else:
            assert np.all(values < mu_edges[mu_index + 1])
