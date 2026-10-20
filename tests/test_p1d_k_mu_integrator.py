"""Checks of the split Gauss--Legendre ln k projection."""
import numpy as np
import pytest

from forestflow.statistics.p1d import P1D_Mpc, P1DIntegrator, P1DIntegratorKMu


def _p3d_k_mu(linear, z, k_iMpc, mu, parameters):
    # Smooth toy model with a BAO-like wiggle, to exercise the split.
    return (1 + mu**2) * np.exp(-((k_iMpc / 5.0) ** 2)) * k_iMpc**-1.5 * (
        1 + 0.05 * np.sin(k_iMpc / 0.02) * np.exp(-((k_iMpc / 0.3) ** 2))
    )


def _p3d_kpar_kperp(linear, z, k_par_iMpc, k_perp_iMpc, parameters):
    k = np.hypot(k_par_iMpc, k_perp_iMpc)
    return _p3d_k_mu(linear, z, k, k_par_iMpc / k, parameters)


def test_k_mu_projection_matches_k_perp_projection():
    k_par = np.geomspace(0.05, 6.0, 25)
    # Same transverse domain for both rules (the defaults differ).
    reference = P1D_Mpc(
        None, 3.0, k_par, _p3d_kpar_kperp,
        integrator=P1DIntegrator(k_perp_min_iMpc=1e-3, n_k_perp=4001),
    )
    result = P1D_Mpc(
        None, 3.0, k_par, _p3d_k_mu,
        integrator=P1DIntegratorKMu(k_perp_min_iMpc=1e-3, n_low=48, n_high=48),
    )
    np.testing.assert_allclose(result, reference, rtol=1e-6)


def test_geometry_splits_only_below_k_split_and_keeps_mu_in_range():
    integrator = P1DIntegratorKMu(n_low=5, n_high=7, k_split_iMpc=0.4)
    k_par = np.array([0.0, 0.1, 0.39, 0.4, 2.0])
    k, mu, weights = integrator.geometry(k_par)
    assert k.shape == mu.shape == weights.shape == (5, 12)
    assert np.all((mu >= 0) & (mu <= 1))
    # Weights integrate d ln k over each row's full domain.
    lo = 0.5 * np.log(k_par**2 + integrator.k_perp_min_iMpc**2)
    hi = 0.5 * np.log(k_par**2 + 100.0**2)
    np.testing.assert_allclose(weights.sum(axis=-1), hi - lo, rtol=1e-12)
    # Split rows place exactly n_low nodes below the split.
    split = np.exp(lo) < 0.4
    np.testing.assert_array_equal((k < 0.4).sum(axis=-1)[split], 5)
    assert np.all(k[~split] >= k_par[~split, None] * (1 - 1e-12))


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        P1DIntegratorKMu(k_split_iMpc=200.0)
    with pytest.raises(ValueError):
        P1DIntegratorKMu(n_low=0)


def test_no_split_uses_one_rule_on_every_row():
    integrator = P1DIntegratorKMu(n_low=4, n_high=4, k_split_iMpc=None)
    k_par = np.array([0.01, 0.1, 2.0])
    k, _, weights = integrator.geometry(k_par)
    lo = 0.5 * np.log(k_par**2 + integrator.k_perp_min_iMpc**2)
    hi = 0.5 * np.log(k_par**2 + 100.0**2)
    nodes, _ = np.polynomial.legendre.leggauss(8)
    np.testing.assert_allclose(
        np.log(k), lo[:, None] + (hi - lo)[:, None] * 0.5 * (nodes + 1), rtol=1e-12
    )
    np.testing.assert_allclose(weights.sum(axis=-1), hi - lo, rtol=1e-12)
