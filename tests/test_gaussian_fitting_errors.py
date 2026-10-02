"""Gaussian mode-count weights without changing the MP-Gadget defaults."""
from types import SimpleNamespace

import numpy as np
import pytest

from forestflow.model_fits import ArinyoFitter, gaussian_p3d_relative_error


def test_conjugate_count_conventions_and_inverse_square_root_scaling():
    full = np.array([[8., 32., 128.], [18., 72., 288.]])
    errors = gaussian_p3d_relative_error(full)
    np.testing.assert_allclose(errors[:, 1:], errors[:, :-1] / 2)
    np.testing.assert_allclose(
        errors, gaussian_p3d_relative_error(full / 2, count_convention="independent")
    )
    combined = gaussian_p3d_relative_error(full, fractional_floor=.05)
    np.testing.assert_allclose(combined**2 - errors**2, .05**2)


def test_gaussian_variance_matches_independent_complex_fourier_draws():
    # A conjugate pair contributes one exponential |delta_k|^2 variate.
    rng = np.random.default_rng(31)
    modes = rng.normal(size=(20000, 20, 2))
    power = np.mean(np.sum(modes**2, axis=-1) / 2, axis=1)
    expected = gaussian_p3d_relative_error(40)
    assert power.std(ddof=1) / power.mean() == pytest.approx(expected, rel=.025)


@pytest.mark.parametrize("counts", [[0], [-1], [np.nan], [np.inf]])
def test_invalid_counts_are_not_silently_regularized(counts):
    with pytest.raises(ValueError, match="mode_counts"):
        gaussian_p3d_relative_error(counts)


def test_invalid_floor_and_count_convention():
    for floor in (-.01, np.nan, np.inf):
        with pytest.raises(ValueError, match="fractional_floor"):
            gaussian_p3d_relative_error([10], fractional_floor=floor)
    with pytest.raises(ValueError, match="count_convention"):
        gaussian_p3d_relative_error([10], count_convention="unknown")


def test_fitter_objective_downweights_sparse_bins_and_preserves_mean_contract():
    fitter = object.__new__(ArinyoFitter)
    fitter.data = SimpleNamespace(
        p3d=np.ones((2, 3)), p1d=np.ones(4), std_p1d=np.ones(4),
        std_p3d=gaussian_p3d_relative_error(np.full((2, 3), 20.)),
    )
    fitter.predict = lambda params: (np.full((2, 3), 1.1), np.ones(4))
    initial = fitter.chi2(None)
    assert initial == pytest.approx(.1)
    fitter.data.std_p3d = gaussian_p3d_relative_error(np.full((2, 3), 80.))
    assert fitter.chi2(None) == pytest.approx(4 * initial)
