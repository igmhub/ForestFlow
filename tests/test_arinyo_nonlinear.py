"""Contract tests for the reusable bias-free Arinyo nonlinear factor."""
import numpy as np

from forestflow.model.arinyo import ArinyoModel


def test_nonlinear_correction_reconstructs_full_kernel():
    model = ArinyoModel()
    k = np.array([0.12, 0.41, 0.83])
    mu = np.array([0.0, 0.5, 1.0])
    lin = np.array([510.0, 120.0, 35.0])
    params = {"bias": -0.17, "bias_eta": -0.24, "q1": 0.41,
              "q2": 0.07, "kvav": 0.58, "av": 0.29,
              "bv": 1.55, "kp": 10.5}
    fz = 0.97
    expected = model._arinyo_kernel(lin, fz, k, mu, params)
    reconstructed = lin * (params["bias"] + params["bias_eta"] * fz * mu**2)**2
    reconstructed *= model.nonlinear_correction(lin, k, mu, params)
    np.testing.assert_allclose(reconstructed, expected)


def test_nonlinear_correction_broadcasts_and_uses_absolute_mu():
    lin = np.array([[100.0, 80.0, 60.0], [110.0, 90.0, 70.0]])
    k = np.array([0.2, 0.4, 0.8])
    pars = {"q1": np.array([[0.2], [0.4]]), "q2": 0.1,
            "kvav": 0.5, "av": 0.3, "bv": 1.2, "kp": 8.0}
    positive = ArinyoModel.nonlinear_correction(lin, k, .6, pars)
    negative = ArinyoModel.nonlinear_correction(lin, k, -.6, pars)
    assert positive.shape == (2, 3)
    np.testing.assert_allclose(positive, negative)
