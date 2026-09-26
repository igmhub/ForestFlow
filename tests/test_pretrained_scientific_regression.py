from pathlib import Path

import numpy as np
import pytest
import torch

import forestflow
from forestflow.P3D_cINN import P3DEmulator
from forestflow.model_p3d_arinyo import ArinyoModel
from lace.cosmo import cosmology


pytestmark = pytest.mark.pretrained_model

CENTRAL_INPUT = {
    "Delta2_p": 0.3501252719027313,
    "n_p": -2.300047197725595,
    "mF": 0.6604100706377194,
    "sigT_Mpc": 0.12817463664956008,
    "gamma": 1.512170923999183,
    "kF_Mpc": 10.6348381789184,
}
CENTRAL_COSMOLOGY = {
    "H0": 67.0,
    "omch2": 0.11999994800000002,
    "ombh2": 0.022000140100000003,
    "mnu": 0.0,
    "omk": -5.551115123125783e-17,
    "As": 2.006055e-09,
    "ns": 0.967565,
    "nrun": 0.0,
    "w": -1.0,
}
EXPECTED_ARINYO = {
    "bias": -0.23107877107989805,
    "bias_eta": -0.28610976850151576,
    "q1": 0.39531900175414714,
    "kvav": 0.6524547332538043,
    "av": 0.5174394062801557,
    "bv": 1.7885556466461254,
    "kp": 15.507150014342459,
    "q2": 0.266083656536723,
}
EXPECTED_P3D_MPC = np.array(
    [16.380650386336058, 2.0328428868294637, 0.2670374337917371]
)
EXPECTED_P1D_MPC = np.array(
    [0.49765690342976293, 0.3073870279033289, 0.12941236753868576]
)
MODEL_DIRECTORY = (
    Path(forestflow.__file__).resolve().parent.parent
    / "data"
    / "emulator_models"
)
REQUIRED_MODEL_FILES = (
    "forest_mpg.pt",
    "forest_mpg_metadata.npy",
    "forest_mpg_transf.npy",
)


def _scalar_prediction(prediction):
    return {
        name: float(np.asarray(value).reshape(-1)[0])
        for name, value in prediction.items()
    }


@pytest.fixture(scope="module")
def central_prediction():
    missing = [
        filename
        for filename in REQUIRED_MODEL_FILES
        if not (MODEL_DIRECTORY / filename).is_file()
    ]
    if missing:
        pytest.skip(
            "pretrained-model regression assets are unavailable: "
            + ", ".join(missing)
        )
    torch.set_num_threads(1)
    emulator = P3DEmulator(key="forest_mpg", Nrealizations=3000)
    arinyo = _scalar_prediction(emulator.evaluate(CENTRAL_INPUT, seed=0))
    return emulator, arinyo


def test_central_simulation_arinyo_parameters(central_prediction):
    _, actual = central_prediction
    assert actual.keys() == EXPECTED_ARINYO.keys()
    for name, expected in EXPECTED_ARINYO.items():
        assert actual[name] == pytest.approx(expected, rel=2e-5, abs=1e-8)


def test_central_simulation_power_spectra_in_both_p3d_coordinates(
    central_prediction,
):
    _, arinyo = central_prediction
    model = ArinyoModel(
        cosmology.Cosmology(cosmo_params_dict=CENTRAL_COSMOLOGY)
    )
    linear = model.linear_theory(3.0)
    k_iMpc = np.array([0.2, 0.7, 2.0])
    mu = np.array([0.0, 0.5, 1.0])

    P3D_Mpc_k_mu = model.P3D_Mpc_k_mu(linear, 3.0, k_iMpc, mu, arinyo)
    k_par_iMpc = k_iMpc * mu
    k_perp_iMpc = k_iMpc * np.sqrt(1 - mu**2)
    P3D_Mpc_kpar_kperp = model.P3D_Mpc_kpar_kperp(
        linear, 3.0, k_par_iMpc, k_perp_iMpc, arinyo
    )
    P1D_Mpc = model.P1D_Mpc(linear, 3.0, k_iMpc, arinyo)

    np.testing.assert_allclose(P3D_Mpc_k_mu, EXPECTED_P3D_MPC, rtol=5e-5)
    np.testing.assert_allclose(
        P3D_Mpc_kpar_kperp, EXPECTED_P3D_MPC, rtol=5e-5
    )
    np.testing.assert_allclose(
        P3D_Mpc_k_mu, P3D_Mpc_kpar_kperp, rtol=1e-12
    )
    np.testing.assert_allclose(P1D_Mpc, EXPECTED_P1D_MPC, rtol=5e-5)


def test_saved_emulator_reload_preserves_prediction(central_prediction, tmp_path):
    emulator, expected = central_prediction
    model_prefix = tmp_path / "central_emulator"
    metadata = np.load(
        MODEL_DIRECTORY / "forest_mpg_metadata.npy", allow_pickle=True
    ).item()
    torch.save(emulator.emulator.state_dict(), str(model_prefix) + ".pt")
    np.save(str(model_prefix) + "_metadata.npy", metadata)
    reloaded = P3DEmulator(
        key=None,
        model_path=str(model_prefix),
        transf_file=str(MODEL_DIRECTORY / "forest_mpg_transf.npy"),
        Nrealizations=3000,
    )

    actual = _scalar_prediction(reloaded.evaluate(CENTRAL_INPUT, seed=0))
    for name in expected:
        assert actual[name] == pytest.approx(expected[name], rel=0, abs=0)


def test_batched_predictions_reuse_deterministic_latent_samples(
    central_prediction,
):
    emulator, _ = central_prediction
    inputs = [
        dict(CENTRAL_INPUT, mF=CENTRAL_INPUT["mF"] + offset)
        for offset in (-0.02, 0.0, 0.02)
    ]

    first = emulator.evaluate(inputs, Nrealizations=64, seed=17)
    cached_latent = emulator._latent_cache
    second = emulator.evaluate(inputs, Nrealizations=64, seed=17)

    assert emulator._latent_cache is cached_latent
    for name in EXPECTED_ARINYO:
        assert np.asarray(first[name]).shape == (len(inputs),)
        np.testing.assert_array_equal(first[name], second[name])

    emulator.evaluate(inputs, Nrealizations=64, seed=18)


def test_latent_indices_preserve_independent_batch_predictions(
    central_prediction,
):
    emulator, _ = central_prediction
    first = [
        dict(CENTRAL_INPUT, mF=CENTRAL_INPUT["mF"] + offset)
        for offset in (-0.02, 0.0, 0.02)
    ]
    second = [
        dict(CENTRAL_INPUT, gamma=CENTRAL_INPUT["gamma"] + offset)
        for offset in (-0.03, 0.0, 0.03)
    ]

    expected_first = emulator.evaluate(first, Nrealizations=64, seed=9)
    expected_second = emulator.evaluate(second, Nrealizations=64, seed=9)
    combined = emulator.evaluate(
        first + second,
        Nrealizations=64,
        seed=9,
        latent_indices=[0, 1, 2, 0, 1, 2],
    )

    for name in EXPECTED_ARINYO:
        expected = np.concatenate(
            [np.asarray(expected_first[name]), np.asarray(expected_second[name])]
        )
        np.testing.assert_allclose(combined[name], expected, rtol=2e-6, atol=1e-8)


def test_batched_p1d_matches_scalar_quadrature(central_prediction):
    """The public P1D method dispatches stacked linear grids without a walker loop."""
    _, arinyo = central_prediction
    model = ArinyoModel(cosmology.Cosmology(cosmo_params_dict=CENTRAL_COSMOLOGY))
    zs = np.array([3.0])
    cosmologies = [{}, {"ns": CENTRAL_COSMOLOGY["ns"] + 1.0e-4}]
    linear_batch = model.linear_theory_batch(zs, cosmologies)
    k = np.array([[[0.2, 0.7, 2.0]], [[0.2, 0.7, 2.0]]])
    arinyo_batch = {name: np.array([[value], [value]]) for name, value in arinyo.items()}
    batched = model.P1D_Mpc(linear_batch, zs, k, arinyo_batch)
    for index, parameters in enumerate(cosmologies):
        scalar = model.P1D_Mpc(
            model.linear_theory(zs, new_cosmo_params=parameters), zs, k[index], arinyo
        )
        np.testing.assert_allclose(batched[index], scalar, rtol=1.0e-12)
