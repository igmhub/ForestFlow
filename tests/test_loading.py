"""The common emulator loader: one call, different emulators by label."""

import os

import numpy as np
import pytest

from forestflow.emulator import P1DEmulator, load_emulator
from forestflow.emulator.arinyo_gp import ArinyoGPEmulator
from forestflow.emulator.loading import gp_emulator_path
from forestflow.emulator.p3d_cinn import DEFAULT_EMULATOR_KEY, P3DEmulator

needs_gp = pytest.mark.skipif(
    not os.path.isfile(gp_emulator_path("forest_mpg_gp")),
    reason="forest_mpg_gp bundle is not installed",
)
INPUT = {"Delta2_p": 0.35, "n_p": -2.3, "mF": 0.66, "sigT_Mpc": 0.128,
         "gamma": 1.51, "kF_Mpc": 10.6}


def test_default_label_loads_the_default_cinn():
    emulator = load_emulator()
    assert isinstance(emulator, P3DEmulator)
    assert emulator.uses_sampling
    assert load_emulator(DEFAULT_EMULATOR_KEY).output_labels == emulator.output_labels


def test_unknown_gp_options_are_rejected():
    with pytest.raises(TypeError):
        load_emulator("forest_mpg_gp", not_an_option=1)


@needs_gp
def test_gp_and_cinn_share_the_interface():
    cinn, gp = load_emulator(), load_emulator("forest_mpg_gp", compile_model=True)
    assert isinstance(gp, ArinyoGPEmulator) and not gp.uses_sampling
    assert list(gp.input_labels) == list(cinn.input_labels)
    assert list(gp.output_labels) == list(cinn.output_labels)
    assert gp.kp_iMpc == cinn.kp_iMpc and gp.zmax == cinn.zmax
    assert gp.list_sim_cube == cinn.list_sim_cube
    out_gp, out_cinn = gp.evaluate(emu_params=INPUT), cinn.evaluate(emu_params=INPUT)
    assert out_gp.keys() == out_cinn.keys()
    for name in out_gp:
        assert np.isfinite(out_gp[name])


@needs_gp
def test_gp_mean_matches_scikit_learn_predict():
    gp = load_emulator("forest_mpg_gp")
    matrix, _ = gp._input_matrix([INPUT, dict(INPUT, mF=0.5)])
    np.testing.assert_array_equal(gp._posterior_mean(matrix), gp.gp.predict(matrix))


@needs_gp
def test_p1d_emulator_accepts_either_label():
    for label in ("forest_mpg_cinn", "forest_mpg_gp"):
        emulator = P1DEmulator(name_emu=label)
        assert emulator.emulator_label == label
        out = emulator._evaluate_emulator([INPUT, dict(INPUT, mF=0.5)])
        assert np.asarray(out["bias"]).shape == (2,)
