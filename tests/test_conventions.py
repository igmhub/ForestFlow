import numpy as np
import pytest

from forestflow.conventions import (
    ARINYO_PARAMETER_NAMES,
    canonicalize_unit_keys,
    validate_wavenumber,
)
from forestflow.p1d import P1D_Mpc, P1D_kms


def test_arinyo_parameter_order_is_canonical():
    assert ARINYO_PARAMETER_NAMES == (
        "bias", "beta", "q1", "kvav", "av", "bv", "kp", "q2"
    )


def test_legacy_keys_are_canonicalized_without_overwriting_new_values():
    values = canonicalize_unit_keys({"k_Mpc": 1, "k_iMpc": 2, "p1d_Mpc": 3})
    assert values["k_iMpc"] == 2
    assert values["P1D_Mpc"] == 3


def test_wavenumber_contract():
    np.testing.assert_equal(validate_wavenumber([0.1, 1], name="k_iMpc"), [0.1, 1])
    with pytest.raises(ValueError, match="k_iMpc"):
        validate_wavenumber([[0.1]], name="k_iMpc")


def test_velocity_conversion_uses_one_power_for_p1d():
    def P3D_Mpc(z, k_iMpc, mu, params, **kwargs):
        return np.ones_like(k_iMpc)

    P3D_Mpc.coordinates = "k_mu"
    k_ikms = np.array([0.001, 0.002])
    conversion = 70.0
    expected_Mpc = P1D_Mpc(
        3.0,
        k_ikms * conversion,
        P3D_Mpc,
        k_perp_min_iMpc=0.01 * conversion,
        k_perp_max_iMpc=0.02 * conversion,
    )
    result_kms = P1D_kms(
        3.0,
        k_ikms,
        P3D_Mpc,
        conversion,
        k_perp_min_ikms=0.01,
        k_perp_max_ikms=0.02,
    )
    np.testing.assert_allclose(result_kms, expected_Mpc * conversion)
