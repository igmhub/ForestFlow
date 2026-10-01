"""Regression tests for the scalar ForestFlow P1D linear-theory cache."""

from types import SimpleNamespace

import numpy as np

from forestflow.emulator.p1d import P1DEmulator


class _LinearTheoryFactory:
    def __init__(self):
        self.requests = []

    def get_linear_theory(self, z, new_cosmo_params=None):
        self.requests.append(
            None if new_cosmo_params is None else dict(new_cosmo_params)
        )
        return SimpleNamespace(z=np.asarray(z, dtype=float))


def _emulator_with_controlled_linear_theory():
    emulator = object.__new__(P1DEmulator)
    factory = _LinearTheoryFactory()
    emulator.cosmo_params_dict = {"As": 2.0e-9, "ns": 0.96}
    emulator.model_Arinyo = SimpleNamespace(linear=factory)
    emulator.linear = None
    emulator._linear_cosmology_parameters = None
    return emulator, factory


def test_scalar_linear_cache_is_invariant_to_cosmology_request_order():
    emulator, factory = _emulator_with_controlled_linear_theory()

    emulator.set_linear_theory(3.0)
    fiducial_grid = emulator.linear
    emulator.set_linear_theory(3.0, {"As": 3.0e-9})
    perturbed_grid = emulator.linear
    emulator.set_linear_theory(3.0)

    assert len(factory.requests) == 3
    assert emulator.linear is not perturbed_grid
    assert emulator.linear is not fiducial_grid
    assert emulator._linear_cosmology_parameters == emulator.cosmo_params_dict


def test_scalar_linear_cache_reuses_an_unchanged_effective_cosmology():
    emulator, factory = _emulator_with_controlled_linear_theory()

    emulator.set_linear_theory([2.5, 3.0], {"As": 3.0e-9})
    first_grid = emulator.linear
    emulator.set_linear_theory([3.0, 2.5], {"As": 3.0e-9})

    assert len(factory.requests) == 1
    assert emulator.linear is first_grid
