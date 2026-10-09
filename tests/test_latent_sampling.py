"""Fast invariants for ForestFlow latent sampling methods."""

import torch

from forestflow.emulator.p1d import P1DEmulator
from forestflow.emulator.p3d_cinn import P3DEmulator


def _emulator_stub():
    emulator = object.__new__(P3DEmulator)
    emulator.dim_inputSpace = 3
    return emulator


def test_nested_gaussian_draws_have_stable_prefixes():
    emulator = _emulator_stub()
    small = emulator._draw_latents(
        2, 8, torch.device("cpu"), 17, None, "gaussian", "nested", None
    ).reshape(2, 8, 3)
    large = emulator._draw_latents(
        2, 16, torch.device("cpu"), 17, None, "gaussian", "nested", None
    ).reshape(2, 16, 3)
    torch.testing.assert_close(small, large[:, :8])


def test_antithetic_draws_are_paired_and_require_even_count():
    emulator = _emulator_stub()
    values = emulator._draw_latents(
        1, 10, torch.device("cpu"), 3, None, "antithetic", "nested", None
    ).reshape(1, 10, 3)
    torch.testing.assert_close(values[:, 0::2], -values[:, 1::2])


def test_sobol_draws_are_reproducible_and_finite():
    emulator = _emulator_stub()
    first = emulator._draw_latents(
        1, 16, torch.device("cpu"), 91, None, "sobol", "nested", None
    )
    second = emulator._draw_latents(
        1, 16, torch.device("cpu"), 91, None, "sobol", "nested", None
    )
    torch.testing.assert_close(first, second)
    assert torch.isfinite(first).all()


def test_p1d_adapter_forwards_sampling_options_and_invalidates_cache():
    class FakeP3D:
        def __init__(self):
            self.calls = []

        def evaluate(self, calls, **options):
            self.calls.append((calls, options))
            return {"bias": 1.0}

    adapter = object.__new__(P1DEmulator)
    adapter.emulator = FakeP3D()
    adapter.sampling_options = {
        "sampler": "sobol",
        "statistic": "median",
        "aggregation_space": "physical",
        "draw_policy": "nested",
    }
    adapter._prediction_cache = {"old": "prediction"}
    adapter.set_sampling_options(statistic="mean")
    assert adapter._prediction_cache is None
    adapter._evaluate_emulator([{"input": 1.0}])
    assert adapter.emulator.calls[0][1] == adapter.sampling_options


def test_p1d_prediction_key_includes_sampling_options():
    adapter = object.__new__(P1DEmulator)
    adapter.emu_params = ("x",)
    adapter.sampling_options = {
        "sampler": "gaussian",
        "statistic": "mean",
        "aggregation_space": "transformed",
        "draw_policy": "nested",
    }
    first = adapter._prediction_key({"x": 1.0})
    adapter.sampling_options["sampler"] = "sobol"
    second = adapter._prediction_key({"x": 1.0})
    assert first != second
