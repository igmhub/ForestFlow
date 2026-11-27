import numpy as np
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


def test_sobol_antithetic_draws_are_paired():
    emulator = _emulator_stub()
    values = emulator._draw_latents(
        1, 16, torch.device("cpu"), 7, None, "sobol_antithetic", "nested", None
    ).reshape(1, 16, 3)
    torch.testing.assert_close(values[:, 0::2], -values[:, 1::2])


def test_stable_group_ids_do_not_depend_on_chunk_membership():
    emulator = _emulator_stub()
    combined = emulator._draw_latents(
        2, 16, torch.device("cpu"), 12, None, "sobol", "nested", None,
        latent_group_ids=[41, 93],
    ).reshape(2, 16, 3)
    alone = emulator._draw_latents(
        1, 16, torch.device("cpu"), 12, None, "sobol", "nested", None,
        latent_group_ids=[93],
    ).reshape(1, 16, 3)
    torch.testing.assert_close(combined[1], alone[0])


def test_p1d_adapter_forwards_sampling_options_and_latent_block_zero():
    class FakeP3D:
        output_labels = ("bias",)

        def __init__(self):
            self.calls = []

        def evaluate(self, calls, **options):
            self.calls.append((calls, options))
            return {"bias": np.ones(len(calls))}

    adapter = object.__new__(P1DEmulator)
    adapter.emulator = FakeP3D()
    adapter.emu_params = ("x",)
    adapter.sampling_options = {
        "sampler": "sobol",
        "statistic": "median",
        "aggregation_space": "physical",
        "draw_policy": "nested",
    }
    adapter.set_sampling_options(statistic="mean")
    out = adapter.evaluate_arinyo({"x": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])})
    assert out["bias"].shape == (2, 3)
    calls, options = adapter.emulator.calls[0]
    assert len(calls) == 6
    # every row draws latent block 0: independent of batch and redshift order
    np.testing.assert_array_equal(options.pop("latent_indices"), np.zeros(6))
    assert options == adapter.sampling_options
