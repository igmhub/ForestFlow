import torch

from forestflow.P3D_cINN import P3DEmulator


def test_compile_wraps_model_only_once(monkeypatch):
    emulator = object.__new__(P3DEmulator)
    model = torch.nn.Linear(2, 2)
    compiled_model = torch.nn.Sequential(model)
    emulator.emulator = model
    emulator._compiled = False
    calls = []

    def fake_compile(actual_model, *, mode):
        calls.append((actual_model, mode))
        return compiled_model

    monkeypatch.setattr(torch, "compile", fake_compile)

    emulator.compile()
    emulator.compile()

    assert calls == [(model, "reduce-overhead")]
    assert emulator.emulator is compiled_model
    assert emulator._compiled is True


