"""Deterministic inference using a distilled ForestFlow MLP."""

from collections.abc import Mapping, Sequence
import numpy as np
import torch


class DeterministicArinyoEmulator:
    """Predict ForestFlow transformed coefficients without latent sampling."""

    def __init__(self, model: torch.nn.Module, transf_data, input_labels: Sequence[str], output_labels: Sequence[str]):
        self.model = model.eval()
        self.transf_data = transf_data
        self.input_labels = tuple(input_labels)
        self.output_labels = tuple(output_labels)

    def predict_transformed(self, emu_params: Mapping[str, float] | Sequence[Mapping[str, float]]):
        single = isinstance(emu_params, Mapping)
        rows = [emu_params] if single else list(emu_params)
        normalized = []
        for params in rows:
            values = self.transf_data.transf_stand(params, type_stand="input", direct=True)
            normalized.append([values[name] for name in self.input_labels])
        device = next(self.model.parameters()).device
        with torch.no_grad():
            prediction = self.model(torch.as_tensor(normalized, dtype=torch.float32, device=device)).cpu().numpy()
        result = {name: prediction[:, index] for index, name in enumerate(self.output_labels)}
        return {name: values[0] for name, values in result.items()} if single else result

    def evaluate(self, emu_params: Mapping[str, float] | Sequence[Mapping[str, float]]):
        """Return physical coefficients using the teacher's exact inverse transform."""
        return self.transf_data.transf_stand(
            self.predict_transformed(emu_params), type_stand="output", direct=False
        )

    def save(self, path, *, metadata=None):
        """Save a deployable student and its exact transformation metadata."""
        from .network import MLPConfig
        linear_layers = [layer for layer in self.model.modules() if isinstance(layer, torch.nn.Linear)]
        config = MLPConfig(len(self.input_labels), len(self.output_labels),
                           linear_layers[0].out_features, len(linear_layers) - 1,
                           "tanh" if any(isinstance(layer, torch.nn.Tanh) for layer in self.model.modules()) else "silu")
        torch.save({"state_dict": self.model.state_dict(), "config": config,
                    "transf_data": self.transf_data, "input_labels": self.input_labels,
                    "output_labels": self.output_labels, "metadata": dict(metadata or {})}, path)

    @classmethod
    def load(cls, path, *, map_location="cpu"):
        """Load a student saved with :meth:`save`; only load trusted local files."""
        from .network import build_mlp
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model = build_mlp(payload["config"])
        model.load_state_dict(payload["state_dict"])
        return cls(model, payload["transf_data"], payload["input_labels"], payload["output_labels"])
