"""One entry point for loading any ForestFlow Arinyo-parameter emulator.

Most users want the default settings, so loading differs between emulators only
by ``emulator_label``::

    from forestflow.emulator import load_emulator

    emulator = load_emulator()                    # cINN, "forest_mpg_cinn"
    emulator = load_emulator("forest_mpg_gp")     # Gaussian process

Both return an object whose ``evaluate(emu_params)`` maps the six emulator
inputs to the eight Arinyo parameters. Options specific to one emulator are
passed as keyword arguments, only when needed.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import forestflow
from forestflow.emulator.arinyo_gp import ArinyoGPEmulator
from forestflow.emulator.p3d_cinn import DEFAULT_EMULATOR_KEY, P3DEmulator

__all__ = ["EMULATOR_LABELS", "GP_EMULATOR_LABELS", "load_emulator", "load_l1o_emulator"]

#: Labels of the Gaussian-process emulators, stored in ``data/gp_emulators``.
GP_EMULATOR_LABELS = ("forest_mpg_gp",)
#: Labels of the emulators users are expected to choose between.
EMULATOR_LABELS = (DEFAULT_EMULATOR_KEY,) + GP_EMULATOR_LABELS


def gp_emulator_path(emulator_label: str) -> str:
    """Return the path of the GP bundle with the given label."""
    return os.path.join(
        os.path.dirname(forestflow.__path__[0]),
        "data",
        "gp_emulators",
        f"{emulator_label}.pkl",
    )


def load_emulator(emulator_label: str | None = DEFAULT_EMULATOR_KEY, **kwargs: Any):
    """Load an Arinyo-parameter emulator by label, with default options.

    Parameters
    ----------
    emulator_label : str, default="forest_mpg_cinn"
        ``"forest_mpg_cinn"`` loads the cINN (:class:`P3DEmulator`);
        ``"forest_mpg_gp"`` loads the Gaussian process
        (:class:`ArinyoGPEmulator`). Any other label is treated as a cINN
        bundle key, e.g. ``"l1O/forest_mpg_cinn_l1O_0"``. ``None`` loads a
        cINN from explicit ``model_path`` and ``transf_file`` keyword arguments.
    **kwargs
        Extra options forwarded to the emulator's constructor (cINN) or loader
        (GP), e.g. ``Nrealizations`` for the cINN. ``compile_model`` is
        accepted for every emulator and ignored by those that are not neural
        networks, so callers need not special-case the label.

    Returns
    -------
    P3DEmulator or ArinyoGPEmulator
        Emulator with ``evaluate``, ``input_labels``, ``output_labels``,
        ``uses_sampling`` and the training-domain metadata properties.
    """
    if emulator_label == "forest_mpg_fix":
        warnings.warn(
            "Emulator label 'forest_mpg_fix' was renamed 'forest_mpg_cinn'.",
            FutureWarning,
            stacklevel=2,
        )
        emulator_label = DEFAULT_EMULATOR_KEY
    if emulator_label in GP_EMULATOR_LABELS:
        kwargs.pop("compile_model", None)
        return ArinyoGPEmulator.load(gp_emulator_path(emulator_label), **kwargs)
    return P3DEmulator(key=emulator_label, **kwargs)


def load_l1o_emulator(emulator_label: str, simulation_index: int, simulation_label: str):
    """Load the leave-one-out emulator that excluded one hypercube simulation.

    Parameters
    ----------
    emulator_label : str
        Label of the full emulator, e.g. ``"forest_mpg_cinn"`` or
        ``"forest_mpg_gp"``.
    simulation_index, simulation_label : int, str
        Position and label of the excluded simulation, e.g. ``0`` and
        ``"mpg_0"``. cINN bundles are indexed by position, GP bundles by label.
    """
    if emulator_label in GP_EMULATOR_LABELS:
        return ArinyoGPEmulator.load(gp_emulator_path(f"l1O_{simulation_label}"))
    return P3DEmulator(key=f"l1O/{emulator_label}_l1O_{simulation_index}")
