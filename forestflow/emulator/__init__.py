"""
Emulator models, training transforms, and prediction covariance.
"""
from .p1d import P1DEmulator
from .training import Transf_data, get_training_data
from .arinyo_gp import ArinyoGPConfig, ArinyoGPEmulator, train_arinyo_gp
from .loading import EMULATOR_LABELS, load_emulator

__all__ = ["P1DEmulator", "P3DEmulator", "Transf_data", "get_training_data", "ArinyoGPConfig", "ArinyoGPEmulator", "train_arinyo_gp", "load_emulator", "EMULATOR_LABELS"]


def __getattr__(name):
    """Import the torch-based cINN emulator only when it is requested."""
    if name == "P3DEmulator":
        from .p3d_cinn import P3DEmulator

        return P3DEmulator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
