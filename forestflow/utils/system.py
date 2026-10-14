"""
Shared utility helpers.
"""
from __future__ import annotations

from os import PathLike
import torch

def print_memory_usage(step_description: str | PathLike[str]) -> None:
    """
    Print the current process memory usage.

    Parameters
    ----------
    step_description : str or pathlib.Path
        Label printed with the memory measurements.
    """
    import psutil

    process = psutil.Process()
    memory_info = process.memory_info()
    print(
        f"{step_description} - RSS: {memory_info.rss / (1024 ** 2):.2f} MB, VMS: {memory_info.vms / (1024 ** 2):.2f} MB"
    )
    if torch.cuda.is_available():
        print(
            f"GPU memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB"
        )
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
