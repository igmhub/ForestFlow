"""
Locations of ForestFlow's data products and of external analysis data.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["get_path_repo", "get_data_path", "get_lya_data_path"]

#: Environment variable naming the root of the external Lyman-alpha data
#: (BAO chains, P1D chains, ACCEL2 measurements) read by the prior analysis.
LYA_DATA_ENV = "LYA_DATA_PATH"


def get_path_repo() -> Path:
    """Return the ForestFlow repository root inferred from the installed package."""
    import forestflow

    return Path(forestflow.__path__[0]).parent


def get_data_path() -> Path:
    """Return the folder holding ForestFlow's own data (``<repo>/data``)."""
    return get_path_repo() / "data"


def get_lya_data_path(path: str | os.PathLike | None = None) -> Path:
    """
    Return the root of the external Lyman-alpha analysis data.

    Parameters
    ----------
    path : str or path-like, optional
        Explicit root, taking priority over everything else.

    Returns
    -------
    pathlib.Path
        ``path`` if given; otherwise the ``LYA_DATA_PATH`` environment
        variable; otherwise ``<repo>/../../data``, i.e. the ``data`` folder
        next to the folder holding the sibling igmhub repositories.

    Notes
    -----
    The prior analysis (:mod:`forestflow.priors`) reads, relative to this
    root, ``lya_bao/dr1``, ``lya_bao/dr2`` and ``lya_bao/fits_andreu``.
    """
    if path is not None:
        return Path(path).expanduser()
    env = os.environ.get(LYA_DATA_ENV)
    if env:
        return Path(env).expanduser()
    return get_path_repo().parent.parent / "data"
