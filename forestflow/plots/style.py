"""
House plotting style of ForestFlow figures.

Importing ForestFlow never changes matplotlib's global state: the style is
applied explicitly by :func:`apply_house_style`, which every plotting entry
point calls (through :func:`house_style`) before drawing. Labels of the
compressed linear-power parameters use ``\\star`` (``\\Delta^2_\\star``,
``n_\\star``).
"""

from __future__ import annotations

import functools
from typing import Any, Callable

__all__ = ["HOUSE_STYLE", "apply_house_style", "house_style"]

#: rcParams of the house style: STIX text and mathtext fonts.
HOUSE_STYLE = {"font.family": "STIXGeneral", "mathtext.fontset": "stix"}


def apply_house_style() -> None:
    """Set the ForestFlow house style (STIX fonts) in matplotlib's rcParams."""
    from matplotlib import rcParams

    rcParams.update(HOUSE_STYLE)


def house_style(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate a plotting function so that it applies the house style first."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        apply_house_style()
        return func(*args, **kwargs)

    return wrapper
