"""Utilities for evaluating model predictions averaged over measured bins."""

import numpy as np


def logarithmic_bin_edges(centres, edges=None, *, name="k"):
    """Return explicit edges or infer them from strictly log-spaced centres.

    A single centre is intrinsically insufficient to determine bin widths, so
    callers must provide edges in that case.
    """
    centres = np.asarray(centres, dtype=float)
    if centres.ndim != 1 or centres.size == 0 or np.any(centres <= 0):
        raise ValueError(f"{name} centres must be a non-empty positive 1D array")
    if edges is not None:
        edges = np.asarray(edges, dtype=float)
        if (edges.shape != (centres.size + 1,) or np.any(edges <= 0)
                or np.any(np.diff(edges) <= 0)):
            raise ValueError(
                f"{name}_edges must be positive, increasing, and have "
                f"len({name}) + 1 entries"
            )
        return edges
    log_centres = np.log(centres)
    if centres.size == 1 or np.any(np.diff(log_centres) <= 0):
        raise ValueError(
            f"Cannot infer logarithmic {name} bin edges; pass explicit {name}_edges"
        )
    log_edges = np.empty(centres.size + 1)
    log_edges[1:-1] = 0.5 * (log_centres[:-1] + log_centres[1:])
    log_edges[0] = log_centres[0] - 0.5 * (log_centres[1] - log_centres[0])
    log_edges[-1] = log_centres[-1] + 0.5 * (log_centres[-1] - log_centres[-2])
    return np.exp(log_edges)


def linear_bin_edges(centres, edges=None, *, name="coordinate"):
    """Return explicit edges or infer midpoint edges from increasing centres."""
    centres = np.asarray(centres, dtype=float)
    if centres.ndim != 1 or centres.size == 0 or not np.all(np.isfinite(centres)):
        raise ValueError(f"{name} centres must be a non-empty finite 1D array")
    if edges is not None:
        edges = np.asarray(edges, dtype=float)
        if (edges.shape != (centres.size + 1,) or not np.all(np.isfinite(edges))
                or np.any(np.diff(edges) <= 0)):
            raise ValueError(
                f"{name}_edges must be finite, increasing, and have "
                f"len({name}) + 1 entries"
            )
        return edges
    if centres.size == 1 or np.any(np.diff(centres) <= 0):
        raise ValueError(
            f"Cannot infer linear {name} bin edges; pass explicit {name}_edges"
        )
    edges = np.empty(centres.size + 1)
    edges[1:-1] = 0.5 * (centres[:-1] + centres[1:])
    edges[0] = centres[0] - 0.5 * (centres[1] - centres[0])
    edges[-1] = centres[-1] + 0.5 * (centres[-1] - centres[-2])
    return edges
