"""
Plots for leave-one-out ForestFlow covariance calculations.

The implementations live in :mod:`lace.plotting.covariance`; they are
re-exported here so that ``forestflow.plots.covariance`` keeps working.
"""

from lace.plotting.covariance import plot_l1o_correlation, plot_l1o_errors

__all__ = ["plot_l1o_correlation", "plot_l1o_errors"]
