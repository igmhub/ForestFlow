"""
Plotting utilities for ForestFlow analyses.

Every plotting function applies the house style (:func:`apply_house_style`)
before drawing; importing this package does not change matplotlib settings.
"""

from forestflow.plots.covariance import plot_l1o_correlation, plot_l1o_errors
from forestflow.plots.style import apply_house_style, house_style

__all__ = ["apply_house_style", "house_style", "plot_l1o_correlation", "plot_l1o_errors"]
