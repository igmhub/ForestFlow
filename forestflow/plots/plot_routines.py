"""
Provide shared plotting and confidence-interval utilities.
"""

from typing import Any

import numpy as np
from scipy.stats import gaussian_kde


def plot_template(
    ax: Any,
    ax2: Any | None=None,
    ay2: Any | None=None,
    xlabel: Any | None=None,
    ylabel: Any | None=None,
    title: Any | None=None,
    legend: Any | None=None,
    legend_loc: str | None="best",
    ftsize: int | None=17,
    extra_xaxis: bool | None=False,
    extra_yaxis: bool | None=False,
    xcolor: str | None="k",
    ycolor: str | None="k",
    xcolor2: str | None="k",
    ycolor2: str | None="k",
    ylabelpad: Any | None=None,
    handlelength: int | None=2,
    legend_title: Any | None=None,
    legend_columns: int | None=1,
    ftsize_legend: int | None=15,
    title_fontsize: str | None="x-large",
) -> None:
    """
    Apply common labels, title, and optional legend formatting to axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Primary axes to configure.
    ax2, ay2 : matplotlib.axes.Axes, optional
        Reserved secondary axes. Tick formatting for these axes is currently
        disabled, so they are accepted only for API compatibility.
    xlabel : object, optional
        Primary x-axis label.
    ylabel : object, optional
        Primary y-axis label.
    title : object, optional
        Axes title.
    legend : object, optional
        Set to ``0`` to create a legend from plotted artists.
    legend_loc : str, optional
        Matplotlib legend location.
    ftsize : int, optional
        Base font size in points.
    extra_xaxis : bool, optional
        Reserved API-compatibility flag; currently has no effect.
    extra_yaxis : bool, optional
        Reserved API-compatibility flag; currently has no effect.
    xcolor : str, optional
        Primary x-label color.
    ycolor : str, optional
        Primary y-label color.
    xcolor2 : str, optional
        Reserved secondary x-axis color; currently unused.
    ycolor2 : str, optional
        Reserved secondary y-axis color; currently unused.
    ylabelpad : object, optional
        Padding between the y label and axes.
    handlelength : int, optional
        Legend handle length.
    legend_title : object, optional
        Legend title.
    legend_columns : int, optional
        Number of legend columns.
    ftsize_legend : int, optional
        Legend font size in points.
    title_fontsize : str, optional
        Font-size specifier for the legend title.
    """

    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))

    if legend == 0:
        ax.legend(
            fontsize=ftsize_legend,
            loc=legend_loc,
            handlelength=handlelength,
            title=legend_title,
            title_fontsize=title_fontsize,
            ncol=legend_columns,
        )

    if title:
        ax.set_title(
            title,
            fontsize=ftsize + 2,
        )

    if xlabel:
        ax.set_xlabel(
            xlabel,
            fontsize=ftsize + 2,
            color=xcolor,
        )
    if ylabel:
        ax.set_ylabel(
            ylabel,
            labelpad=ylabelpad,
            fontsize=ftsize + 2,
            color=ycolor,
        )

    """for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(ftsize)
        tick.label.set_color(xcolor)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(ftsize)
        tick.label.set_color(ycolor)

    if extra_xaxis:
        for tick in ax2.xaxis.get_major_ticks():
            tick.label2.set_fontsize(ftsize)
            tick.label2.set_color(xcolor2)
    if extra_yaxis:
        for tick in ay2.yaxis.get_major_ticks():
            tick.label2.set_fontsize(ftsize)
            tick.label2.set_color(ycolor2)"""


def plot_vec(cen: Any, vv: Any, length: Any, ax: Any, label: Any, col: Any, direction: Any | None=None) -> None:
    """
    Draw a scaled quiver vector, optionally reorienting it upward-right.

    Parameters
    ----------
    cen, vv : array_like of shape (2,)
        Vector origin and Cartesian components.
    length : float
        Reference vector length used to set the quiver scale.
    ax : matplotlib.axes.Axes
        Axes receiving the quiver artist.
    label : str
        Legend label for the vector.
    col : matplotlib color
        Quiver color.
    direction : {"up_right"}, optional
        When ``"up_right"``, flip components as needed to place the arrow in
        an upward-right orientation.
    """

    # vectors look up and right
    if direction == "up_right":
        if (vv[0] < 0) & (vv[1] < 0):
            vv = abs(vv)
        if (vv[0] > 0) & (vv[1] < 0):
            vv = -vv

    ax.quiver(
        cen[0],
        cen[1],
        vv[0],
        vv[1],
        color=col,
        width=0.02,
        scale=1 / length,
        scale_units="xy",
        angles="xy",
        alpha=0.5,
        label=label,
    )

    return


def density_estimation(m1: Any, m2: Any, ntt: Any | None=100j) -> tuple[Any, ...]:
    """
    Estimate a two-dimensional Gaussian-kernel density on a regular grid.

    Parameters
    ----------
    m1, m2 : array_like of shape (n_samples,)
        Sample coordinates.
    ntt : complex, default=100j
        ``numpy.mgrid`` complex-step specification for both grid dimensions.

    Returns
    -------
    X, Y, Z : ndarray
        Mesh coordinates and Gaussian KDE values.
    """
    xmin = np.min(m1) * 0.95
    xmax = np.max(m1) * 1.05
    ymin = np.min(m2) * 0.95
    ymax = np.max(m2) * 1.05
    X, Y = np.mgrid[xmin:xmax:ntt, ymin:ymax:ntt]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    # ax.imshow(np.rot90(Z), cmap=cmap,
    # extent=[xmin, xmax, ymin, ymax])
    return X, Y, Z


def find_confidence_interval(x: Any, pdf: Any, confidence_level: Any) -> Any:
    """
    Return the probability-mass difference above a trial density level.

    Parameters
    ----------
    x : float
        Trial density threshold.
    pdf : array_like
        Discretized probability-density values.
    confidence_level : float
        Target enclosed probability mass.

    Returns
    -------
    float
        Mass above ``x`` minus ``confidence_level``; useful as a root-finding
        objective for a highest-density contour.
    """
    return pdf[pdf > x].sum() - confidence_level
