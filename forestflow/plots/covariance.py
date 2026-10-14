"""
Plots for leave-one-out ForestFlow covariance calculations.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_l1o_correlation(cov_zk, *, ax=None):
    """
    Plot a leave-one-out redshift--wavenumber correlation matrix.

    Parameters
    ----------
    cov_zk : array_like of shape (n_bins, n_bins)
        Covariance of flattened redshift--wavenumber residuals.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure and axes are created when omitted.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the correlation image and colorbar.
    """
    if ax is None:
        _, ax = plt.subplots()

    scale = np.sqrt(np.diag(cov_zk))
    corr = cov_zk / np.outer(scale, scale)
    image = ax.imshow(corr, origin="lower", aspect="auto")
    ax.figure.colorbar(image, ax=ax, label="Correlation")
    ax.set_xlabel(r"$(z, k)$ bin")
    ax.set_ylabel(r"$(z, k)$ bin")
    return ax


def plot_l1o_errors(zz, k_Mpc, rel_diff, cov_zk, *, ax=None):
    """
    Plot leave-one-out standard deviations and absolute biases by redshift.

    Parameters
    ----------
    zz : array_like of shape (n_z,)
        Redshift values.
    k_Mpc : array_like of shape (n_k,)
        Wavenumbers in ``Mpc^-1``.
    rel_diff : array_like of shape (n_folds, n_z, n_k)
        Fractional prediction residuals.
    cov_zk : array_like of shape (n_z * n_k, n_z * n_k)
        Covariance for the flattened residual vector.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new axes is created when omitted.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing one solid standard-deviation line and one dashed bias
        line per redshift.
    """
    if ax is None:
        _, ax = plt.subplots()

    standard_deviation = np.sqrt(np.diag(cov_zk)).reshape(len(zz), len(k_Mpc))
    bias = np.abs(np.mean(rel_diff, axis=0))
    for iz, z in enumerate(zz):
        line = ax.plot(k_Mpc, standard_deviation[iz], label=rf"$z={z:.2f}$")[0]
        ax.plot(k_Mpc, bias[iz], linestyle="--", color=line.get_color())

    ax.set_xscale("log")
    ax.set_xlabel(r"$k\,[\mathrm{Mpc}^{-1}]$")
    ax.set_ylabel("Relative error")
    ax.legend()
    return ax
