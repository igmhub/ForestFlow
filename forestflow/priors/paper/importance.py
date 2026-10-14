"""
Combine weighted importance samples.
"""

from typing import Any
from numpy.typing import ArrayLike

import numpy as np
from copy import deepcopy

from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

from getdist import plots


def fit_gaussian(samples: Any, make_plot: bool | None=True) -> Any:
    """
    Fit a weighted Gaussian to two BAO-derived GetDist parameters.

    Parameters
    ----------
    samples : getdist.MCSamples
        Samples containing ``b_delta_sigma8`` and ``b_eta_f_sigma8``.
    make_plot : bool, default=True
        Overlay 68% and 95% fitted ellipses on a new GetDist figure.

    Returns
    -------
    dict
        Weighted means, standard deviations, and correlation under keys
        ``x_val``, ``y_val``, ``x_err``, ``y_err``, and ``r``.
    """
    p1, p2 = "b_delta_sigma8", "b_eta_f_sigma8"

    # --- Gaussian approximation from samples ---
    params = samples.getParams()
    x = getattr(params, p1)
    y = getattr(params, p2)
    w = samples.weights.copy()
    w = w / np.sum(w)  # normalize weights

    data = np.vstack([x, y]).T
    mean = np.average(data, axis=0, weights=w)
    cov = np.cov(data, rowvar=False, aweights=w)

    x_val, y_val = mean
    x_err = np.sqrt(cov[0, 0])
    y_err = np.sqrt(cov[1, 1])
    r = cov[0, 1] / (x_err * y_err)

    fits = {
        "x_val": x_val,
        "y_val": y_val,
        "x_err": x_err,
        "y_err": y_err,
        "r": r,
    }

    # eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # 68% contour scaling for 2D Gaussian
    # chi2_2(0.68) ≈ 2.30
    scale = np.sqrt(2.30)

    theta = np.linspace(0, 2 * np.pi, 400)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    ellipse1 = (eigvecs @ np.diag(np.sqrt(eigvals)) @ circle) * scale
    ellipse1[0] += mean[0]
    ellipse1[1] += mean[1]

    scale = np.sqrt(5.99)  # 95% contour scaling for 2D Gaussian
    ellipse2 = (eigvecs @ np.diag(np.sqrt(eigvals)) @ circle) * scale
    ellipse2[0] += mean[0]
    ellipse2[1] += mean[1]

    # --- GetDist plot ---
    # g = plots.get_subplot_plotter()

    if make_plot:
        g = plots.get_subplot_plotter(width_inch=6)
        g.plot_2d(samples, p1, p2, filled=True)

        ax = g.subplots[0, 0]
        ax.plot(ellipse1[0], ellipse1[1], color="k", lw=2, label="Gaussian (68%)")
        ax.plot(
            ellipse2[0], ellipse2[1], color="k", lw=2, ls="--", label="Gaussian (95%)"
        )
        ax.legend()

    return fits


def gaussian_chi2(x: Any, y: Any, x_val: Any, y_val: Any, x_err: ArrayLike, y_err: ArrayLike, r: Any) -> Any:
    """
    Compute a correlated two-dimensional Gaussian chi-squared.

    Parameters
    ----------
    x, y : array_like
        Coordinates at which to evaluate the Gaussian.
    x_val, y_val : float
        Gaussian mean coordinates.
    x_err, y_err : float
        Marginal standard deviations.
    r : float
        Correlation coefficient.

    Returns
    -------
    ndarray or float
        Correlated Gaussian chi-squared.
    """
    chi2 = (
        (y - y_val) ** 2 / y_err**2
        + (x - x_val) ** 2 / x_err**2
        - 2 * r * (x - x_val) * (y - y_val) / y_err / x_err
    ) / (1 - r * r)
    return chi2


def combine_inplace(samples: ArrayLike, fit: Any) -> Any:
    """
    Reweight an ``MCSamples`` object in place using a Gaussian fit.

    Parameters
    ----------
    samples : getdist.MCSamples
        Samples whose log-likelihood weights are modified in place.
    fit : mapping
        Output of :func:`fit_gaussian`.

    Returns
    -------
    getdist.MCSamples
        The same, now reweighted, object.
    """
    p1, p2 = "b_delta_sigma8", "b_eta_f_sigma8"

    # --- Gaussian approximation from samples ---
    params = samples.getParams()
    x = getattr(params, p1)
    y = getattr(params, p2)

    logw = 0.5 * gaussian_chi2(
        x,
        y,
        fit["x_val"],
        fit["y_val"],
        fit["x_err"],
        fit["y_err"],
        fit["r"],
    )

    samples.reweightAddingLogLikes(logw)

    return samples


def combine(samples: ArrayLike, fit: Any, label: Any) -> Any:

    """
    Return a copied ``MCSamples`` object reweighted by a Gaussian fit.

    Parameters
    ----------
    samples : getdist.MCSamples
        Source samples, left unmodified.
    fit : mapping
        Output of :func:`fit_gaussian`.
    label : str
        Label assigned to the copied samples.

    Returns
    -------
    getdist.MCSamples
        Copied samples with Gaussian importance weights applied.
    """
    p1, p2 = "b_delta_sigma8", "b_eta_f_sigma8"

    # extract parameters
    params = samples.getParams()
    x = getattr(params, p1)
    y = getattr(params, p2)

    logw = 0.5 * gaussian_chi2(
        x,
        y,
        fit["x_val"],
        fit["y_val"],
        fit["x_err"],
        fit["y_err"],
        fit["r"],
    )

    # copy samples
    new_samples = deepcopy(samples)

    # compute new weights
    new_weights = samples.weights * np.exp(-logw)

    new_samples.setSamples(
        samples.samples,
        weights=new_weights,
        loglikes=samples.loglikes,
    )

    new_samples.label = label

    return new_samples
