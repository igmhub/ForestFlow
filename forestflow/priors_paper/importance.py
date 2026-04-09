import numpy as np
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

from getdist import plots


def fit_gaussian(samples, make_plot=True):

    # assume `samples` is an MCSamples object
    """
    Compute Gaussian approximation of the 2D distribution from samples.

    Parameters
    ----------
    samples : MCSamples
        An MCSamples object containing the samples of the distribution.
    make_plot : bool, optional
        If True, make a GetDist plot of the samples with the Gaussian approximation.
        Default is True.

    Returns
    -------
    fits : dict
        A dictionary containing the Gaussian approximation parameters:
            "x_val", "y_val" : the mean values of x and y
            "x_err", "y_err" : the standard deviations of x and y
            "r" : the correlation coefficient between x and y
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


def gaussian_chi2(x, y, x_val, y_val, x_err, y_err, r):
    """Given central values and errors for Delta_L^2 and n_eff, and its
    cross-correlation coefficient r, compute Gaussian delta chi^2 at
    points (neff,DL2).
    """
    chi2 = (
        (y - y_val) ** 2 / y_err**2
        + (x - x_val) ** 2 / x_err**2
        - 2 * r * (x - x_val) * (y - y_val) / y_err / x_err
    ) / (1 - r * r)
    return chi2


def combine_inplace(samples, fit):

    # assume `samples` is an MCSamples object
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


def combine(samples, fit, label):

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
