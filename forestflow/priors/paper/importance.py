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


def fit_to_mean_cov(fit: Any) -> tuple[Any, Any]:
    """
    Convert a :func:`fit_gaussian` output to a mean vector and covariance.

    Parameters
    ----------
    fit : mapping
        Output of :func:`fit_gaussian`.

    Returns
    -------
    mean : ndarray of shape (2,)
        Means of ``b_delta_sigma8`` and ``b_eta_f_sigma8``.
    cov : ndarray of shape (2, 2)
        Their covariance.
    """
    mean = np.array([fit["x_val"], fit["y_val"]])
    cxy = fit["r"] * fit["x_err"] * fit["y_err"]
    cov = np.array([[fit["x_err"] ** 2, cxy], [cxy, fit["y_err"] ** 2]])
    return mean, cov


def _sqrtm_sym(cov: Any) -> Any:
    """Symmetric square root of a positive-definite matrix."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def combine_gaussian_corr(fit1: Any, fit2: Any, rho: float) -> tuple[Any, Any]:
    """
    Combine two correlated Gaussian measurements of the same two parameters.

    Parameters
    ----------
    fit1, fit2 : mapping
        Outputs of :func:`fit_gaussian`, e.g. for P1D and BAO.
    rho : float
        Correlation between the two measurements, ``|rho| < 1``. The
        cross-covariance is ``rho * C1^(1/2) C2^(1/2)``, with symmetric matrix
        square roots; ``rho=0`` is the independent combination.

    Returns
    -------
    mean : ndarray of shape (2,)
        Combined means of ``b_delta_sigma8`` and ``b_eta_f_sigma8``.
    cov : ndarray of shape (2, 2)
        Combined covariance.

    Notes
    -----
    Minimum-variance combination ``cov = (A^T S^-1 A)^-1`` and
    ``mean = cov A^T S^-1 d``, with ``d = (mean1, mean2)``, ``A = (I, I)^T``
    and ``S`` the joint covariance of the two measurements.
    """
    mean1, cov1 = fit_to_mean_cov(fit1)
    mean2, cov2 = fit_to_mean_cov(fit2)
    cov12 = rho * _sqrtm_sym(cov1) @ _sqrtm_sym(cov2)

    joint_cov = np.block([[cov1, cov12], [cov12.T, cov2]])
    icov = np.linalg.inv(joint_cov)
    amat = np.vstack([np.eye(2), np.eye(2)])
    cov = np.linalg.inv(amat.T @ icov @ amat)
    mean = cov @ amat.T @ icov @ np.concatenate([mean1, mean2])
    return mean, cov


def gaussian_ellipse(mean: Any, cov: Any, chi2: float | None = 2.30) -> Any:
    """
    Points on the ``chi2`` contour of a 2D Gaussian.

    Parameters
    ----------
    mean : array_like of shape (2,)
        Gaussian mean.
    cov : array_like of shape (2, 2)
        Gaussian covariance.
    chi2 : float, default=2.30
        Contour level, 2.30 for 68% and 5.99 for 95%.

    Returns
    -------
    ndarray of shape (2, 400)
        Contour coordinates.
    """
    theta = np.linspace(0, 2 * np.pi, 400)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    ellipse = _sqrtm_sym(cov) @ circle * np.sqrt(chi2)
    return ellipse + np.asarray(mean)[:, None]


def combine_corr(
    samples: Any, fit_p1d: Any, fit_bao: Any, rho: float, label: Any
) -> Any:
    """
    Return P1D samples reweighted to a P1D + BAO combination with correlation.

    Parameters
    ----------
    samples : getdist.MCSamples
        P1D samples, left unmodified.
    fit_p1d, fit_bao : mapping
        Outputs of :func:`fit_gaussian` for the P1D and BAO samples.
    rho : float
        Correlation between the P1D and BAO measurements, see
        :func:`combine_gaussian_corr`.
    label : str
        Label assigned to the copied samples.

    Returns
    -------
    getdist.MCSamples
        Copied samples with importance weights applied.

    Notes
    -----
    The weights are the ratio of the combined Gaussian to the Gaussian fit to
    the P1D samples. For ``rho=0`` this ratio is the BAO Gaussian, so the
    result matches :func:`combine`.
    """
    p1, p2 = "b_delta_sigma8", "b_eta_f_sigma8"

    params = samples.getParams()
    x = getattr(params, p1)
    y = getattr(params, p2)

    mean, cov = combine_gaussian_corr(fit_p1d, fit_bao, rho)
    err = np.sqrt(np.diag(cov))
    chi2_comb = gaussian_chi2(
        x, y, mean[0], mean[1], err[0], err[1], cov[0, 1] / err[0] / err[1]
    )
    chi2_p1d = gaussian_chi2(
        x,
        y,
        fit_p1d["x_val"],
        fit_p1d["y_val"],
        fit_p1d["x_err"],
        fit_p1d["y_err"],
        fit_p1d["r"],
    )
    logw = 0.5 * (chi2_comb - chi2_p1d)

    new_samples = deepcopy(samples)
    new_weights = samples.weights * np.exp(-(logw - np.min(logw)))
    new_samples.setSamples(
        samples.samples,
        weights=new_weights,
        loglikes=samples.loglikes,
    )
    new_samples.label = label

    return new_samples
