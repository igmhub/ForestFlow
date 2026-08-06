"""
Rebin three-dimensional power spectra.
"""

from collections.abc import Callable
from typing import Any
from numpy.typing import ArrayLike, NDArray

import numpy as np


def p3d_rebin_mu(k3d: Any, mu: Any, p3d: ArrayLike, kmu_modes: Any, n_mubins: int | None=4, return_modes: bool | None=False) -> NDArray[Any]:
    """
    Rebin p3d to fewer mu bins

    Parameters
    ----------
    k3d : object
        K3d used by the calculation.
    mu : object
        Mu used by the calculation.
    p3d : numpy.ndarray
        P3d used by the calculation.
    kmu_modes : object
        Kmu modes used by the calculation.
    n_mubins : int, optional
        N mubins used by the calculation.
    return_modes : bool, optional
        Return modes used by the calculation.

    Returns
    -------
    object
        Result produced when the function is used to rebin p3d to fewer mu bins.
    """

    def wmean(data: ArrayLike, weight: Any) -> Any:
        """
        Weighted mean

        Parameters
        ----------
        data : numpy.ndarray
            Input data.
        weight : object
            Weight used by the calculation.

        Returns
        -------
        object
            Result produced when the function is used to weighted mean.
        """
        return np.sum(data * weight) / np.sum(weight)

    n_kbins = k3d.shape[0]
    mu_bins = np.linspace(0, 1, n_mubins + 1)
    # get modes for each bin
    modes = np.zeros((n_kbins, k3d.shape[1]))
    for jj in range(n_kbins):
        for ii in range(k3d.shape[1]):
            flag = str(jj) + "_" + str(ii) + "_k"
            if flag in kmu_modes:
                modes[jj, ii] = kmu_modes[flag].shape[0]

    k3d_new = np.zeros((n_kbins, n_mubins))
    mu_new = np.zeros((n_kbins, n_mubins))
    modes_new = np.zeros((n_kbins, n_mubins))
    p3d_new = np.zeros((n_kbins, n_mubins))
    for ii in range(n_mubins):
        for jj in range(n_kbins):
            if ii != n_mubins - 1:
                _ = (mu[jj] >= mu_bins[ii]) & (mu[jj] < mu_bins[ii + 1])
            else:
                _ = (mu[jj] >= mu_bins[ii]) & (mu[jj] <= mu_bins[ii + 1])
            k3d_new[jj, ii] = wmean(k3d[jj, _], modes[jj, _])
            modes_new[jj, ii] = np.sum(modes[jj, _])
            mu_new[jj, ii] = wmean(mu[jj, _], modes[jj, _])
            p3d_new[jj, ii] = wmean(p3d[jj, _], modes[jj, _])

    if return_modes:
        return k3d_new, mu_new, p3d_new, mu_bins, modes_new
    else:
        return k3d_new, mu_new, p3d_new, mu_bins


def get_p3d_modes(kmax: int | float, lbox: float | None=67.5, k_Mpc_max: int | None=20, n_k_bins: int | None=20, n_mu_bins: int | None=16) -> NDArray[Any]:
    """
    Get k and mu of p3d modes

    Parameters
    ----------
    kmax : int or float
        Kmax used by the calculation.
    lbox : float, optional
        Lbox used by the calculation.
    k_Mpc_max : int, optional
        K mpc max used by the calculation.
    n_k_bins : int, optional
        N k bins used by the calculation.
    n_mu_bins : int, optional
        N mu bins used by the calculation.

    Returns
    -------
    object
        Result produced when the function is used to get k and mu of p3d modes.
    """

    # fundamental frequency
    k_fun = 2 * np.pi / lbox

    # define k-binning (in 1/Mpc)
    lnk_max = np.log(k_Mpc_max)
    # set minimum k to make sure we cover fundamental mode
    lnk_min = np.log(0.9999 * k_fun)
    lnk_bin_max = lnk_max + (lnk_max - lnk_min) / (n_k_bins - 1)
    lnk_bin_edges = np.linspace(lnk_min, lnk_bin_max, n_k_bins + 1)
    k_bin_edges = np.exp(lnk_bin_edges)
    # define mu-binning
    mu_bin_edges = np.linspace(0.0, 1.0, n_mu_bins + 1)

    ind = np.argwhere(k_bin_edges > kmax)[0, 0]
    k_bin_edges = k_bin_edges[: ind + 1]
    n_k_bins = k_bin_edges.shape[0] - 1
    nn = k_bin_edges[-1] // k_fun + 1

    # define grid of k modes
    _ = np.mgrid[-nn : nn + 1 : 1, -nn : nn + 1 : 1, -nn : nn + 1 : 1] * k_fun
    xgrid, ygrid, zgrid = _
    # nper = np.sqrt(nx**2+ny**2)
    kgrid = np.sqrt(xgrid**2 + ygrid**2 + zgrid**2)
    mugrid = np.abs(zgrid / kgrid)

    dict_out = {}
    for ii in range(n_k_bins):
        for jj in range(n_mu_bins):
            _ = (
                (kgrid > k_bin_edges[ii])
                & (kgrid <= k_bin_edges[ii + 1])
                & (mugrid >= mu_bin_edges[jj])
                & (mugrid <= mu_bin_edges[jj + 1])
            )
            if np.sum(_) != 0:
                flag = str(ii) + "_" + str(jj)
                dict_out[flag + "_k"] = kgrid[_]
                dict_out[flag + "_mu"] = mugrid[_]

    return dict_out


def get_p3d_modes_kparkper(
    lbox: float | None=67.5, kpar_Mpc_max: int | None=20, n_k_bins: int | None=20, n_mu_bins: int | None=16
) -> NDArray[Any]:
    """
    Get k and mu of p3d modes

    Parameters
    ----------
    lbox : float, optional
        Lbox used by the calculation.
    kpar_Mpc_max : int, optional
        Kpar mpc max used by the calculation.
    n_k_bins : int, optional
        N k bins used by the calculation.
    n_mu_bins : int, optional
        N mu bins used by the calculation.

    Returns
    -------
    object
        Result produced when the function is used to get k and mu of p3d modes.
    """

    # fundamental frequency
    k_fun = 2 * np.pi / lbox

    # define k-binning (in 1/Mpc)
    lnk_max = np.log(k_Mpc_max)
    # set minimum k to make sure we cover fundamental mode
    lnk_min = np.log(0.9999 * k_fun)
    lnk_bin_max = lnk_max + (lnk_max - lnk_min) / (n_k_bins - 1)
    lnk_bin_edges = np.linspace(lnk_min, lnk_bin_max, n_k_bins + 1)
    k_bin_edges = np.exp(lnk_bin_edges)
    # define mu-binning
    mu_bin_edges = np.linspace(0.0, 1.0, n_mu_bins + 1)

    ind = np.argwhere(k_bin_edges > kmax)[0, 0]
    k_bin_edges = k_bin_edges[: ind + 1]
    n_k_bins = k_bin_edges.shape[0] - 1
    nn = k_bin_edges[-1] // k_fun + 1

    # define grid of k modes
    _ = np.mgrid[-nn : nn + 1 : 1, -nn : nn + 1 : 1, -nn : nn + 1 : 1] * k_fun
    xgrid, ygrid, zgrid = _
    # nper = np.sqrt(nx**2+ny**2)
    kgrid = np.sqrt(xgrid**2 + ygrid**2 + zgrid**2)
    mugrid = np.abs(zgrid / kgrid)

    dict_out = {}
    for ii in range(n_k_bins):
        for jj in range(n_mu_bins):
            _ = (
                (kgrid > k_bin_edges[ii])
                & (kgrid <= k_bin_edges[ii + 1])
                & (mugrid >= mu_bin_edges[jj])
                & (mugrid <= mu_bin_edges[jj + 1])
            )
            if np.sum(_) != 0:
                flag = str(ii) + "_" + str(jj)
                dict_out[flag + "_k"] = kgrid[_]
                dict_out[flag + "_mu"] = mugrid[_]

    return dict_out


def p3d_allkmu(
    model: Callable[..., Any],
    zs: Any,
    arinyo: Any,
    kmu_modes: Any,
    nk: int | None=14,
    nmu: int | None=16,
    compute_plin: bool | None=True,
) -> NDArray[Any]:
    """
    Get p3d and plin for all k-mu bins

    Parameters
    ----------
    model : callable
        Model used by the calculation.
    zs : object
        Zs used by the calculation.
    arinyo : object
        Arinyo used by the calculation.
    kmu_modes : object
        Kmu modes used by the calculation.
    nk : int, optional
        Nk used by the calculation.
    nmu : int, optional
        Nmu used by the calculation.
    compute_plin : bool, optional
        Compute plin used by the calculation.

    Returns
    -------
    object
        Result produced when the function is used to get p3d and plin for all k-mu bins.
    """
    p3d = np.zeros((nk, nmu))
    if compute_plin:
        plin = np.zeros((nk, nmu))

    for ii in range(nk):
        # print("ii = ", ii, " / ", nk)
        for jj in range(nmu):
            flag = str(ii) + "_" + str(jj)
            if flag + "_k" in kmu_modes:
                kev = kmu_modes[flag + "_k"]
                muev = kmu_modes[flag + "_mu"]
                p3d_allmodes = model.P3D_Mpc(zs, kev, muev, arinyo)
                p3d[ii, jj] = np.mean(p3d_allmodes)
                if compute_plin:
                    plin_allmodes = model.linP_Mpc(zs, kev)
                    plin[ii, jj] = np.mean(plin_allmodes)
    if compute_plin:
        return p3d, plin
    else:
        return p3d
