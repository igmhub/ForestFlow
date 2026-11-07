"""
Rebin three-dimensional power spectra.
"""

from collections.abc import Callable
from typing import Any
from numpy.typing import ArrayLike, NDArray

import numpy as np



# Native MP-Gadget P3D measurement geometry.  These are the post-processing
# bin definitions, not estimates reconstructed from mode-weighted centres.
MPG_P3D_BINNING = {
    "Lbox_Mpc": 67.5,
    "k_grid_max_iMpc": 20.0,
    "n_k_bins": 20,
    "n_mu_bins": 16,
}


def get_P3D_k_mu_bin_edges(
    k_max_iMpc=None,
    Lbox_Mpc=MPG_P3D_BINNING["Lbox_Mpc"],
    k_grid_max_iMpc=MPG_P3D_BINNING["k_grid_max_iMpc"],
    n_k_bins=MPG_P3D_BINNING["n_k_bins"],
    n_mu_bins=MPG_P3D_BINNING["n_mu_bins"],
):
    """
    Return the exact logarithmic k and uniform mu edges of MPG P3D bins.

    The lowest k edge is exactly the fundamental mode ``2 pi / Lbox_Mpc``.
    Passing ``k_max_iMpc`` retains complete native bins through the first edge
    above that scale, matching the archive's radial-bin selection.
    """
    if k_grid_max_iMpc <= 0 or Lbox_Mpc <= 0:
        raise ValueError("k_grid_max_iMpc and Lbox_Mpc must be positive")
    if n_k_bins < 2 or n_mu_bins < 1:
        raise ValueError("n_k_bins must be >= 2 and n_mu_bins must be positive")
    k_fundamental_iMpc = 2 * np.pi / Lbox_Mpc
    log_k_min = np.log(k_fundamental_iMpc)
    log_k_max = np.log(k_grid_max_iMpc)
    log_last_edge = log_k_max + (log_k_max - log_k_min) / (n_k_bins - 1)
    k_iMpc_edges = np.exp(np.linspace(log_k_min, log_last_edge, n_k_bins + 1))
    if k_max_iMpc is not None:
        if k_max_iMpc <= 0:
            raise ValueError("k_max_iMpc must be positive")
        beyond = np.flatnonzero(k_iMpc_edges > k_max_iMpc)
        if len(beyond) == 0:
            raise ValueError("k_max_iMpc exceeds the parent MPG binning")
        k_iMpc_edges = k_iMpc_edges[: beyond[0] + 1]
    mu_edges = np.linspace(0.0, 1.0, n_mu_bins + 1)
    return k_iMpc_edges, mu_edges

def rebin_P3D_Mpc_mode_weighted(
    k_iMpc: Any,
    mu: Any,
    P3D_Mpc: ArrayLike,
    k_mu_modes: Any,
    n_mu_bins: int = 4,
    return_mode_counts: bool = False,
) -> NDArray[Any]:
    """
    Mode-weighted rebinning of measured ``P3D_Mpc(k, mu)``.

    This is the discrete-Fourier-mode counterpart to
    :func:`forestflow.statistics.p3d.P3D_Mpc_k_mu_bin_averaged`.
    It reweights values measured in fine ``(k, mu)`` cells by the number of
    discrete Fourier modes and combines them into uniform mu bins.  Use it for
    simulation measurements.  For an analytic model bin average, use the
    Arinyo-model method instead.

    Parameters
    ----------
    k_iMpc, mu, P3D_Mpc : array_like
        Fine-grid arrays with common shape ``(n_k, n_mu)``.  Wavenumbers are in
        inverse Mpc and power is in Mpc cubed.
    k_mu_modes : mapping
        Dictionary returned by :func:`get_P3D_k_mu_modes`.
    n_mu_bins : int, default=4
        Number of uniform output mu bins.
    return_mode_counts : bool, default=False
        Also return the summed discrete-mode count in each output cell.
    """
    k_iMpc = np.asarray(k_iMpc, dtype=float)
    mu = np.asarray(mu, dtype=float)
    P3D_Mpc = np.asarray(P3D_Mpc, dtype=float)
    if k_iMpc.shape != mu.shape or k_iMpc.shape != P3D_Mpc.shape:
        raise ValueError("k_iMpc, mu, and P3D_Mpc must have the same shape")
    if k_iMpc.ndim != 2:
        raise ValueError("k_iMpc, mu, and P3D_Mpc must be two-dimensional")
    if not isinstance(n_mu_bins, int) or n_mu_bins < 1:
        raise ValueError("n_mu_bins must be a positive integer")

    def weighted_mean(values, weights):
        return np.sum(values * weights) / np.sum(weights)

    n_k_bins = k_iMpc.shape[0]
    mu_edges = np.linspace(0.0, 1.0, n_mu_bins + 1)
    mode_counts_fine = np.zeros(k_iMpc.shape)
    for k_index in range(n_k_bins):
        for mu_index in range(k_iMpc.shape[1]):
            key = f"{k_index}_{mu_index}_k"
            if key in k_mu_modes:
                mode_counts_fine[k_index, mu_index] = k_mu_modes[key].shape[0]

    k_output = np.full((n_k_bins, n_mu_bins), np.nan)
    mu_output = np.full((n_k_bins, n_mu_bins), np.nan)
    P3D_output = np.full((n_k_bins, n_mu_bins), np.nan)
    mode_counts = np.zeros((n_k_bins, n_mu_bins))
    for mu_bin in range(n_mu_bins):
        for k_index in range(n_k_bins):
            if mu_bin == n_mu_bins - 1:
                mask = (mu[k_index] >= mu_edges[mu_bin]) & (mu[k_index] <= mu_edges[mu_bin + 1])
            else:
                mask = (mu[k_index] >= mu_edges[mu_bin]) & (mu[k_index] < mu_edges[mu_bin + 1])
            mask &= np.isfinite(k_iMpc[k_index]) & np.isfinite(P3D_Mpc[k_index])
            weights = mode_counts_fine[k_index, mask]
            if np.sum(weights) == 0:
                continue
            k_output[k_index, mu_bin] = weighted_mean(k_iMpc[k_index, mask], weights)
            mu_output[k_index, mu_bin] = weighted_mean(mu[k_index, mask], weights)
            P3D_output[k_index, mu_bin] = weighted_mean(P3D_Mpc[k_index, mask], weights)
            mode_counts[k_index, mu_bin] = np.sum(weights)

    output = (k_output, mu_output, P3D_output, mu_edges)
    return output + (mode_counts,) if return_mode_counts else output

def get_P3D_k_mu_modes(
    k_max_iMpc: int | float,
    Lbox_Mpc: float = 67.5,
    k_grid_max_iMpc: int | float = 20,
    n_k_bins: int = 20,
    n_mu_bins: int = 16,
) -> NDArray[Any]:
    """
    Return discrete Fourier modes in logarithmic ``(k, mu)`` cells.

    Parameters
    ----------
    k_max_iMpc : float
        Largest wavenumber to retain, in inverse Mpc.
    Lbox_Mpc : float, default=67.5
        Simulation-box side length in Mpc.
    k_grid_max_iMpc : float, default=20
        Largest wavenumber used to construct the parent logarithmic grid.
    n_k_bins : int, default: 20
        Number of logarithmic radial bins in the parent grid.
    n_mu_bins : int, default: 16
        Number of uniform absolute-direction-cosine bins.

    Returns
    -------
    dict of str to ndarray
        Populated cell arrays named ``"i_j_k"`` and ``"i_j_mu"``. k values
        are in ``1 / Mpc`` and mu values lie in ``[0, 1]``.
    """

    k_iMpc_edges, mu_edges = get_P3D_k_mu_bin_edges(
        k_max_iMpc,
        Lbox_Mpc=Lbox_Mpc,
        k_grid_max_iMpc=k_grid_max_iMpc,
        n_k_bins=n_k_bins,
        n_mu_bins=n_mu_bins,
    )
    k_fun = 2 * np.pi / Lbox_Mpc
    n_k_bins = len(k_iMpc_edges) - 1
    n_mu_bins = len(mu_edges) - 1
    nn = int(k_iMpc_edges[-1] // k_fun + 1)

    # define grid of k modes
    _ = np.mgrid[-nn : nn + 1 : 1, -nn : nn + 1 : 1, -nn : nn + 1 : 1] * k_fun
    xgrid, ygrid, zgrid = _
    # nper = np.sqrt(nx**2+ny**2)
    kgrid = np.sqrt(xgrid**2 + ygrid**2 + zgrid**2)
    mugrid = np.divide(np.abs(zgrid), kgrid, out=np.zeros_like(kgrid), where=kgrid > 0)

    dict_out = {}
    for ii in range(n_k_bins):
        for jj in range(n_mu_bins):
            lower_k = kgrid >= k_iMpc_edges[ii] if ii == 0 else kgrid > k_iMpc_edges[ii]
            upper_mu = mugrid <= mu_edges[jj + 1] if jj == n_mu_bins - 1 else mugrid < mu_edges[jj + 1]
            _ = (
                lower_k
                & (kgrid <= k_iMpc_edges[ii + 1])
                & (mugrid >= mu_edges[jj])
                & upper_mu
            )
            if np.sum(_) != 0:
                flag = str(ii) + "_" + str(jj)
                dict_out[flag + "_k"] = kgrid[_]
                dict_out[flag + "_mu"] = mugrid[_]

    return dict_out


def _bin_lattice_plane(
    kperp2: NDArray[Any],
    kz: float,
    k_iMpc_edges: NDArray[Any],
    mu_edges: NDArray[Any],
) -> tuple[NDArray[Any], ...]:
    """
    Assign the lattice modes of one ``k_z`` plane to ``(k, mu)`` cells.

    The conventions follow :func:`get_P3D_k_mu_modes`: radial bins are
    ``(lower, upper]`` except the first, which includes its lower edge, and
    mu bins are ``[lower, upper)`` except the last, which includes ``mu=1``.
    """
    kk = np.sqrt(kperp2 + kz**2)
    mu = np.divide(np.abs(kz), kk, out=np.zeros_like(kk), where=kk > 0)
    ik = np.searchsorted(k_iMpc_edges, kk, side="left") - 1
    ik[kk == k_iMpc_edges[0]] = 0
    imu = np.searchsorted(mu_edges, mu, side="right") - 1
    imu[mu == mu_edges[-1]] = len(mu_edges) - 2
    valid = (
        (kk > 0)
        & (ik >= 0)
        & (ik < len(k_iMpc_edges) - 1)
        & (imu >= 0)
        & (imu < len(mu_edges) - 1)
    )
    return kk[valid], mu[valid], ik[valid], imu[valid]


def get_P3D_k_mu_modes_from_edges(
    k_iMpc_edges: ArrayLike,
    mu_edges: ArrayLike,
    Lbox_Mpc: float,
    max_discrete_modes: int = 256,
) -> tuple[dict, NDArray[Any]]:
    """
    Count the discrete Fourier modes in arbitrary ``(k, mu)`` cells of a box.

    Parameters
    ----------
    k_iMpc_edges : array_like
        Increasing radial bin edges in inverse Mpc, e.g. the linear edges of
        an external simulation measurement.
    mu_edges : array_like
        Increasing absolute direction-cosine edges in ``[0, 1]``.
    Lbox_Mpc : float
        Simulation-box side length in Mpc.
    max_discrete_modes : int, default=256
        Cells with at most this many modes also return their mode
        coordinates, as needed by
        :func:`forestflow.statistics.p3d.P3D_Mpc_k_mu_hybrid_averaged`.

    Returns
    -------
    k_mu_modes : dict of str to ndarray
        Mode wavenumbers and direction cosines of the populated cells with at
        most ``max_discrete_modes`` modes, named ``"i_j_k"`` and ``"i_j_mu"``.
    mode_counts : ndarray of shape (n_k, n_mu)
        Number of lattice modes in every cell, counting both members of
        conjugate pairs (the ``"full"`` convention of
        :func:`forestflow.model_fits.gaussian_p3d_relative_error`).

    Notes
    -----
    The lattice is ``2 pi / Lbox_Mpc`` times integer vectors and mu is
    measured with respect to the third axis. Modes are binned one ``k_z``
    plane at a time, so memory scales with the square of the number of modes
    per side rather than its cube; coordinates are kept only for the sparse
    cells.
    """
    k_iMpc_edges = np.asarray(k_iMpc_edges, dtype=float)
    mu_edges = np.asarray(mu_edges, dtype=float)
    if k_iMpc_edges.ndim != 1 or np.any(np.diff(k_iMpc_edges) <= 0):
        raise ValueError("k_iMpc_edges must be one-dimensional and increasing")
    if (mu_edges.ndim != 1 or np.any(np.diff(mu_edges) <= 0)
            or mu_edges[0] < 0 or mu_edges[-1] > 1):
        raise ValueError("mu_edges must be increasing and within [0, 1]")

    k_fun = 2 * np.pi / Lbox_Mpc
    n_k = len(k_iMpc_edges) - 1
    n_mu = len(mu_edges) - 1

    def plane_grid(nn: int) -> NDArray[Any]:
        n1d = np.arange(-nn, nn + 1) * k_fun
        kx, ky = np.meshgrid(n1d, n1d, indexing="ij")
        return kx**2 + ky**2

    # first pass: mode counts of every cell
    nn = int(k_iMpc_edges[-1] // k_fun) + 1
    kperp2 = plane_grid(nn)
    mode_counts = np.zeros((n_k, n_mu), dtype=int)
    for nz in range(-nn, nn + 1):
        _, _, ik, imu = _bin_lattice_plane(kperp2, nz * k_fun, k_iMpc_edges, mu_edges)
        np.add.at(mode_counts, (ik, imu), 1)

    # second pass: coordinates of the sparse cells only
    sparse = (mode_counts > 0) & (mode_counts <= max_discrete_modes)
    k_mu_modes = {}
    if not np.any(sparse):
        return k_mu_modes, mode_counts

    nn = int(k_iMpc_edges[np.nonzero(sparse)[0].max() + 1] // k_fun) + 1
    kperp2 = plane_grid(nn)
    cell_ids, k_list, mu_list = [], [], []
    for nz in range(-nn, nn + 1):
        kk, mu, ik, imu = _bin_lattice_plane(kperp2, nz * k_fun, k_iMpc_edges, mu_edges)
        keep = sparse[ik, imu]
        cell_ids.append(ik[keep] * n_mu + imu[keep])
        k_list.append(kk[keep])
        mu_list.append(mu[keep])
    cell_ids = np.concatenate(cell_ids)
    k_all = np.concatenate(k_list)
    mu_all = np.concatenate(mu_list)

    order = np.argsort(cell_ids, kind="stable")
    cell_ids, k_all, mu_all = cell_ids[order], k_all[order], mu_all[order]
    unique_ids, starts = np.unique(cell_ids, return_index=True)
    stops = np.append(starts[1:], len(cell_ids))
    for cell, start, stop in zip(unique_ids, starts, stops):
        flag = f"{cell // n_mu}_{cell % n_mu}"
        k_mu_modes[flag + "_k"] = k_all[start:stop]
        k_mu_modes[flag + "_mu"] = mu_all[start:stop]
        if stop - start != mode_counts[cell // n_mu, cell % n_mu]:
            raise RuntimeError("inconsistent lattice mode counts")

    return k_mu_modes, mode_counts


def p3d_allkmu(
    model: Callable[..., Any],
    zs: Any,
    arinyo: Any,
    kmu_modes: Any,
    nk: int | None = 14,
    nmu: int | None = 16,
    compute_plin: bool | None = True,
) -> NDArray[Any]:
    """
    Average model P3D and optionally linear power over discrete mode bins.

    Parameters
    ----------
    model : object
        Model exposing ``P3D_Mpc`` and a linear-power evaluator.
    zs : float or array_like
        Model redshift coordinate.
    arinyo : mapping
        Arinyo parameter mapping.
    kmu_modes : mapping
        Exact mode mapping returned by :func:`get_P3D_k_mu_modes`.
    nk, nmu : int, default: 14, 16
        Output radial and direction-cosine bin counts.
    compute_plin : bool, default: True
        Also return bin-averaged linear three-dimensional power.

    Returns
    -------
    ndarray or tuple of ndarray
        ``(n_k, n_mu)`` P3D in ``Mpc**3``, optionally paired with the
        corresponding bin-averaged linear power.
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
                    plin_allmodes = model.linear.get_linP_Mpc(zs, kev)
                    plin[ii, jj] = np.mean(plin_allmodes)
    if compute_plin:
        return p3d, plin
    else:
        return p3d
