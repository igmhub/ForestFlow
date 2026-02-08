import numpy as np


def p3d_rebin_mu(k3d, mu, p3d, kmu_modes, n_mubins=4, return_modes=False):
    """Rebin p3d to fewer mu bins"""

    def wmean(data, weight):
        """Weighted mean"""
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


def get_p3d_modes(kmax, lbox=67.5, k_Mpc_max=20, n_k_bins=20, n_mu_bins=16):
    """Get k and mu of p3d modes"""

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
    lbox=67.5, kpar_Mpc_max=20, n_k_bins=20, n_mu_bins=16
):
    """Get k and mu of p3d modes"""

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
    model,
    zs,
    arinyo,
    kmu_modes,
    nk=14,
    nmu=16,
    compute_plin=True,
):
    """Get p3d and plin for all k-mu bins"""
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
