"""
Estimate and smooth power-spectrum variances.
"""

import numpy as np


def get_nmod(k, dk, Lbox):
    """
    Calculate the number of modes in a given k bin.

    Parameters:
        k (float): Center of the k bin.
        dk (float): Width of the k bin.
        Lbox (float): Size of the simulation box.

    Returns:
        Nk (float): Number of modes in the k bin.
    """
    Vs = 4 * np.pi**2 * k**2 * dk * (1 + 1 / 12 * (dk / k) ** 2)
    kf = 2 * np.pi / Lbox
    Vk = kf**3
    Nk = Vs / Vk
    return Nk


def normalize_power(arch, arch_av):
    """
    Normalize power.

    Parameters
    ----------
    arch : object
        Arch used by the calculation.
    arch_av : object
        Arch av used by the calculation.

    Returns
    -------
    tuple
        Result produced when the function is used to normalize power.
    """
    nav = len(arch_av)
    nall = len(arch.data)
    norm_p1d = arch4cov[0]["k_Mpc"] / np.pi
    norm_p3d = arch4cov[0]["k3d_Mpc"] ** 3 / 2 / np.pi**2

    arr_norm_kp1d = np.zeros((nall, norm_p1d.shape[0]))
    arr_norm_k3p3d = np.zeros((nall, norm_p3d.shape[0], norm_p3d.shape[1]))

    for ii in range(nav):
        ind = np.argwhere(
            (arch.sim_label == arch_av[ii]["sim_label"])
            & (arch.ind_rescaling == arch_av[ii]["ind_rescaling"])
            & (arch.ind_snap == arch_av[ii]["ind_snap"])
        )[:, 0]
        for jj in ind:
            arr_norm_kp1d[jj] = (
                arch.data[jj]["p1d_Mpc"] - arch_av[ii]["p1d_Mpc"]
            ) * norm_p1d
            arr_norm_k3p3d[jj] = (
                arch.data[jj]["p3d_Mpc"] - arch_av[ii]["p3d_Mpc"]
            ) * norm_p3d
    return arr_norm_kp1d, arr_norm_k3p3d


def smooth_p1d_variance(k1d, std_p1d, kmax=10):
    """
    Smooth one-dimensional power spectrum variance.

    Parameters
    ----------
    k1d : object
        K1d used by the calculation.
    std_p1d : numpy.ndarray
        Std p1d used by the calculation.
    kmax : int, optional
        Kmax used by the calculation.

    Returns
    -------
    tuple
        Result produced when the function is used to smooth one-dimensional power spectrum variance.
    """
    sm_std_p1d = np.zeros_like(std_p1d)

    xx = k1d
    yy = np.log10(std_p1d)

    _ = np.argwhere(np.isfinite(xx) & np.isfinite(yy) & (k1d < kmax))[0:, 0]
    fit = np.polyfit(xx[_], yy[_], 1)

    sm_std_p1d = 10 ** (xx * fit[0] + fit[1])

    return k1d, sm_std_p1d


def smooth_p3d_variance(k3d, std_p3d, kmin=0.5, kmax=10):
    """
    Smooth three-dimensional power spectrum variance.

    Parameters
    ----------
    k3d : object
        K3d used by the calculation.
    std_p3d : numpy.ndarray
        Std p3d used by the calculation.
    kmin : float, optional
        Kmin used by the calculation.
    kmax : int, optional
        Kmax used by the calculation.

    Returns
    -------
    object
        Result produced when the function is used to smooth three-dimensional power spectrum variance.
    """
    sm_std_p3d = np.zeros_like(k3d)
    xx = np.log10(np.nanmean(k3d, axis=1))
    yy = np.log10(np.nanmean(std_p3d, axis=1))
    _ = np.argwhere(
        np.isfinite(xx)
        & np.isfinite(yy)
        & (xx < np.log10(kmax))
        & (xx > np.log10(kmin))
    )[:, 0]

    fit = np.polyfit(xx[_], yy[_], 1)

    sm_std_p3d[...] = (10 ** (xx * fit[0] + fit[1]))[:, np.newaxis]
    return sm_std_p3d
