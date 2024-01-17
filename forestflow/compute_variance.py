import numpy as np


def normalize_power(arch, arch_av):
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
    sm_std_p1d = np.zeros_like(std_p1d)

    xx = k1d
    yy = np.log10(std_p1d)

    _ = np.argwhere(np.isfinite(xx) & np.isfinite(yy) & (k1d < kmax))[0:, 0]
    fit = np.polyfit(xx[_], yy[_], 1)

    sm_std_p1d = 10 ** (xx * fit[0] + fit[1])

    return k1d, sm_std_p1d


def smooth_p3d_variance(k3d, std_p3d, kmin=0.5, kmax=10):
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
