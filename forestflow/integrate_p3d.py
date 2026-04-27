import numpy as np


def compute_px_from_p3d_kmu_Mpc(kp_Mpc, rt_Mpc, p3d_func_kmu_Mpc,
                        hankl_kt_Mpc_min=10.0**-7,
                        hankl_kt_Mpc_max=10.0**3,
                        hankl_nkt=2**11,
                        interp_rt_Mpc_min=0.005,
                        interp_rt_Mpc_max=0.2,
                        p3d_k_Mpc_max=200):
    """Given P3D(k, mu) function, use Hankl to compute Px(rt, kp)

    This is the user-friendly interface to `Px_Mpc_detailed`, used in cupix.

    Parameters
    ----------
    kp_Mpc : array-like
        Parallel wavenumbers k_parallel in units of Mpc⁻¹.
    rt_Mpc : array-like
        Transverse separations r_perp (in Mpc) at which to evaluate the cross-power spectrum.
    p3d_func_kmu_Mpc : callable
        Function returning P3D(k, mu) in Mpc units.
    hankl_kt_Mpc_{min, max} : float, optional
        Minimum and maximum k_perp (Mpc⁻¹) used for the Hankel transform. Default: 1e-7, 1e3.
    hankl_nkt : int, optional
        Number of k_perp points for the Hankel transform. Controls the output r_perp sampling.
        Default is 2**11 (~2048).
    interp_rt_Mpc_{min, max} : float, optional
        r_perp range (in Mpc) over which to smoothly interpolate between the Px and P1D
        to avoid divergences. Default: 0.005–0.2 Mpc.
    p3d_k_Mpc_max : float, optional
        maximum wavenumber for which we trust the P3D function (use zero past that)

    Returns
    -------
    Px : ndarray, shape [Nr, Nk]
        Cross-power spectrum P_cross in Mpc units evaluated at each input r_perp and k_parallel.

    """

    # ideally this function would be math only, but for now I'm recycling existing functions
    from forestflow.model_p3d_arinyo import coordinates
    from forestflow.pcross import Px_Mpc_detailed

    @coordinates("k_mu")
    def dummy_p3d_func_kmu(dummy, k, mu, ari_pp=None, new_cosmo_params=None):
        return p3d_func_kmu_Mpc(k, mu)

    dummy_z = 123456789
    dummy_p3d_params = {'dummy': 123456789}
    Px = Px_Mpc_detailed(
        z=dummy_z,
        kpar_iMpc=kp_Mpc,
        rperp_Mpc=rt_Mpc,
        p3d_fun_Mpc=dummy_p3d_func_kmu,
        min_kperp=hankl_kt_Mpc_min,
        max_kperp=hankl_kt_Mpc_max,
        nkperp=hankl_nkt,
        interpmin=interp_rt_Mpc_min,
        interpmax=interp_rt_Mpc_max,
        p3d_params=dummy_p3d_params,
        max_k_for_p3d=p3d_k_Mpc_max)

    return Px

