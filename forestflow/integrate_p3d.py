"""
Integrate three-dimensional power spectra into one-dimensional spectra.
"""

from collections.abc import Callable, Mapping
from typing import Any
from numpy.typing import ArrayLike, NDArray

import numpy as np
from scipy.integrate import simpson


def compute_px_from_p3d_kmu_Mpc(
    kp_Mpc: ArrayLike,
    rt_Mpc: ArrayLike,
    p3d_func_kmu_Mpc: Callable[..., Any],
    hankl_kt_Mpc_min: Any=10.0**-7,
    hankl_kt_Mpc_max: Any=10.0**3,
    hankl_nkt: int | None=2**11,
    interp_rt_Mpc_min: Any=0.005,
    interp_rt_Mpc_max: Any=0.2,
    p3d_k_Mpc_max: float | None=200,
) -> NDArray[Any]:
    """
    Given P3D(k, mu) function, use Hankl to compute Px(rt, kp)

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

    Other Parameters
    ----------------
    hankl_kt_Mpc_min : object
        Hankl kt mpc min used by the calculation.
    hankl_kt_Mpc_max : object
        Hankl kt mpc max used by the calculation.
    interp_rt_Mpc_min : object
        Interp rt mpc min used by the calculation.
    interp_rt_Mpc_max : object
        Interp rt mpc max used by the calculation.
    """

    # ideally this function would be math only, but for now I'm recycling existing functions
    from forestflow.model_p3d_arinyo import coordinates
    from forestflow.pcross import Px_Mpc_detailed

    @coordinates("k_mu")
    def dummy_p3d_func_kmu(dummy: Any, k: Any, mu: Any, ari_pp: Any | None=None, new_cosmo_params: Any | None=None) -> NDArray[Any]:
        """
        Compute dummy three-dimensional power spectrum func kmu.

        Parameters
        ----------
        dummy : object
            Dummy used by the calculation.
        k : object
            K used by the calculation.
        mu : object
            Mu used by the calculation.
        ari_pp : object, optional
            Ari pp used by the calculation.
        new_cosmo_params : object, optional
            New cosmo params used by the calculation.

        Returns
        -------
        object
            Result produced when the function is used to compute dummy three-dimensional power spectrum func kmu.
        """
        return p3d_func_kmu_Mpc(k, mu)

    dummy_z = 123456789
    dummy_p3d_params = {"dummy": 123456789}
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
        max_k_for_p3d=p3d_k_Mpc_max,
    )

    return Px


class P1DIntegrator:
    """
    Fast computation of the one-dimensional power spectrum from a 3D model.

    The integration grid in k_perp is built once. The evaluation is fully
    vectorized over both redshift and k_parallel.

    Parameters
    ----------
    k_perp_min : float
        Minimum perpendicular wavenumber.
    k_perp_max : float
        Maximum perpendicular wavenumber.
    n_k_perp : int
        Number of logarithmically-spaced k_perp points.
    """

    def __init__(self, k_perp_min: float | None=1e-3, k_perp_max: int | None=100, n_k_perp: int | None=99) -> None:

        """
        Initialize the instance.

        Parameters
        ----------
        k_perp_min : float, optional
            K perp min used by the calculation.
        k_perp_max : int, optional
            K perp max used by the calculation.
        n_k_perp : int, optional
            N k perp used by the calculation.
        """
        self.ln_k_perp = np.linspace(np.log(k_perp_min), np.log(k_perp_max), n_k_perp)

        self.dlnk = self.ln_k_perp[1] - self.ln_k_perp[0]

        self.k_perp = np.exp(self.ln_k_perp)

        # shape (1,1,Nkperp)
        self.k_perp3 = self.k_perp[None, None, :]

        # shape (1,1,Nkperp)
        self.prefactor = self.k_perp3**2 / (2 * np.pi)

    def __call__(self, linear: Any, z: ArrayLike, k_par: ArrayLike, p3d_fun: ArrayLike, p3d_params: Mapping[str, Any]) -> Any:
        """
        Compute P1D.

        Parameters
        ----------
        z : (Nz,) array_like or float

        k_par : (Nz,Nk) array_like

        Returns
        -------
        p1d : (Nz,Nk) ndarray

        Other Parameters
        ----------------
        linear : object
            Precomputed linear-theory grid.
        p3d_fun : numpy.ndarray
            Callable three-dimensional power-spectrum model.
        p3d_params : dict
            Parameters passed to the three-dimensional model.
        """

        z = np.asarray(z)
        k_par = np.asarray(k_par)

        if z.ndim == 0:
            # One redshift
            if k_par.ndim == 1:
                k_par = k_par[None, :]
        else:
            # Multiple redshifts

            if k_par.ndim == 1:
                # Same k_parallel grid for every redshift
                k_par = np.broadcast_to(
                    k_par,
                    (len(z), len(k_par)),
                )

            elif k_par.shape[0] != len(z):
                raise ValueError("Leading dimension of k_par must match len(z).")

        # (Nz,Nk,1)
        k_par3 = k_par[..., None]

        k = np.hypot(
            k_par3,
            self.k_perp3,
        )

        mu = k_par3 / k

        p3d = p3d_fun(linear, z, k, mu, p3d_params)

        integrate = simpson(p3d * self.prefactor, dx=self.dlnk, axis=-1)

        return integrate


_P1D_integrator = P1DIntegrator()


def P1D_Mpc(linear: Any, zs: Any, k_par: Any, p3d_fun: ArrayLike, p3d_params: Mapping[str, Any]) -> NDArray[Any]:

    """
    Compute one-dimensional power spectrum Mpc.

    Parameters
    ----------
    linear : object
        Linear used by the calculation.
    zs : object
        Zs used by the calculation.
    k_par : object
        K par used by the calculation.
    p3d_fun : numpy.ndarray
        P3d fun used by the calculation.
    p3d_params : dict
        P3d params used by the calculation.

    Returns
    -------
    object
        Result produced when the function is used to compute one-dimensional power spectrum mpc.
    """
    return _P1D_integrator(linear, zs, k_par, p3d_fun, p3d_params)
