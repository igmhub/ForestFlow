"""
Integrate three-dimensional power spectra into one-dimensional spectra.
"""

from collections.abc import Callable, Mapping
import hashlib
from typing import Any
from numpy.typing import ArrayLike, NDArray

import numpy as np
from scipy.integrate import simpson


def compute_px_from_p3d_kmu_Mpc(
    kp_Mpc: ArrayLike,
    rt_Mpc: ArrayLike,
    p3d_func_kmu_Mpc: Callable[..., Any],
    hankl_kt_Mpc_min: Any = 10.0**-7,
    hankl_kt_Mpc_max: Any = 10.0**3,
    hankl_nkt: int | None = 2**11,
    interp_rt_Mpc_min: Any = 0.005,
    interp_rt_Mpc_max: Any = 0.2,
    p3d_k_Mpc_max: float | None = 200,
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
    def dummy_p3d_func_kmu(
        dummy: Any,
        k: Any,
        mu: Any,
        ari_pp: Any | None = None,
        new_cosmo_params: Any | None = None,
    ) -> NDArray[Any]:
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
    """Vectorized P3D-to-P1D projection with cached quadrature geometry.

    The default is a 48-node Simpson rule in log(k_perp), selected from the
    convergence benchmark as the standard inference trade-off. Gauss--Legendre
    remains available for dedicated convergence studies.
    """

    def __init__(
        self,
        k_perp_min: float = 1e-3,
        k_perp_max: float = 100,
        n_k_perp: int = 48,
        method: str = "simpson",
        max_cached_geometries: int = 8,
    ) -> None:
        if method not in {"simpson", "gauss_legendre"}:
            raise ValueError("method must be 'simpson' or 'gauss_legendre'")
        minimum_nodes = 3 if method == "simpson" else 1
        if n_k_perp < minimum_nodes:
            raise ValueError(f"{method} requires at least {minimum_nodes} nodes")
        if not 0 < k_perp_min < k_perp_max:
            raise ValueError("require 0 < k_perp_min < k_perp_max")
        if max_cached_geometries < 0:
            raise ValueError("max_cached_geometries must be non-negative")

        self.method = method
        self.n_k_perp = n_k_perp
        self.max_cached_geometries = max_cached_geometries
        log_min, log_max = np.log(k_perp_min), np.log(k_perp_max)
        if method == "simpson":
            self.ln_k_perp = np.linspace(log_min, log_max, n_k_perp)
            self.weights = None
        else:
            nodes, weights = np.polynomial.legendre.leggauss(n_k_perp)
            midpoint, half_width = 0.5 * (log_min + log_max), 0.5 * (log_max - log_min)
            self.ln_k_perp = midpoint + half_width * nodes
            self.weights = half_width * weights
        self.dlnk = self.ln_k_perp[1] - self.ln_k_perp[0] if n_k_perp > 1 else 0.0
        self.k_perp = np.exp(self.ln_k_perp)
        self.k_perp3 = self.k_perp[None, None, :]
        self.prefactor = self.k_perp3**2 / (2 * np.pi)
        self._geometry_cache = {}

    def _geometry_key(self, k_par: np.ndarray):
        contiguous = np.ascontiguousarray(k_par)
        digest = hashlib.blake2b(contiguous.view(np.uint8), digest_size=16).digest()
        return contiguous.shape, contiguous.dtype.str, digest

    def _get_geometry(self, k_par: np.ndarray):
        """Return cached arrays of k and mu for this k_parallel grid."""
        key = self._geometry_key(k_par)
        cached = self._geometry_cache.get(key)
        if cached is not None:
            self._geometry_cache.pop(key)
            self._geometry_cache[key] = cached
            return cached
        k_par3 = k_par[..., None]
        k = np.hypot(k_par3, self.k_perp3)
        geometry = (k, k_par3 / k)
        if self.max_cached_geometries:
            if len(self._geometry_cache) >= self.max_cached_geometries:
                self._geometry_cache.pop(next(iter(self._geometry_cache)))
            self._geometry_cache[key] = geometry
        return geometry

    def clear_geometry_cache(self) -> None:
        """Discard cached k_parallel-derived geometry."""
        self._geometry_cache.clear()

    def __call__(
        self,
        linear: Any,
        z: ArrayLike,
        k_par: ArrayLike,
        p3d_fun: Callable[..., Any],
        p3d_params: Mapping[str, Any],
        coordinates: str = "k_mu",
        **kwargs: Any,
    ) -> Any:
        """Compute P1D for scalar, redshift-vector, or batched k grids."""
        z = np.asarray(z)
        k_par = np.asarray(k_par)
        if z.ndim == 0:
            if k_par.ndim == 1:
                k_par = k_par[None, :]
        elif k_par.ndim == 3:
            if k_par.shape[1] != len(z):
                raise ValueError("redshift axis of batched k_par must match len(z)")
        elif k_par.ndim == 1:
            k_par = np.broadcast_to(k_par, (len(z), len(k_par)))
        elif k_par.shape[0] != len(z):
            raise ValueError("leading dimension of k_par must match len(z)")

        k, mu = self._get_geometry(k_par)
        if coordinates == "k_mu":
            p3d = p3d_fun(linear, z, k, mu, p3d_params, **kwargs)
        elif coordinates == "kpar_kperp" and z.ndim == 0:
            kpar2d = np.broadcast_to(k_par[0, :, None], k.shape[1:])
            kperp2d = np.broadcast_to(self.k_perp[None, :], k.shape[1:])
            p3d = p3d_fun(linear, z, kpar2d, kperp2d, p3d_params, **kwargs)[None, ...]
        else:
            raise ValueError("kpar_kperp P3D callables require scalar z")
        integrand = p3d * self.prefactor
        if self.method == "simpson":
            return simpson(integrand, dx=self.dlnk, axis=-1)
        return np.sum(integrand * self.weights, axis=-1)


_P1D_integrator = P1DIntegrator()


def P1D_Mpc(
    linear: Any, zs: Any, k_par: Any, p3d_fun: ArrayLike, p3d_params: Mapping[str, Any], coordinates: str = "k_mu", **kwargs: Any
) -> NDArray[Any]:
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
    return _P1D_integrator(
        linear, zs, k_par, p3d_fun, p3d_params, coordinates=coordinates, **kwargs
    )
