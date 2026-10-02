"""P1D projections and bin averaging.

A P1D projection always integrates a P3D callable over transverse wavenumber.
Consequently, the sole P3D callable contract in this module is
``p3d_kpar_kperp(linear, z, k_par_iMpc, k_perp_iMpc, parameters)``.
"""

import hashlib

import numpy as np
from scipy.integrate import simpson

from forestflow.conventions import validate_wavenumber
from forestflow.statistics.binning import logarithmic_bin_edges


class P1DIntegrator:
    """
    Project Cartesian P3D to P1D with cached integration geometry.

    The P3D callable must have the signature
    ``p3d_kpar_kperp(linear, z, k_par_iMpc, k_perp_iMpc, parameters)``.
    The default is Simpson integration with 48 log-spaced transverse nodes.
    Use a separately constructed instance to choose a different quadrature
    geometry, for example for a convergence study.
    """

    def __init__(
        self,
        k_perp_min_iMpc=1e-3,
        k_perp_max_iMpc=100.0,
        n_k_perp=48,
        method="simpson",
        max_cached_geometries=8,
    ):
        """Configure a reusable transverse P3D-to-P1D quadrature.

        Parameters
        ----------
        k_perp_min_iMpc, k_perp_max_iMpc : float, default=1e-3, 100.0
            Inclusive transverse comoving-wavenumber range in 1/Mpc.
        n_k_perp : int, default=48
            Number of log-spaced quadrature nodes. Simpson integration needs
            at least three nodes; Gauss--Legendre needs at least one.
        method : {"simpson", "gauss_legendre"}, default="simpson"
            Quadrature in ``ln(k_perp)``.
        max_cached_geometries : int, default=8
            Number of recently used parallel-wavenumber grids retained in the
            in-memory geometry cache. Zero disables caching.

        Notes
        -----
        The projection is ``integral dln(k_perp) k_perp**2 P3D/(2*pi)``.
        """
        if method not in {"simpson", "gauss_legendre"}:
            raise ValueError("method must be 'simpson' or 'gauss_legendre'")
        minimum_nodes = 3 if method == "simpson" else 1
        if not isinstance(n_k_perp, (int, np.integer)) or n_k_perp < minimum_nodes:
            raise ValueError(f"{method} requires at least {minimum_nodes} nodes")
        if not 0 < k_perp_min_iMpc < k_perp_max_iMpc:
            raise ValueError("require 0 < k_perp_min_iMpc < k_perp_max_iMpc")
        if max_cached_geometries < 0:
            raise ValueError("max_cached_geometries must be non-negative")
        self.method = method
        self.n_k_perp = int(n_k_perp)
        self.max_cached_geometries = max_cached_geometries
        log_min = np.log(k_perp_min_iMpc)
        log_max = np.log(k_perp_max_iMpc)
        if method == "simpson":
            self.ln_k_perp = np.linspace(log_min, log_max, self.n_k_perp)
            self.weights = None
        else:
            nodes, weights = np.polynomial.legendre.leggauss(self.n_k_perp)
            midpoint = 0.5 * (log_min + log_max)
            half_width = 0.5 * (log_max - log_min)
            self.ln_k_perp = midpoint + half_width * nodes
            self.weights = half_width * weights
        self.k_perp_iMpc = np.exp(self.ln_k_perp)
        self._geometry_cache = {}

    def _geometry_key(self, k_par_iMpc):
        contiguous = np.ascontiguousarray(k_par_iMpc)
        digest = hashlib.blake2b(contiguous.view(np.uint8), digest_size=16).digest()
        return contiguous.shape, contiguous.dtype.str, digest

    def geometry(self, k_par_iMpc):
        """Return cached grids of ``k_parallel`` and ``k_perp`` in inverse Mpc."""
        key = self._geometry_key(k_par_iMpc)
        cached = self._geometry_cache.get(key)
        if cached is not None:
            self._geometry_cache.pop(key)
            self._geometry_cache[key] = cached
            return cached
        k_par = np.broadcast_to(
            k_par_iMpc[..., None], k_par_iMpc.shape + (self.n_k_perp,)
        )
        k_perp = np.broadcast_to(
            self.k_perp_iMpc,
            k_par_iMpc.shape + (self.n_k_perp,),
        )
        geometry = k_par, k_perp
        if self.max_cached_geometries:
            if len(self._geometry_cache) >= self.max_cached_geometries:
                self._geometry_cache.pop(next(iter(self._geometry_cache)))
            self._geometry_cache[key] = geometry
        return geometry

    def clear_geometry_cache(self):
        """Discard cached k-parallel/k-perpendicular grids."""
        self._geometry_cache.clear()

    def integrate(self, p3d_Mpc):
        """Integrate P3D values whose final axis is this integrator's k_perp grid."""
        p3d_Mpc = np.asarray(p3d_Mpc)
        if p3d_Mpc.shape[-1] != self.n_k_perp:
            raise ValueError("last P3D axis must match integrator.n_k_perp")
        integrand = p3d_Mpc * self.k_perp_iMpc**2 / (2 * np.pi)
        if self.method == "simpson":
            return simpson(integrand, x=self.ln_k_perp, axis=-1)
        return np.sum(integrand * self.weights, axis=-1)

    def __call__(
        self, linear, z, k_par_iMpc, p3d_kpar_kperp, p3d_params=None, **kwargs
    ):
        """Evaluate a Cartesian P3D callable and project it into P1D in Mpc."""
        z = np.asarray(z)
        k_par_iMpc = np.asarray(k_par_iMpc, dtype=float)
        if (
            k_par_iMpc.ndim == 0
            or not np.all(np.isfinite(k_par_iMpc))
            or np.any(k_par_iMpc < 0)
        ):
            raise ValueError("k_par_iMpc must be a finite, non-negative array")
        if z.ndim == 0 and k_par_iMpc.ndim == 1:
            k_par_iMpc = k_par_iMpc[None, :]
        elif z.ndim > 0 and k_par_iMpc.ndim == 1:
            k_par_iMpc = np.broadcast_to(k_par_iMpc, (len(z), len(k_par_iMpc)))
        elif z.ndim > 0 and k_par_iMpc.ndim >= 2 and k_par_iMpc.shape[-2] != len(z):
            raise ValueError("redshift axis of k_par_iMpc must match len(z)")
        k_par, k_perp = self.geometry(k_par_iMpc)
        p3d = p3d_kpar_kperp(
            linear, z, k_par, k_perp, {} if p3d_params is None else p3d_params, **kwargs
        )
        return self.integrate(p3d)


_DEFAULT_P1D_INTEGRATOR = P1DIntegrator()


def P1D_Mpc(
    linear, z, k_par_iMpc, p3d_kpar_kperp, p3d_params=None, integrator=None, **kwargs
):
    """
    Project Cartesian P3D into P1D in Mpc units.

    ``p3d_kpar_kperp`` must accept ``(linear, z, k_par_iMpc, k_perp_iMpc,
    parameters)``. Pass a :class:`P1DIntegrator` for a non-default quadrature.
    """
    if integrator is None:
        integrator = _DEFAULT_P1D_INTEGRATOR
    result = integrator(linear, z, k_par_iMpc, p3d_kpar_kperp, p3d_params, **kwargs)
    return result[0] if np.ndim(z) == 0 else result


def P1D_kms(
    linear,
    z,
    k_par_ikms,
    p3d_kpar_kperp,
    dkms_diMpc,
    p3d_params=None,
    integrator=None,
    **kwargs,
):
    """
    Project Cartesian Mpc-space P3D into P1D in velocity units.

    ``p3d_kpar_kperp`` must accept ``(linear, z, k_par_iMpc, k_perp_iMpc,
    parameters)``. The same Mpc-space integrator is used before applying the one-power P1D
    Jacobian. Pass an explicit integrator to change the transverse range.
    """
    k_par_ikms = np.asarray(k_par_ikms, dtype=float)
    if (
        k_par_ikms.ndim != 1
        or not np.all(np.isfinite(k_par_ikms))
        or np.any(k_par_ikms < 0)
    ):
        raise ValueError("k_par_ikms must be a finite, non-negative 1D array")
    if not np.isfinite(dkms_diMpc) or dkms_diMpc <= 0:
        raise ValueError("dkms_diMpc must be finite and positive")
    return dkms_diMpc * P1D_Mpc(
        linear,
        z,
        k_par_ikms * dkms_diMpc,
        p3d_kpar_kperp,
        p3d_params,
        integrator,
        **kwargs,
    )


def P1D_Mpc_bin_averaged(
    linear,
    z,
    k_par_iMpc,
    p1d_fun,
    p1d_params=None,
    k_par_edges=None,
    fine_factor=8,
    **kwargs,
):
    """
    Evaluate a P1D function on fine logarithmic bins and average them.

    Unlike :func:`P1D_Mpc`, ``p1d_fun`` is already a one-dimensional callable
    with signature ``p1d_fun(linear, z, k_par_iMpc, parameters)``.
    """
    if not isinstance(fine_factor, int) or fine_factor < 1:
        raise ValueError("fine_factor must be a positive integer")
    k_par_iMpc = validate_wavenumber(k_par_iMpc, name="k_par_iMpc")
    edges = logarithmic_bin_edges(k_par_iMpc, k_par_edges, name="k_par")
    fractions = (np.arange(fine_factor) + 0.5) / fine_factor
    fine_k = np.exp(
        np.log(edges[:-1, None]) + fractions * np.diff(np.log(edges))[:, None]
    )
    prediction = np.asarray(p1d_fun(linear, z, fine_k.ravel(), p1d_params, **kwargs))
    return prediction.reshape(
        prediction.shape[:-1] + (len(k_par_iMpc), fine_factor)
    ).mean(axis=-1)


def p1d_from_p3d(
    linear,
    k_par_iMpc,
    p3d_kpar_kperp,
    z,
    p3d_params=None,
    volume_Mpc3=None,
    n_realizations=1000,
    seed=0,
    integrator=None,
):
    """
    Project Cartesian P3D and optionally draw finite-volume Gaussian realizations.

    The deterministic and realization projections use the same
    :class:`P1DIntegrator` object and therefore exactly the same quadrature.
    """
    k_par_iMpc = np.asarray(k_par_iMpc, dtype=float)
    if (
        k_par_iMpc.ndim != 1
        or not np.all(np.isfinite(k_par_iMpc))
        or np.any(k_par_iMpc < 0)
    ):
        raise ValueError("k_par_iMpc must be a finite, non-negative 1D array")
    if integrator is None:
        integrator = _DEFAULT_P1D_INTEGRATOR
    kpar2d, kperp2d = integrator.geometry(k_par_iMpc)
    p3d = p3d_kpar_kperp(
        linear, z, kpar2d, kperp2d, {} if p3d_params is None else p3d_params
    )
    p1d = integrator.integrate(p3d)
    result = {
        "kpar_iMpc": kpar2d,
        "kperp_iMpc": kperp2d,
        "P3D_Mpc": p3d,
        "P1D_Mpc": p1d,
    }
    if volume_Mpc3 is not None:
        if volume_Mpc3 <= 0:
            raise ValueError("volume_Mpc3 must be positive")
        dkpar = np.gradient(kpar2d, axis=0)
        dkperp = np.gradient(kperp2d, axis=1)
        n_modes = volume_Mpc3 / (2 * np.pi) ** 2 * kperp2d * dkperp * dkpar
        sigma = np.sqrt(2.0 / n_modes) * p3d
        rng = np.random.default_rng(seed)
        p3d_realizations = p3d + rng.normal(
            scale=sigma, size=(n_realizations,) + p3d.shape
        )
        result["P3D_Mpc_realizations"] = p3d_realizations
        result["P1D_Mpc_realizations"] = integrator.integrate(p3d_realizations)
    return result
