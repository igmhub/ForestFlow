"""
P1D projections and bin averaging.

A P1D projection integrates P3D over transverse wavenumber. Two equivalent
quadratures are provided, each with its own P3D callable contract:

* :class:`P1DIntegrator` integrates over ``ln(k_perp)`` and calls
  ``p3d_kpar_kperp(linear, z, k_par_iMpc, k_perp_iMpc, parameters)``.
* :class:`P1DIntegratorKMu` uses ``k_perp dk_perp = k dk`` to integrate over
  ``ln(k)`` at fixed ``k_par`` and calls
  ``p3d_k_mu(linear, z, k_iMpc, mu, parameters)``.
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
    The default is Simpson integration with 99 log-spaced transverse nodes.
    Use a separately constructed instance to choose a different quadrature
    geometry, for example for a convergence study.
    """

    def __init__(
        self,
        k_perp_min_iMpc=1e-3,
        k_perp_max_iMpc=100.0,
        n_k_perp=99,
        method="simpson",
        max_cached_geometries=8,
    ):
        """
        Configure a reusable transverse P3D-to-P1D quadrature.

        Parameters
        ----------
        k_perp_min_iMpc, k_perp_max_iMpc : float, default=1e-3, 100.0
            Inclusive transverse comoving-wavenumber range in 1/Mpc.
        n_k_perp : int, default=99
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
            # Simpson integration is linear. Integrating an identity matrix
            # once gives the exact weights used by scipy for this grid,
            # including its even-node endpoint convention.
            self.weights = simpson(
                np.eye(self.n_k_perp), x=self.ln_k_perp, axis=0
            )
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
        """
        Return cached grids of ``k_parallel`` and ``k_perp`` in inverse Mpc.
        """
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
        """
        Discard cached k-parallel/k-perpendicular grids.
        """
        self._geometry_cache.clear()

    def integrate(self, p3d_Mpc):
        """
        Integrate P3D values whose final axis is this integrator's k_perp grid.
        """
        p3d_Mpc = np.asarray(p3d_Mpc)
        if p3d_Mpc.shape[-1] != self.n_k_perp:
            raise ValueError("last P3D axis must match integrator.n_k_perp")
        integrand = p3d_Mpc * self.k_perp_iMpc**2 / (2 * np.pi)
        return np.sum(integrand * self.weights, axis=-1)

    def __call__(
        self, linear, z, k_par_iMpc, p3d_kpar_kperp, p3d_params=None, **kwargs
    ):
        """
        Evaluate a Cartesian P3D callable and project it into P1D in Mpc.
        """
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


def _unit_gauss_legendre(n_nodes):
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    return 0.5 * (nodes + 1.0), 0.5 * weights


class P1DIntegratorKMu:
    """
    Project P3D(k, mu) to P1D with split Gauss--Legendre quadrature in ln k.

    Changing variables with ``k_perp dk_perp = k dk`` gives

    ``P1D(k_par) = 1/(2 pi) int_{k_lo}^{k_hi} k**2 P3D(k, mu=k_par/k) dln k``,

    with ``k_lo, k_hi = sqrt(k_par**2 + k_perp_{min,max}**2)``, so the domain
    is identical to that of :class:`P1DIntegrator`. Unlike the ``ln(k_perp)``
    grid, no nodes are spent at ``k_perp << k_par``, where the integrand is
    negligible.

    For small ``k_par`` the integrand peaks at ``k ~ k_par`` and crosses the
    BAO features of the linear power, which limit the accuracy of any single
    smooth rule over the whole interval. Rows with ``k_lo < k_split_iMpc``
    therefore use ``n_low`` nodes on ``[k_lo, k_split]`` and ``n_high`` nodes
    on ``[k_split, k_hi]``; the smooth high-k tail then converges quickly.
    Rows with ``k_lo >= k_split_iMpc`` use all ``n_low + n_high`` nodes on a
    single interval. See ``notebooks/emulator/convergence_p1d_quadrature``.
    """

    def __init__(
        self,
        k_perp_min_iMpc=1e-3,
        k_perp_max_iMpc=100.0,
        k_split_iMpc=0.4,
        n_low=20,
        n_high=20,
        max_cached_geometries=8,
    ):
        """
        Configure the split Gauss--Legendre ``ln k`` quadrature.

        Parameters
        ----------
        k_perp_min_iMpc, k_perp_max_iMpc : float, default=1e-3, 100.0
            Transverse comoving-wavenumber range in 1/Mpc defining the domain.
        k_split_iMpc : float or None, default=0.4
            Break between the low-k (BAO) and high-k segments in 1/Mpc.
            ``None`` uses a single Gauss--Legendre rule with
            ``n_low + n_high`` nodes for every row (no split).
        n_low, n_high : int, default=20
            Gauss--Legendre nodes below and above ``k_split_iMpc``. With
            these defaults the maximum relative P1D error is ~5e-6 for the
            MP-Gadget Arinyo fits, and stays below 1e-5 for splits in
            [0.35, 0.5]/Mpc. Fewer than ~16 high-k nodes, or a split below
            ~0.3/Mpc (inside the BAO range), degrades it to ~1e-4.
        max_cached_geometries : int, default=8
            Number of recently used parallel-wavenumber grids retained in the
            in-memory geometry cache. Zero disables caching.
        """
        for name, value in {"n_low": n_low, "n_high": n_high}.items():
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if k_split_iMpc is None:
            if not 0 < k_perp_min_iMpc < k_perp_max_iMpc:
                raise ValueError("require 0 < k_perp_min_iMpc < k_perp_max_iMpc")
        elif not 0 < k_perp_min_iMpc < k_split_iMpc < k_perp_max_iMpc:
            raise ValueError(
                "require 0 < k_perp_min_iMpc < k_split_iMpc < k_perp_max_iMpc"
            )
        if max_cached_geometries < 0:
            raise ValueError("max_cached_geometries must be non-negative")
        self.k_perp_min_iMpc = float(k_perp_min_iMpc)
        self.k_perp_max_iMpc = float(k_perp_max_iMpc)
        self.k_split_iMpc = None if k_split_iMpc is None else float(k_split_iMpc)
        self.n_low = int(n_low)
        self.n_high = int(n_high)
        self.n_nodes = self.n_low + self.n_high
        self.max_cached_geometries = max_cached_geometries
        self._rule_low = _unit_gauss_legendre(self.n_low)
        self._rule_high = _unit_gauss_legendre(self.n_high)
        self._rule_all = _unit_gauss_legendre(self.n_nodes)
        self._geometry_cache = {}

    _geometry_key = P1DIntegrator._geometry_key

    def geometry(self, k_par_iMpc):
        """
        Return cached ``k`` (1/Mpc), ``mu`` and ``ln k`` weights.

        All arrays have shape ``k_par_iMpc.shape + (n_low + n_high,)``.
        """
        key = self._geometry_key(k_par_iMpc)
        cached = self._geometry_cache.get(key)
        if cached is not None:
            self._geometry_cache.pop(key)
            self._geometry_cache[key] = cached
            return cached
        k_par = np.asarray(k_par_iMpc, dtype=float)[..., None]
        ln_lo = 0.5 * np.log(k_par**2 + self.k_perp_min_iMpc**2)
        ln_hi = 0.5 * np.log(k_par**2 + self.k_perp_max_iMpc**2)
        # Without a split every row uses the single-interval rule.
        ln_split = -np.inf if self.k_split_iMpc is None else np.log(self.k_split_iMpc)
        (x_low, w_low), (x_high, w_high) = self._rule_low, self._rule_high
        x_all, w_all = self._rule_all
        ln_k = np.where(
            ln_lo < ln_split,
            np.concatenate(
                [
                    np.broadcast_to(
                        ln_lo + (ln_split - ln_lo) * x_low,
                        ln_lo.shape[:-1] + (self.n_low,),
                    ),
                    np.broadcast_to(
                        ln_split + (ln_hi - ln_split) * x_high,
                        ln_hi.shape[:-1] + (self.n_high,),
                    ),
                ],
                axis=-1,
            ),
            ln_lo + (ln_hi - ln_lo) * x_all,
        )
        weights = np.where(
            ln_lo < ln_split,
            np.concatenate(
                [
                    np.broadcast_to(
                        (ln_split - ln_lo) * w_low,
                        ln_lo.shape[:-1] + (self.n_low,),
                    ),
                    np.broadcast_to(
                        (ln_hi - ln_split) * w_high,
                        ln_hi.shape[:-1] + (self.n_high,),
                    ),
                ],
                axis=-1,
            ),
            (ln_hi - ln_lo) * w_all,
        )
        k = np.exp(ln_k)
        # k >= k_par analytically; clip round-off so that mu stays in [0, 1].
        mu = np.minimum(k_par / k, 1.0)
        geometry = k, mu, weights
        if self.max_cached_geometries:
            if len(self._geometry_cache) >= self.max_cached_geometries:
                self._geometry_cache.pop(next(iter(self._geometry_cache)))
            self._geometry_cache[key] = geometry
        return geometry

    def clear_geometry_cache(self):
        """
        Discard cached ``k``/``mu`` grids.
        """
        self._geometry_cache.clear()

    @staticmethod
    def integrate(p3d_Mpc, k_iMpc, weights):
        """
        Integrate P3D values sampled on a :meth:`geometry` grid.
        """
        integrand = np.asarray(p3d_Mpc) * k_iMpc**2 / (2 * np.pi)
        return np.sum(integrand * weights, axis=-1)

    def __call__(self, linear, z, k_par_iMpc, p3d_k_mu, p3d_params=None, **kwargs):
        """
        Evaluate a ``P3D(k, mu)`` callable and project it into P1D in Mpc.
        """
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
        k, mu, weights = self.geometry(k_par_iMpc)
        p3d = p3d_k_mu(
            linear, z, k, mu, {} if p3d_params is None else p3d_params, **kwargs
        )
        return self.integrate(p3d, k, weights)


_DEFAULT_P1D_K_MU_INTEGRATOR = P1DIntegratorKMu()


def P1D_Mpc(
    linear, z, k_par_iMpc, p3d_kpar_kperp, p3d_params=None, integrator=None, **kwargs
):
    """
    Project Cartesian P3D into P1D in Mpc units.

    ``p3d_kpar_kperp`` must accept ``(linear, z, k_par_iMpc, k_perp_iMpc,
    parameters)``. Pass a :class:`P1DIntegrator` for a non-default quadrature,
    or a :class:`P1DIntegratorKMu` together with a ``p3d_k_mu(linear, z,
    k_iMpc, mu, parameters)`` callable in place of ``p3d_kpar_kperp``.
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
