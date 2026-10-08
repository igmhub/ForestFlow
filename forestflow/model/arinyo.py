"""
Evaluate the Arinyo Lyman-alpha forest flux-power model.
"""

from collections.abc import Mapping
from typing import Any
from numpy.typing import ArrayLike, NDArray
import numpy as np
from forestflow.statistics import px as pcross
from forestflow.conventions import validate_finite_array, validate_mu

from forestflow.statistics.p1d import P1D_Mpc as compute_P1D
from forestflow.utils import broadcast_leading_dimensions


class ArinyoModel:
    """
    Evaluate the Arinyo flux-power model from LaCE linear theory.

    Model coefficients follow ``ARINYO_PARAMETER_NAMES``. In particular, the
    large-scale redshift-space factor is ``(bias + bias_eta*f*mu**2)**2``;
    ``bias_eta`` is not interchangeable with beta.
    """

    def __init__(
        self,
        fiducial_cosmology: Any | None = None,
        default_bias: float | None = -0.18,
        default_bias_eta: float | None = -0.23,
        default_q1: float | None = 0.4,
        default_q2: float | None = 0.0,
        default_kvav: float | None = 0.58,
        default_av: float | None = 0.29,
        default_bv: float | None = 1.55,
        default_kp: float | None = 10.5,
    ) -> None:
        from forestflow.model.linear import LinearTheory

        self.linear = LinearTheory(fiducial_cosmology)
        self.default_params = {
            "bias": default_bias,
            "bias_eta": default_bias_eta,
            "q1": default_q1,
            "q2": default_q2,
            "kvav": default_kvav,
            "av": default_av,
            "bv": default_bv,
            "kp": default_kp,
        }

    def P3D_Mpc_kpar_kperp(
        self,
        linear: Any,
        z: float,
        k_par_iMpc: ArrayLike,
        k_perp_iMpc: ArrayLike,
        ari_pp: Mapping[str, Any],
    ) -> NDArray[Any]:
        """
        Evaluate Cartesian Arinyo P3D in comoving units.

        Parameters
        ----------
        linear : forestflow.model.linear.LinearTheoryGrid
            Linear-theory grid supplying the evolved ``bc`` power and growth
            rate at ``z``.
        z : float or array_like
            Redshift scalar or redshift axis accepted by ``linear``.
        k_par_iMpc, k_perp_iMpc : array_like
            Non-negative parallel and transverse comoving wavenumbers in
            1/Mpc. They are broadcast to the output shape.
        ari_pp : mapping
            Arinyo coefficients; omitted coefficient names use model defaults.

        Returns
        -------
        numpy.ndarray
            Flux P3D in Mpc^3, with broadcast k, redshift, and supported
            leading batch axes.

        Raises
        ------
        ValueError
            If a wavenumber is non-finite, negative, or both components are
            zero.
        """

        k_par_iMpc = validate_finite_array(
            k_par_iMpc, "k_par_iMpc", minimum=0.0
        )
        k_perp_iMpc = validate_finite_array(k_perp_iMpc, "k_perp_iMpc", minimum=0.0)
        k_iMpc = np.hypot(k_par_iMpc, k_perp_iMpc)
        if np.any(k_iMpc == 0):
            raise ValueError("k_iMpc must be positive; k=0 is outside the linear-theory grid")
        mu = k_par_iMpc / k_iMpc
        return self.P3D_Mpc_k_mu(linear, z, k_iMpc, mu, ari_pp)

    def _arinyo_kernel(self, linP_Mpc, fz, k_iMpc, mu, ari_pp):
        """
        Evaluate the Arinyo nonlinear flux-power kernel.
        """
        bias = broadcast_leading_dimensions(ari_pp["bias"], k_iMpc)
        bias_eta = broadcast_leading_dimensions(ari_pp["bias_eta"], k_iMpc)
        lowk_bias = bias + bias_eta * fz * mu**2
        return linP_Mpc * lowk_bias**2 * self.nonlinear_correction(
            linP_Mpc, k_iMpc, mu, ari_pp
        )

    @staticmethod
    def nonlinear_correction(linP_Mpc, k_iMpc, mu, ari_pp):
        """Return the dimensionless Arinyo nonlinear correction ``D_NL``.

        Parameters
        ----------
        linP_Mpc : array_like
            Linear matter power spectrum in Mpc^3.
        k_iMpc : array_like
            Comoving wavenumber in 1/Mpc.
        mu : array_like
            Absolute line-of-sight cosine.
        ari_pp : mapping
            Named ``q1``, ``q2``, ``kvav``, ``av``, ``bv``, and ``kp``
            coefficients. Values may have leading batch dimensions.

        Returns
        -------
        numpy.ndarray
            ``exp[Delta2 * (q1 + q2 * Delta2) *
            (1 - k**av / kvav * abs(mu)**bv) - (k / kp)**2]``.

        Notes
        -----
        This bias-free factor is public so correlation-function consumers can
        apply it exactly once while retaining their own tracer Kaiser terms.
        ``k`` is in 1/Mpc and ``linP_Mpc`` is in Mpc^3.
        """
        q1 = broadcast_leading_dimensions(ari_pp["q1"], k_iMpc)
        q2 = broadcast_leading_dimensions(ari_pp["q2"], k_iMpc)
        av = broadcast_leading_dimensions(ari_pp["av"], k_iMpc)
        kvav = broadcast_leading_dimensions(ari_pp["kvav"], k_iMpc)
        bv = broadcast_leading_dimensions(ari_pp["bv"], k_iMpc)
        kp = broadcast_leading_dimensions(ari_pp["kp"], k_iMpc)
        delta2 = k_iMpc**3 * linP_Mpc / (2 * np.pi**2)
        nonlin = delta2 * (q1 + q2 * delta2)
        velocity = k_iMpc**av / kvav * np.abs(mu)**bv
        pressure = (k_iMpc / kp) ** 2
        return np.exp(nonlin * (1 - velocity) - pressure)

    def P3D_Mpc_k_mu(
        self,
        linear: Any,
        z: float,
        k_iMpc: ArrayLike,
        mu: float,
        ari_pp: Mapping[str, Any],
    ) -> float:
        """
        Evaluate Arinyo P3D on wavenumber--angle coordinates.

        Parameters
        ----------
        linear : forestflow.model.linear.LinearTheoryGrid
            Linear-theory grid associated with the supplied redshift(s).
        z : float or array_like
            Redshift scalar or axis accepted by the linear grid.
        k_iMpc, mu : array_like
            Comoving wavenumber in 1/Mpc and line-of-sight cosine in ``[0, 1]``.
            Inputs are broadcast and may include padded NaN cells, which are
            returned as NaN.
        ari_pp : mapping
            Named Arinyo coefficients. Missing entries use defaults.

        Returns
        -------
        numpy.ndarray
            P3D in Mpc^3 with the supported redshift and leading batch axes.

        Notes
        -----
        The large-scale term is ``(bias + bias_eta * f * mu**2)**2``;
        ``bias_eta`` is not the redshift-space parameter beta.
        """

        z = np.asarray(z)
        k_iMpc, mu = np.broadcast_arrays(
            np.asarray(k_iMpc, dtype=float), np.asarray(mu, dtype=float)
        )
        valid = np.isfinite(k_iMpc) & np.isfinite(mu)
        if np.any(k_iMpc[valid] < 0):
            raise ValueError("k_iMpc must be non-negative where it is finite")
        if np.any((mu[valid] < 0) | (mu[valid] > 1)):
            raise ValueError("mu must lie in [0, 1] where it is finite")
        # Archive/rebinning grids may contain padded NaN cells. Evaluate safe
        # placeholders and restore those cells as NaN in the returned model.
        k_iMpc_safe = np.where(valid, k_iMpc, 1.0)
        mu_safe = np.where(valid, mu, 0.0)

        scalar_z = z.ndim == 0

        if linear.is_batched:
            linP_Mpc = self.linear.get_linP_Mpc_batch(linear, k_iMpc_safe)
            fz = self.linear.get_fz_batch(linear)
            while fz.ndim < k_iMpc_safe.ndim:
                fz = fz[..., None]
            params = self.default_params | ari_pp
            result = self._arinyo_kernel(linP_Mpc, fz, k_iMpc_safe, mu_safe, params)
            return np.where(valid, result, np.nan)

        if not scalar_z:
            # Add a redshift axis only if it is missing.
            if k_iMpc_safe.ndim == z.ndim + 1:
                k_iMpc_safe = np.broadcast_to(
                    k_iMpc_safe, (len(z),) + k_iMpc_safe.shape
                )
                mu_safe = np.broadcast_to(mu_safe, (len(z),) + mu_safe.shape)
                valid = np.broadcast_to(valid, k_iMpc_safe.shape)

        # Check if all the default parameters are present in the ari_pp dictionary
        params = self.default_params | ari_pp

        linP_Mpc = self.linear.get_linP_Mpc(linear, z, k_iMpc_safe)
        fz = self.linear.get_fz(linear, z)

        while fz.ndim < k_iMpc_safe.ndim:
            fz = fz[..., None]

        result = self._arinyo_kernel(linP_Mpc, fz, k_iMpc_safe, mu_safe, params)
        return np.where(valid, result, np.nan)

    def P3D_Mpc_kpar_kperp_Gaussian_noise(
        self,
        linear,
        z,
        k_par_iMpc,
        k_perp_iMpc,
        ari_pp,
        seed=0,
        Lbox_Mpc=100,
        epsilon=0.0,
    ):
        """
        Return a Gaussian finite-volume realization of Cartesian P3D.
        """
        from forestflow.statistics.covariance import compute_Gaussian_cov

        p3d_Mpc = self.P3D_Mpc_kpar_kperp(linear, z, k_par_iMpc, k_perp_iMpc, ari_pp)
        sigma = compute_Gaussian_cov(
            k_par_iMpc, k_perp_iMpc, np.ravel(p3d_Mpc), Lbox_Mpc**3
        )
        return (
            np.ravel(p3d_Mpc)
            + np.random.default_rng(seed).normal(scale=sigma)
            + epsilon
        ).reshape(np.shape(p3d_Mpc))

    def P1D_Mpc(
        self, linear: Any, z: int | float, k_par_iMpc: Any, ari_pp: Any
    ) -> NDArray[Any]:
        """
        Project this model's Cartesian P3D directly into P1D.

        Scalar, redshift-vector, and leading-batch inputs are dispatched by
        :func:`forestflow.statistics.p1d.P1D_Mpc`; Arinyo has no separate
        scalar or batch P1D implementation.
        """
        return compute_P1D(
            linear,
            z,
            k_par_iMpc,
            self.P3D_Mpc_kpar_kperp,
            ari_pp,
        )

    def P1D_Mpc_Gaussian_noise(
        self,
        linear: Any,
        z: float,
        k_par_iMpc: ArrayLike,
        ari_pp: Mapping[str, Any] | None,
        seed: int = 0,
        Lbox_Mpc: Any = 100,
    ) -> NDArray[Any]:
        """
        Project a Gaussian finite-volume P3D realization into P1D.

        Parameters
        ----------
        linear : forestflow.model.linear.LinearTheoryGrid
            Precomputed linear-theory grid.
        z : float or array_like
            Redshift coordinate accepted by ``linear``.
        k_par_iMpc : array_like
            Parallel comoving wavenumbers in ``1 / Mpc``.
        ari_pp : mapping or None
            Arinyo coefficients forwarded to the noisy Cartesian P3D model.
        seed : int, default: 0
            Random seed for the independent Gaussian P3D perturbations.
        Lbox_Mpc : float, default: 100
            Cubic simulation-box side length in comoving Mpc, used to set mode
            counts in the diagonal Gaussian covariance.

        Returns
        -------
        ndarray
            Noisy P1D in comoving Mpc units, with the same scalar/redshift/
            leading-batch conventions as :meth:`P1D_Mpc`.

        Notes
        -----
        The stochastic perturbation is diagonal in the Cartesian P3D cells;
        it represents finite-volume Gaussian scatter, not emulator uncertainty.
        """

        return compute_P1D(
            linear,
            z,
            k_par_iMpc,
            self.P3D_Mpc_kpar_kperp_Gaussian_noise,
            ari_pp,
            seed=seed,
            Lbox_Mpc=Lbox_Mpc,
        )

    def Px_Mpc(
        self,
        z: float,
        k_par_iMpc: ArrayLike,
        r_perp_Mpc: Any,
        ari_pp: Any,
        new_cosmo_params: int | None = None,
    ) -> NDArray[Any]:
        """
        Project Arinyo P3D into transverse cross power.

        Parameters
        ----------
        z : float
            Single redshift at which a linear-theory grid is constructed.
        k_par_iMpc : array_like
            Parallel comoving wavenumbers in 1/Mpc.
        r_perp_Mpc : array_like
            Transverse separations in comoving Mpc.
        ari_pp : mapping
            Named Arinyo coefficients.
        new_cosmo_params : mapping, optional
            Cosmological changes forwarded to the linear-theory constructor.

        Returns
        -------
        numpy.ndarray
            Cross power with axes corresponding to ``k_par_iMpc`` and
            ``r_perp_Mpc``.
        """

        # NEEDS TO BE UPDATED!!!

        # check kmax in the fiducial cosmology
        camb_kmax_iMpc = self.linear.fiducial_cosmology.camb_kmax_Mpc

        linear = self.linear.get_linear_theory(z, new_cosmo_params=new_cosmo_params)
        Px_Mpc = pcross.Px_Mpc(
            linear,
            z,
            k_par_iMpc,
            r_perp_Mpc,
            self.P3D_Mpc_k_mu,
            p3d_params=ari_pp,
            max_k_for_p3d=camb_kmax_iMpc,
        )
        return Px_Mpc
