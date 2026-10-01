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
    """Arinyo flux-power model evaluated from a supplied linear-theory grid."""

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
        Compute the 3D flux power spectrum for inputs given as k_parallel and k_perp.

        Parameters:
            z (float): Redshift (scalar). It modifies the linear power spectrum but not the value of the Arinyo parameters
            kpar (float or array-like): Wavenumber component along the line-of-sight (Mpc^-1).
            kperp (float or array-like): Wavenumber component perpendicular to the line-of-sight (Mpc^-1).
            ari_pp (dict): Arinyo model parameters (missing keys will use defaults).
            new_cosmo_params (dict, optional): Optional cosmology override passed through to `P3D_Mpc`.

        Returns:
            float or array-like: 3D flux power spectrum in units of Mpc^3 with the same shape as the broadcasted
            inputs.

        Other Parameters
        ----------------
        linear : object
            Precomputed linear-theory grid.
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
        """Evaluate the Arinyo nonlinear flux-power kernel."""
        bias = broadcast_leading_dimensions(ari_pp["bias"], k_iMpc)
        bias_eta = broadcast_leading_dimensions(ari_pp["bias_eta"], k_iMpc)
        q1 = broadcast_leading_dimensions(ari_pp["q1"], k_iMpc)
        q2 = broadcast_leading_dimensions(ari_pp["q2"], k_iMpc)
        av = broadcast_leading_dimensions(ari_pp["av"], k_iMpc)
        kvav = broadcast_leading_dimensions(ari_pp["kvav"], k_iMpc)
        bv = broadcast_leading_dimensions(ari_pp["bv"], k_iMpc)
        kp = broadcast_leading_dimensions(ari_pp["kp"], k_iMpc)
        lowk_bias = bias + bias_eta * fz * mu**2
        delta2 = k_iMpc**3 * linP_Mpc / (2 * np.pi**2)
        nonlin = delta2 * (q1 + q2 * delta2)
        velocity = k_iMpc**av / kvav * mu**bv
        pressure = (k_iMpc / kp) ** 2
        return linP_Mpc * lowk_bias**2 * np.exp(nonlin * (1 - velocity) - pressure)

    def P3D_Mpc_k_mu(
        self,
        linear: Any,
        z: float,
        k_iMpc: ArrayLike,
        mu: float,
        ari_pp: Mapping[str, Any],
    ) -> float:
        """
        Compute the model for the 3D flux power spectrum in units of Mpc^3.

        Parameters:
            z (float): Redshift. It modifies the linear power spectrum but not the value of the Arinyo parameters
            k (float): Wavenumber.
            mu (float): Cosine of the angle between the line-of-sight and the wavevector.
            ari_pp (dict): Arinyo parameters

        Returns:
            float: Computed value of the 3D flux power spectrum.

        Other Parameters
        ----------------
        linear : object
            Precomputed linear-theory grid.
        k_iMpc : numpy.ndarray
            Wavenumbers in inverse megaparsecs.
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
        """Return a Gaussian finite-volume realization of Cartesian P3D."""
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
        """Project this model's Cartesian P3D directly into P1D.

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
        Compute the one-dimensional power spectrum (P1D) for the specified values of parallel wavenumber (k_par).

        The error between simulations with Lbox_Mpc2 and Lbox_Mpc scales like fact = (Lbox_Mpc2/Lbox_Mpc)**(3/2).

        The covariance matrix is fully uncorrelated

        Parameters:
            z (float): Redshift at which to compute the P1D. It modifies the linear power spectrum but not the value of the Arinyo parameters
            k_par (array-like): Array or list of values for the parallel wavenumber (k_par) for which the P1D should be computed.
            ari_pp (dict, optional): Additional parameters for the model. Defaults to an empty dictionary `{}`.
            new_cosmo_params (dict, optional): New cosmology parameters. Defaults to `None`, which means the existing cosmology will be used.

        Returns:
            array-like: Computed values of the one-dimensional power spectrum (P1D) for the given `k_par` values.

        Other Parameters
        ----------------
        linear : object
            Precomputed linear-theory grid.
        seed : int
            Random-number generator seed.
        Lbox_Mpc : object
            Simulation-box length in megaparsecs.
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
        Compute P-cross for the P3D model.

        Parameters:
            z (float): Redshift. Cannot be array.
            k_par (array-like): Array of k-parallel values at which to compute Px.
        Returns:
            rperp (array-like): values (float) of separation in Mpc
            Px_per_kpar (array-like): values (float) of Px for each k parallel and rperp. Shape: (len(k_par), len(rperp)).

        Other Parameters
        ----------------
        k_par_iMpc : numpy.ndarray
            Kpar impc used by the calculation.
        r_perp_Mpc : object
            Rperp mpc used by the calculation.
        ari_pp : object
            Ari pp used by the calculation.
        new_cosmo_params : int
            Cosmological parameters overriding the fiducial cosmology.
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
