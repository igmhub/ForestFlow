"""
Evaluate the Arinyo Lyman-alpha forest flux-power model.
"""

import types
import time
import numpy as np
from scipy.interpolate import RectBivariateSpline, CubicSpline
from lace.cosmo import cosmology, rescale_cosmology
from forestflow import pcross

# from forestflow.p1d import P1D_Mpc as compute_P1D
from forestflow.integrate_p3d import P1D_Mpc as compute_P1D

from dataclasses import dataclass


@dataclass(slots=True)
class LinearTheoryGrid:
    """
    Store linear-theory values on redshift and wavenumber grids.
    """
    z: np.ndarray  # (Nz,)
    logk: np.ndarray  # (Nk,)
    loglinP: np.ndarray  # (Nz, Nk)
    fz: np.ndarray  # (Nz,)


class ArinyoModel(object):
    """
    Class representing the Arinyo et al. model for Lyman-alpha forest flux power spectrum.
    """

    def __init__(
        self,
        fid_cosmo=None,
        default_bias=-0.18,
        default_bias_eta=-0.23,
        default_q1=0.4,
        default_q2=0.0,
        default_kvav=0.58,
        default_av=0.29,
        default_bv=1.55,
        default_kp=10.5,
    ):
        """
        Set up the flux power spectrum model.

        Parameters:
            fid_cosmo (Cosmology, optional): object defining the fiducial cosmology.
            default_bias (float, optional): Linear bias. Defaults to -0.18.
            default_bias_eta (float, optional): Linear bias. Defaults to -0.23.
            default_q1 (float, optional): Nonlinear growth. Defaults to 0.4.
            default_q2 (float, optional): Nonlinear growth. Defaults to 0.0.
            default_kvav (float, optional): Nonlinear RSD. Defaults to 0.58.
            default_av (float, optional): Nonlinear RSD. Defaults to 0.29.
            default_bv (float, optional): Nonlinear RSD. Defaults to 1.55.
            default_kp (float, optional): Nonlinear pressure. Defaults to 10.5.
        """

        if fid_cosmo is None:
            self.fid_cosmo = cosmology.Cosmology()
        else:
            self.fid_cosmo = fid_cosmo

        # store bias parameters
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

    def linear_theory(self, zs, k_Mpc_min=1e-3, k_Mpc_max=100, new_cosmo_params=None):
        """
        Compute all linear-theory quantities required by the model.

        Parameters
        ----------
        zs : object
            Zs used by the calculation.
        k_Mpc_min : float, optional
            K mpc min used by the calculation.
        k_Mpc_max : int, optional
            K mpc max used by the calculation.
        new_cosmo_params : object, optional
            New cosmo params used by the calculation.

        Returns
        -------
        object
            Result produced when the function is used to compute all linear-theory quantities required by the model.
        """

        if self.fid_cosmo.same_background(cosmo_params=new_cosmo_params):
            # get cosmology model using fiducial cosmo and input params
            cosmo = rescale_cosmology.RescaledCosmology(
                self.fid_cosmo, new_cosmo_params
            )
        else:
            print("WARNING: computing CAMB again")
            cosmo = cosmology.Cosmology(cosmo_params_dict=new_cosmo_params)

        logk = np.linspace(
            np.log(k_Mpc_min),
            np.log(k_Mpc_max),
            200,
        )

        zs = np.atleast_1d(zs)

        return LinearTheoryGrid(
            z=zs,
            logk=logk,
            loglinP=np.log(cosmo.get_linP_Mpc(zs, np.exp(logk))),
            fz=cosmo.compute_growth_rate(zs),
        )

    def linP_Mpc(self, linear, z, k_Mpc):
        """
        Evaluate the linear power spectrum.

        Parameters
        ----------
        linear : object
            Linear used by the calculation.
        z : int or float
            Redshift.
        k_Mpc : object
            K mpc used by the calculation.

        Returns
        -------
        object
            Result produced when the function is used to evaluate the linear power spectrum.
        """

        def get_iz(zi):
            """
            Return the grid index for a requested redshift.

            Parameters
            ----------
            zi : object
                Zi used by the calculation.

            Returns
            -------
            object
                Result produced when the function is used to return the grid index for a requested redshift.
            """
            matches = np.where(np.isclose(linear.z, zi, atol=1e-3, rtol=0))[0]

            if len(matches) == 0:
                raise ValueError(
                    f"Requested z={zi} is not available in the linear theory grid."
                )

            return matches[0]

        z = np.asarray(z, dtype=float)
        k_Mpc = np.asarray(k_Mpc, dtype=float)
        logk = np.log(k_Mpc)

        # Scalar z
        if z.ndim == 0:
            iz = get_iz(z)
            return np.exp(np.interp(logk, linear.logk, linear.loglinP[iz]))

        # (Nz,) z and (Nk,) k -> (Nz,Nk)
        elif z.ndim == 1 and k_Mpc.ndim == 1:
            out = np.empty((len(z), len(k_Mpc)))
            for i, zi in enumerate(z):
                iz = get_iz(zi)
                out[i] = np.exp(np.interp(logk, linear.logk, linear.loglinP[iz]))
            return out

        elif z.ndim == 1 and k_Mpc.shape[0] == len(z):

            out = np.empty_like(k_Mpc)

            for i, zi in enumerate(z):
                iz = get_iz(zi)

                out[i] = np.exp(
                    np.interp(
                        logk[i].ravel(),
                        linear.logk,
                        linear.loglinP[iz],
                    ).reshape(k_Mpc.shape[1:])
                )

            return out

        else:
            raise NotImplementedError(
                f"Unsupported shapes: z={z.shape}, k_Mpc={k_Mpc.shape}"
            )

    def fz(self, linear, z):
        """
        Evaluate the linear growth rate.

        Parameters
        ----------
        linear : object
            Linear used by the calculation.
        z : int or float
            Redshift.

        Returns
        -------
        object
            Result produced when the function is used to evaluate the linear growth rate.
        """

        return np.interp(
            np.asarray(z, dtype=float),
            linear.z,
            linear.fz,
        )

    def P3D_Mpc_kpar_kperp(self, linear, z, kpar, kperp, ari_pp):
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

        k_Mpc = np.sqrt(kpar**2 + kperp**2)
        mu = kpar / k_Mpc
        return self._P3D_Mpc(linear, z, k_Mpc, mu, ari_pp)

    def P3D_Mpc_k_mu(self, linear, z, k_Mpc, mu, ari_pp):
        """
        Compute the 3D flux power spectrum for inputs given as k (magnitude) and mu (cosine of angle).

        Parameters:
            z (float): Redshift (scalar). It modifies the linear power spectrum but not the value of the Arinyo parameters
            k (float or array-like): Magnitude of the wavevector (Mpc^-1).
            mu (float or array-like): Cosine of the angle between the wavevector and the line-of-sight
                (mu = k_parallel / k).
            ari_pp (dict): Arinyo model parameters (missing keys will use defaults).
            new_cosmo_params (dict, optional): Optional cosmology override passed through to `P3D_Mpc`.

        Returns:
            float or array-like: 3D flux power spectrum in units of Mpc^3 with the same shape as the inputs.

        Other Parameters
        ----------------
        linear : object
            Precomputed linear-theory grid.
        k_Mpc : numpy.ndarray
            Wavenumbers in inverse megaparsecs.
        """
        return self._P3D_Mpc(linear, z, k_Mpc, mu, ari_pp)

    def P3D_Mpc_kpar_kperp_Gaussian_noise(
        self,
        linear,
        z,
        kpar,
        kperp,
        ari_pp,
        seed=0,
        Lbox_Mpc=100,
        epsilon=0.0,
    ):
        """
        Compute the 3D flux power spectrum for inputs given as k (magnitude) and mu (cosine of angle).

        Parameters:
            z (float): Redshift (scalar). It modifies the linear power spectrum but not the value of the Arinyo parameters
            k (float or array-like): Magnitude of the wavevector (Mpc^-1).
            mu (float or array-like): Cosine of the angle between the wavevector and the line-of-sight
                (mu = k_parallel / k).
            ari_pp (dict): Arinyo model parameters (missing keys will use defaults).
            new_cosmo_params (dict, optional): Optional cosmology override passed through to `P3D_Mpc`.

        Returns:
            float or array-like: 3D flux power spectrum in units of Mpc^3 with the same shape as the inputs.

        Other Parameters
        ----------------
        linear : object
            Precomputed linear-theory grid.
        kpar : numpy.ndarray
            Line-of-sight wavenumbers.
        kperp : numpy.ndarray
            Transverse wavenumbers.
        seed : int
            Random-number generator seed.
        Lbox_Mpc : object
            Simulation-box length in megaparsecs.
        epsilon : object
            Regularization amplitude.
        """

        # Evaluate P3D
        P3D = self.P3D_Mpc_kpar_kperp(linear, z, kpar, kperp, ari_pp)
        _P3D = P3D.reshape(-1)

        vol = Lbox_Mpc**3
        sigma = compute_Gaussian_cov(kpar, kperp, _P3D, vol)

        # realization
        rng = np.random.default_rng(seed)
        P3D_err = _P3D + rng.normal(scale=sigma) + epsilon

        P3D_err = P3D_err.reshape(kpar.shape[0], kpar.shape[1])

        return P3D_err

    def _arinyo_kernel(self, linP_Mpc, fz, k_Mpc, mu, ari_pp):
        """
        Compute the nonlinear correction to the flux power spectrum.

        Parameters
        ----------
        linP : ndarray
            Linear matter power spectrum.
        fz : float or ndarray
            Linear growth rate.
        k : ndarray
        mu : ndarray
        ari_pp : dict

        Returns
        -------
        ndarray
            Flux power spectrum.

        Other Parameters
        ----------------
        linP_Mpc : object
            Linp mpc used by the calculation.
        k_Mpc : numpy.ndarray
            Wavenumbers in inverse megaparsecs.
        """

        bias = _bcast(ari_pp["bias"], k_Mpc)
        bias_eta = _bcast(ari_pp["bias_eta"], k_Mpc)
        q1 = _bcast(ari_pp["q1"], k_Mpc)
        q2 = _bcast(ari_pp["q2"], k_Mpc)
        av = _bcast(ari_pp["av"], k_Mpc)
        kvav = _bcast(ari_pp["kvav"], k_Mpc)
        bv = _bcast(ari_pp["bv"], k_Mpc)
        kp = _bcast(ari_pp["kp"], k_Mpc)

        lowk_bias = bias + bias_eta * fz * mu**2

        delta2 = k_Mpc**3 * linP_Mpc / (2 * np.pi**2)

        nonlin = delta2 * (q1 + q2 * delta2)

        vel = k_Mpc**av / kvav * mu**bv

        press = (k_Mpc / kp) ** 2

        dnl = np.exp(nonlin * (1 - vel) - press)

        return linP_Mpc * lowk_bias**2 * dnl

    def _P3D_Mpc(self, linear, z, k_Mpc, mu, ari_pp):
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
        k_Mpc : numpy.ndarray
            Wavenumbers in inverse megaparsecs.
        """

        z = np.asarray(z)
        k_Mpc = np.asarray(k_Mpc)
        mu = np.asarray(mu)

        scalar_z = z.ndim == 0

        if not scalar_z:
            # Add a redshift axis only if it is missing.
            if k_Mpc.ndim == z.ndim + 1:
                k_Mpc = np.broadcast_to(
                    k_Mpc,
                    (len(z),) + k_Mpc.shape,
                )
                mu = np.broadcast_to(
                    mu,
                    (len(z),) + mu.shape,
                )

        # Check if all the default parameters are present in the ari_pp dictionary
        params = self.default_params | ari_pp

        linP_Mpc = self.linP_Mpc(linear, z, k_Mpc)
        fz = self.fz(linear, z)

        while fz.ndim < k_Mpc.ndim:
            fz = fz[..., None]

        res = self._arinyo_kernel(linP_Mpc, fz, k_Mpc, mu, params)

        return res

    def P1D_Mpc(self, linear, z, k_par, ari_pp):
        """
        Compute the one-dimensional flux power spectrum.

        Parameters
        ----------
        linear : object
            Linear used by the calculation.
        z : int or float
            Redshift.
        k_par : object
            K par used by the calculation.
        ari_pp : object
            Ari pp used by the calculation.

        Returns
        -------
        object
            Result produced when the function is used to compute the one-dimensional flux power spectrum.
        """

        scalar_z = np.ndim(z) == 0

        p1d = compute_P1D(
            linear,
            z,
            k_par,
            self.P3D_Mpc_k_mu,
            ari_pp,
        )

        if scalar_z:
            return p1d[0]

        return p1d

    def P1D_Mpc_Gaussian_noise(self, linear, z, k_par, ari_pp, seed=0, Lbox_Mpc=100):
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
            k_par,
            self.P3D_Mpc_kpar_kperp_Gaussian_noise,
            ari_pp,
            seed=seed,
            Lbox_Mpc=Lbox_Mpc,
        )

    def Px_Mpc(self, z, kpar_iMpc, rperp_Mpc, ari_pp, new_cosmo_params=None):
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
        kpar_iMpc : numpy.ndarray
            Kpar impc used by the calculation.
        rperp_Mpc : object
            Rperp mpc used by the calculation.
        ari_pp : object
            Ari pp used by the calculation.
        new_cosmo_params : int
            Cosmological parameters overriding the fiducial cosmology.
        """

        # NEEDS TO BE UPDATED!!!

        # check kmax in the fiducial cosmology
        camb_kmax_Mpc = self.fid_cosmo.camb_kmax_Mpc

        Px_Mpc = pcross.Px_Mpc(
            z,
            kpar_iMpc,
            rperp_Mpc,
            self.P3D_Mpc_k_mu,
            p3d_params=ari_pp,
            max_k_for_p3d=camb_kmax_Mpc,
            new_cosmo_params=new_cosmo_params,
        )
        return Px_Mpc


def compute_Gaussian_cov(kpar, kperp, P3D, vol):

    # linear
    """
    Compute Gaussian covariance.

    Parameters
    ----------
    kpar : numpy.ndarray
        Kpar used by the calculation.
    kperp : numpy.ndarray
        Kperp used by the calculation.
    P3D : numpy.ndarray
        P3d used by the calculation.
    vol : object
        Vol used by the calculation.

    Returns
    -------
    object
        Result produced when the function is used to compute gaussian covariance.
    """
    dkpar = kpar[1, 0] - kpar[0, 0]

    # logarithmic
    dkperp = kperp[0, 1:] - kperp[0, :-1]
    dkperp = np.append(dkperp, dkperp[-1] ** 2 / dkperp[-2])

    # get Gaussian covariance
    Nmodes = (vol / (2 * np.pi) ** 2) * kperp * dkperp[np.newaxis, :] * dkpar
    sigma = np.sqrt(2.0 * P3D**2 / Nmodes.reshape(-1))

    return sigma


def _bcast(x, target):
    """
    Broadcast an array to the requested leading dimensions.

    Parameters
    ----------
    x : object
        X used by the calculation.
    target : object
        Target used by the calculation.

    Returns
    -------
    object
        Result produced when the function is used to broadcast an array to the requested leading dimensions.
    """
    x = np.asarray(x)
    return x.reshape(x.shape + (1,) * (target.ndim - x.ndim))
