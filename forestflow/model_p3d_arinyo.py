import types
import numpy as np
from forestflow.camb_routines import get_linP_interp
from forestflow import pcross
from forestflow.p1d import P1D_Mpc as compute_P1D
from forestflow.cosmo import rescale_pklin


def coordinates(name):
    def decorator(func):
        func.coordinates = name
        return func

    return decorator


class ArinyoModel(object):
    """
    Class representing the Arinyo et al. model for Lyman-alpha forest flux power spectrum.
    """

    def __init__(
        self,
        cosmo=None,
        camb_pk_interp=None,
        default_bias=-0.18,
        default_beta=1.3,
        default_q1=0.4,
        default_q2=0.0,
        default_kvav=0.58,
        default_av=0.29,
        default_bv=1.55,
        default_kp=10.5,
        camb_kmax_Mpc=200.0,
    ):
        """
        Set up the flux power spectrum model.

        Parameters:
            camb_pk_interp (interpolator, optional): Precomputed linear power spectrum interpolator.
                If not provided, it will be obtained using the `get_linP_interp` function.
            cosmo (Cosmology, optional): CAMB params object defining the cosmology.
            default_bias (float, optional): Linear bias. Defaults to -0.18.
            default_beta (float, optional): Linear RSD. Defaults to 1.3.
            default_q1 (float, optional): Nonlinear growth. Defaults to 0.4.
            default_q2 (float, optional): Nonlinear growth. Defaults to 0.0.
            default_kvav (float, optional): Nonlinear RSD. Defaults to 0.58.
            default_av (float, optional): Nonlinear RSD. Defaults to 0.29.
            default_bv (float, optional): Nonlinear RSD. Defaults to 1.55.
            default_kp (float, optional): Nonlinear pressure. Defaults to 10.5.
            camb_kmax_Mpc (float, optional): Maximum k in Mpc^-1 to consider for the linear power spectrum.
                Defaults to 200.0.
        """

        if (cosmo is None) and (camb_pk_interp is None):
            raise ValueError("Must provide either cosmo or camb_pk_interp")

        if camb_pk_interp is None:
            # get a linear power interpolator
            self.get_linpower = get_linP_interp(cosmo, camb_kmax_Mpc=camb_kmax_Mpc)
        else:
            self.get_linpower = camb_pk_interp

        # store bias parameters
        self.default_params = {
            "bias": default_bias,
            "beta": default_beta,
            "q1": default_q1,
            "q2": default_q2,
            "kvav": default_kvav,
            "av": default_av,
            "bv": default_bv,
            "kp": default_kp,
        }
        self.default_bias = default_bias
        self.default_beta = default_beta
        self.default_q1 = default_q1
        self.default_q2 = default_q2
        self.default_kvav = default_kvav
        self.default_av = default_av
        self.default_bv = default_bv
        self.default_kp = default_kp

    def check_background(self, cosmo_new):
        change_expansion = False
        for par in self.get_linpower.cosmo:
            if par not in cosmo_new:
                raise ValueError(
                    "cosmo_new must contain all parameters in cosmo, missing " + par
                )
            if par not in ["As", "ns", "nrun"]:
                if cosmo_new[par] != self.get_linpower.cosmo[par]:
                    print(
                        "Expansion parameters changed for new cosmology: ",
                        par,
                        " ",
                        self.get_linpower.cosmo[par],
                        " ",
                        cosmo_new[par],
                    )
                    change_expansion = True
                    break

        return change_expansion

    def linP_Mpc(self, z, k_Mpc, cosmo_new=None):
        """
        Get the linear power spectrum at the input redshift and wavenumber.

        Parameters:
            z (float): Redshift
            k_Mpc (float): Wavenumber in Mpc^-1.

        Returns:
            linP (float): Linear power spectrum value.
        """

        if cosmo_new is not None:
            if self.check_background(cosmo_new):
                # call camb again if background has changed
                get_linpower = get_linP_interp(
                    cosmo_new, camb_kmax_Mpc=self.get_linpower.camb_kmax_Mpc
                )
                pklin = get_linpower(z, k_Mpc)
            else:
                # rescale pklin to new cosmology, faster than calling camb again
                pklin_orig, pklin = rescale_pklin(
                    z, k_Mpc, self.get_linpower, cosmo_new
                )
        else:
            pklin = self.get_linpower(z, k_Mpc)

        return pklin

    @coordinates("kpar_kperp")
    def P3D_Mpc_kpar_kperp(self, z, kpar, kperp, ari_pp, cosmo_new=None):
        """
        Compute the 3D flux power spectrum for inputs given as k_parallel and k_perp.

        Parameters:
            z (float): Redshift (scalar).
            kpar (float or array-like): Wavenumber component along the line-of-sight (Mpc^-1).
            kperp (float or array-like): Wavenumber component perpendicular to the line-of-sight (Mpc^-1).
            ari_pp (dict): Arinyo model parameters (missing keys will use defaults).
            cosmo_new (dict, optional): Optional cosmology override passed through to `P3D_Mpc`.

        Returns:
            float or array-like: 3D flux power spectrum in units of Mpc^3 with the same shape as the broadcasted
            inputs. The returned value is the same object produced by `P3D_Mpc` but with the attribute
            `coordinates` set to `'kpar_kperp'`.
        """

        k = np.sqrt(kpar**2 + kperp**2)
        mu = kpar / k
        return self._P3D_Mpc(z, k, mu, ari_pp, cosmo_new=cosmo_new)

    @coordinates("k_mu")
    def P3D_Mpc_k_mu(self, z, k, mu, ari_pp, cosmo_new=None):
        """
        Compute the 3D flux power spectrum for inputs given as k (magnitude) and mu (cosine of angle).

        Parameters:
            z (float): Redshift (scalar).
            k (float or array-like): Magnitude of the wavevector (Mpc^-1).
            mu (float or array-like): Cosine of the angle between the wavevector and the line-of-sight
                (mu = k_parallel / k).
            ari_pp (dict): Arinyo model parameters (missing keys will use defaults).
            cosmo_new (dict, optional): Optional cosmology override passed through to `P3D_Mpc`.

        Returns:
            float or array-like: 3D flux power spectrum in units of Mpc^3 with the same shape as the inputs.
            The returned value is the same object produced by `P3D_Mpc` but with the attribute
            `coordinates` set to `'k_mu'`.
        """
        return self._P3D_Mpc(z, k, mu, ari_pp, cosmo_new=cosmo_new)

    def _P3D_Mpc(self, z, k, mu, ari_pp, cosmo_new=None):
        """
        Compute the model for the 3D flux power spectrum in units of Mpc^3.

        Parameters:
            z (float): Redshift.
            k (float): Wavenumber.
            mu (float): Cosine of the angle between the line-of-sight and the wavevector.
            ari_pp (dict): Arinyo parameters

        Returns:
            float: Computed value of the 3D flux power spectrum.
        """

        # Check if all the default parameters are present in the ari_pp dictionary
        for par in self.default_params:
            if par not in ari_pp:
                print(
                    par,
                    " not in input, using default value, ",
                    self.default_params[par],
                )
                ari_pp[par] = self.default_params[par]

        # Evaluate the linear power spectrum at the given (z, k)
        linP = self.linP_Mpc(z, k, cosmo_new=cosmo_new)

        # Model the large-scale biasing for the flux field
        lowk_bias = ari_pp["bias"] * (1 + ari_pp["beta"] * mu**2)

        # Model the small-scale correction (D_NL in Arinyo-i-Prats 2015)
        delta2 = (1 / (2 * np.pi**2)) * k**3 * linP
        nonlin = delta2 * (ari_pp["q1"] + ari_pp["q2"] * delta2)
        vel = k ** ari_pp["av"] / ari_pp["kvav"] * mu ** ari_pp["bv"]
        press = (k / ari_pp["kp"]) ** 2

        D_NL = np.exp(nonlin * (1 - vel) - press)

        # Compute the final 3D flux power spectrum
        return linP * lowk_bias**2 * D_NL

    def P1D_Mpc(self, z, k_par, ari_pp, cosmo_new=None):
        """
        Compute the one-dimensional power spectrum (P1D) for the specified values of parallel wavenumber (k_par).

        Parameters:
            z (float): Redshift at which to compute the P1D.
            k_par (array-like): Array or list of values for the parallel wavenumber (k_par) for which the P1D should be computed.
            ari_pp (dict, optional): Additional parameters for the model. Defaults to an empty dictionary `{}`.
            cosmo_new (dict, optional): New cosmology parameters. Defaults to `None`, which means the existing cosmology will be used.

        Returns:
            array-like: Computed values of the one-dimensional power spectrum (P1D) for the given `k_par` values.
        """

        p1d = compute_P1D(z, k_par, self.P3D_Mpc_k_mu, ari_pp, cosmo_new=cosmo_new)

        return p1d

    def Px_Mpc(self, z, kpar_iMpc, rperp_Mpc, parameters):
        """
        Compute P-cross for the P3D model.

        Parameters:
            z (float): Redshift. Cannot be array.
            k_par (array-like): Array of k-parallel values at which to compute Px.
        Returns:
            rperp (array-like): values (float) of separation in Mpc
            Px_per_kpar (array-like): values (float) of Px for each k parallel and rperp. Shape: (len(k_par), len(rperp)).
        """
        Px_Mpc = pcross.Px_Mpc(
            z,
            kpar_iMpc,
            rperp_Mpc,
            self.P3D_Mpc,
            P3D_mode="pol",
            P3D_params=parameters,
        )
        return Px_Mpc
