import types
import numpy as np
from lace.cosmo import cosmology, rescale_cosmology
from forestflow import pcross
from forestflow.p1d import P1D_Mpc as compute_P1D


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
        fid_cosmo=None,
        default_bias=-0.18,
        default_beta=1.3,
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
            default_beta (float, optional): Linear RSD. Defaults to 1.3.
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
            "beta": default_beta,
            "q1": default_q1,
            "q2": default_q2,
            "kvav": default_kvav,
            "av": default_av,
            "bv": default_bv,
            "kp": default_kp,
        }


    def linP_Mpc(self, z, k_Mpc, new_cosmo_params=None):
        """
        Get the linear power spectrum at the input redshift and wavenumber.

        Parameters:
            z (float): Redshift
            k_Mpc (float): Wavenumber in Mpc^-1.
            new_cosmo_params (dictionary): modify fiducial cosmo

        Returns:
            linP (float): Linear power spectrum value.
        """

        if self.fid_cosmo.same_background(cosmo_params=new_cosmo_params):
            # get cosmology model using fiducial cosmo and input params
            cosmo = rescale_cosmology.RescaledCosmology(self.fid_cosmo, new_cosmo_params)
        else:
            print('WARNING: computing CAMB again')
            cosmo = cosmology.Cosmology(cosmo_params_dict=new_cosmo_params)

        return cosmo.get_linP_Mpc(z, k_Mpc)


    @coordinates("kpar_kperp")
    def P3D_Mpc_kpar_kperp(self, z, kpar, kperp, ari_pp, new_cosmo_params=None):
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
            inputs. The returned value is the same object produced by `P3D_Mpc` but with the attribute
            `coordinates` set to `'kpar_kperp'`.
        """

        k = np.sqrt(kpar**2 + kperp**2)
        mu = kpar / k
        return self._P3D_Mpc(z, k, mu, ari_pp, new_cosmo_params=new_cosmo_params)

    @coordinates("k_mu")
    def P3D_Mpc_k_mu(self, z, k, mu, ari_pp, new_cosmo_params=None):
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
            The returned value is the same object produced by `P3D_Mpc` but with the attribute
            `coordinates` set to `'k_mu'`.
        """
        return self._P3D_Mpc(z, k, mu, ari_pp, new_cosmo_params=new_cosmo_params)

    def _P3D_Mpc(self, z, k, mu, ari_pp, new_cosmo_params=None):
        """
        Compute the model for the 3D flux power spectrum in units of Mpc^3.

        Parameters:
            z (float): Redshift. It modifies the linear power spectrum but not the value of the Arinyo parameters
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
                    " not in ari_pp, using default value, ",
                    self.default_params[par],
                )
                ari_pp[par] = self.default_params[par]

        # Evaluate the linear power spectrum at the given (z, k)
        linP = self.linP_Mpc(z, k, new_cosmo_params=new_cosmo_params)

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

    def P1D_Mpc(self, z, k_par, ari_pp, new_cosmo_params=None):
        """
        Compute the one-dimensional power spectrum (P1D) for the specified values of parallel wavenumber (k_par).

        Parameters:
            z (float): Redshift at which to compute the P1D. It modifies the linear power spectrum but not the value of the Arinyo parameters
            k_par (array-like): Array or list of values for the parallel wavenumber (k_par) for which the P1D should be computed.
            ari_pp (dict, optional): Additional parameters for the model. Defaults to an empty dictionary `{}`.
            new_cosmo_params (dict, optional): New cosmology parameters. Defaults to `None`, which means the existing cosmology will be used.

        Returns:
            array-like: Computed values of the one-dimensional power spectrum (P1D) for the given `k_par` values.
        """

        p1d = compute_P1D(z, k_par, self.P3D_Mpc_k_mu, ari_pp, new_cosmo_params=new_cosmo_params)

        return p1d

    def Px_Mpc(self, z, kpar_iMpc, rperp_Mpc, ari_pp, new_cosmo_params=None):
        """
        Compute P-cross for the P3D model.

        Parameters:
            z (float): Redshift. Cannot be array.
            k_par (array-like): Array of k-parallel values at which to compute Px.
        Returns:
            rperp (array-like): values (float) of separation in Mpc
            Px_per_kpar (array-like): values (float) of Px for each k parallel and rperp. Shape: (len(k_par), len(rperp)).
        """

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
