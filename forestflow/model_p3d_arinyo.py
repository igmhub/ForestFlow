import types
import numpy as np
from lace.cosmo import camb_cosmo
from scipy.integrate import simpson
from forestflow.camb_routines import P_camb
from forestflow import pcross


def rescale_pklin(
    z, k_Mpc, fun_linpower, fid_cosmo, tar_cosmo, kp_Mpc=0.7, ks_Mpc=0.05
):
    ratio_As = tar_cosmo["As"] / fid_cosmo["As"]
    delta_ns = tar_cosmo["ns"] - fid_cosmo["ns"]
    delta_nrun = tar_cosmo["nrun"] - fid_cosmo["nrun"]

    ln_kp_ks = np.log(kp_Mpc / ks_Mpc)
    delta_alpha_p = delta_nrun
    delta_n_p = delta_ns + delta_nrun * ln_kp_ks
    ln_ratio_A_p = (
        np.log(ratio_As) + (delta_ns + 0.5 * delta_nrun * ln_kp_ks) * ln_kp_ks
    )

    rotk = np.log(k_Mpc / kp_Mpc)
    pklin = fun_linpower(z, k_Mpc, grid=False)

    pklin_rescaled = pklin * np.exp(
        ln_ratio_A_p + delta_n_p * rotk + 0.5 * delta_alpha_p * rotk**2
    )

    return pklin, pklin_rescaled


def get_linP_interp(cosmo, zmin=0, zmax=10, nz=256, camb_kmax_Mpc=200.0):
    """
    Obtain an interpolator of the linear power spectrum from CAMB.

    Parameters:
        cosmo (Cosmology): Cosmology object representing the cosmological parameters.
        zs (list): List of redshifts at which to obtain the linear power spectrum.
        camb_results (CAMBResults): CAMBResults object containing the precomputed CAMB results.
        camb_kmax_Mpc (float, optional): Maximum k in Mpc^-1 to consider for the linear power spectrum.
            Defaults to 30.

    Returns:
        linP_interp (interpolator): Interpolator for the linear power spectrum.
    """

    inst_camb = camb_cosmo.get_cosmology_from_dictionary(cosmo)

    camb_results = camb_cosmo.get_camb_results(
        inst_camb, zs=np.linspace(zmin, zmax, nz), camb_kmax_Mpc=camb_kmax_Mpc
    )

    # get interpolator from CAMB
    # meaning of var1 and var2 here
    # https://camb.readthedocs.io/en/latest/transfer_variables.html#transfer-variables
    linP_interp = camb_results.get_matter_power_interpolator(
        nonlinear=False,
        var1=8,
        var2=8,
        hubble_units=False,
        k_hunit=False,
        log_interp=True,
    )

    return linP_interp


class ArinyoModel(object):
    """
    Class representing the Arinyo et al. model for Lyman-alpha forest flux power spectrum.
    """

    def __init__(
        self,
        cosmo,
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
            zs (list, optional): Redshifts for which predictions are desired.
            camb_results (CAMBResults, optional): Precomputed CAMBResults object.
            default_bias (float, optional): Starting value for the flux bias. Defaults to -0.18.
            default_beta (float, optional): RSD parameter for the flux. Defaults to 1.3.
            default_d1_{} (float, optional): Parameters in the non-linear model.
            default_kvav (float, optional): Units (1/Mpc)^(av). Defaults to 0.58.
            default_kp (float, optional): Units 1/Mpc. Defaults to 10.5.
            camb_kmax_Mpc (float, optional): Maximum k in Mpc^-1 to consider for the linear power spectrum.
                Defaults to 100.0.
        """

        self.cosmo = cosmo
        self.camb_kmax_Mpc = camb_kmax_Mpc

        if camb_pk_interp is None:
            # get a linear power interpolator
            self.linP_interp = get_linP_interp(
                cosmo, camb_kmax_Mpc=self.camb_kmax_Mpc
            )
        else:
            self.linP_interp = camb_pk_interp
        self.get_linpower = types.MethodType(P_camb, self.linP_interp)

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
        for par in self.cosmo:
            if par not in cosmo_new:
                raise ValueError(
                    "cosmo_new must contain all parameters in cosmo, missing "
                    + par
                )
            if par not in ["As", "ns", "nrun"]:
                if cosmo_new[par] != self.cosmo[par]:
                    print(
                        "Expansion parameters changed for new cosmology: ",
                        par,
                        " ",
                        self.cosmo[par],
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
                linP_interp = get_linP_interp(
                    cosmo_new, camb_kmax_Mpc=self.camb_kmax_Mpc
                )
                get_linpower = types.MethodType(P_camb, linP_interp)
                pklin = get_linpower(z, k_Mpc, grid=False)
            else:
                # rescale pklin to new cosmology, faster than calling camb again
                pklin_orig, pklin = rescale_pklin(
                    z, k_Mpc, self.get_linpower, self.cosmo, cosmo_new
                )
        else:
            pklin = self.get_linpower(z, k_Mpc, grid=False)

        return pklin

    #
    def P3D_Mpc(self, z, k, mu, ari_pp, cosmo_new=None):
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

        for par in default_params:
            if par not in ari_pp:
                print(
                    par,
                    " not in input, using default value, ",
                    self.default_params[par],
                )
                ari_pp[par] = self.default_params[par]

        # evaluate linear power at input (z,k)
        linP = self.linP_Mpc(z, k, cosmo_new=cosmo_new)

        # model large-scales biasing for delta_flux(k)
        lowk_bias = ari_pp["bias"] * (1 + ari_pp["beta"] * mu**2)

        # model small-scales correction (D_NL in Arinyo-i-Prats 2015)
        delta2 = (1 / (2 * np.pi**2)) * k**3 * linP
        nonlin = delta2 * (ari_pp["q1"] + ari_pp["q2"] * delta2)
        vel = k ** ari_pp["av"] / ari_pp["kvav"] * mu ** ari_pp["bv"]
        press = (k / ari_pp["kp"]) ** 2

        D_NL = np.exp(nonlin * (1 - vel) - press)

        return linP * lowk_bias**2 * D_NL

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

    def P1D_Mpc(
        self,
        z,
        k_par,
        ari_pp,
        k_perp_min=0.001,
        k_perp_max=100,
        n_k_perp=99,
        cosmo_new=None,
    ):
        """
        Returns P1D for specified values of k_par, with the option to specify values of k_perp to be integrated over.

        Parameters:
            z (float): Redshift.
            k_par (array-like): Array or list of values for which P1D is to be computed.
            k_perp_min (float, optional): Lower bound of integral. Defaults to 0.001.
            k_perp_max (float, optional): Upper bound of integral. Defaults to 100.
            n_k_perp (int, optional): Number of points in integral. Defaults to 99.
            parameters (dict, optional): Additional parameters for the model. Defaults to {}.

        Returns:
            array-like: Computed values of P1D.
        """

        ln_k_perp = np.linspace(
            np.log(k_perp_min), np.log(k_perp_max), n_k_perp
        )

        p1d = self._P1D_lnkperp_fast(
            z, ln_k_perp, k_par, ari_pp, cosmo_new=cosmo_new
        )

        return p1d

    def _P1D_lnkperp_fast(self, z, ln_k_perp, kpars, ari_pp, cosmo_new=None):
        """
        Compute P1D by integrating P3D in terms of ln(k_perp) using a fast method.

        Parameters:
            z (float): Redshift.
            ln_k_perp (array-like): Array of natural logarithms of the perpendicular wavenumber.
            kpars (array-like): Array of parallel wavenumbers.
            parameters (dict, optional): Additional parameters for the model. Defaults to {}.

        Returns:
            array-like: Computed values of P1D.
        """

        # get interval for integration
        dlnk = ln_k_perp[1] - ln_k_perp[0]

        # get function to be integrated
        # it is equivalent of the inner loop of _P1D_lnkperp
        k_perp = np.exp(ln_k_perp)
        k = np.sqrt(kpars[np.newaxis, :] ** 2 + k_perp[:, np.newaxis] ** 2)
        mu = kpars[np.newaxis, :] / k
        k = k.swapaxes(0, 1)
        mu = mu.swapaxes(0, 1)

        fact = (1 / (2 * np.pi)) * k_perp[:, np.newaxis] ** 2
        fact = fact.swapaxes(0, 1)

        p3d_fix_k_par = (
            self.P3D_Mpc(z, k, mu, ari_pp, cosmo_new=cosmo_new) * fact
        )

        # perform numerical integration
        p1d = simpson(p3d_fix_k_par, ln_k_perp, dx=dlnk, axis=1)

        return p1d

    def _P1D_lnkperp_fast_smooth(
        self, z, ln_k_perp, kpars, k3d_smooth, parameters={}
    ):
        """
        Compute P1D by integrating P3D in terms of ln(k_perp) with smoothing.

        Parameters:
            z (float): Redshift.
            ln_k_perp (array-like): Array of natural logarithms of the perpendicular wavenumber.
            kpars (array-like): Array of parallel wavenumbers.
            k3d_smooth (float): Smoothing scale in units of k_perp.
            parameters (dict, optional): Additional parameters for the model. Defaults to {}.

        Returns:
            array-like: Computed values of P1D.
        """

        # get interval for integration
        dlnk = ln_k_perp[1] - ln_k_perp[0]

        # get function to be integrated
        # it is equivalent of the inner loop of _P1D_lnkperp
        k_perp = np.exp(ln_k_perp)
        k = np.sqrt(kpars[np.newaxis, :] ** 2 + k_perp[:, np.newaxis] ** 2)
        mu = kpars[np.newaxis, :] / k
        k = k.swapaxes(0, 1)
        mu = mu.swapaxes(0, 1)

        fact = (1 / (2 * np.pi)) * k_perp[:, np.newaxis] ** 2
        fact = fact.swapaxes(0, 1)
        p3d_fix_k_par = self.P3D_Mpc(z, k, mu, parameters) * fact

        # perform numerical integration
        kernel = np.sinc(k3d_smooth * np.exp(ln_k_perp))
        # print(p3d_fix_k_par.shape, kernel.shape, kernel.shape)
        p1d = simpson(
            p3d_fix_k_par * kernel[np.newaxis, :] ** 2,
            ln_k_perp,
            dx=dlnk,
            axis=1,
        )

        return p1d

    def P1D_Mpc_smooth(
        self,
        z,
        k_par,
        k3d_smooth,
        k_perp_min=0.001,
        k_perp_max=100,
        n_k_perp=99,
        parameters={},
    ):
        """
        Returns P1D for specified values of k_par, with the option to specify values of k_perp to be integrated over.

        Smooth refers to computing P3D from grid.

        Parameters:
            z (float): Redshift.
            k_par (array-like): Array or list of values for which P1D is to be computed.
            k3d_smooth (float): Smoothing scale in units of k_perp.
            k_perp_min (float, optional): Lower bound of integral. Defaults to 0.001.
            k_perp_max (float, optional): Upper bound of integral. Defaults to 100.
            n_k_perp (int, optional): Number of points in integral. Defaults to 99.
            parameters (dict, optional): Additional parameters for the model. Defaults to {}.

        Returns:
            array-like: Computed values of P1D.
        """

        ln_k_perp = np.linspace(
            np.log(k_perp_min), np.log(k_perp_max), n_k_perp
        )

        p1d = self._P1D_lnkperp_fast_smooth(
            z, ln_k_perp, k_par, k3d_smooth, parameters
        )

        return p1d
