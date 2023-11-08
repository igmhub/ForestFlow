import types
import numpy as np
from lace.cosmo import camb_cosmo
from scipy.integrate import simpson
from lya_pk.camb_routines import P_camb
from lya_pk.utils import memoize_numpy_arrays


@memoize_numpy_arrays
def get_nmod(k, dk, Lbox):
    """
    Calculate the number of modes in a given k bin.

    Parameters:
        k (float): Center of the k bin.
        dk (float): Width of the k bin.
        Lbox (float): Size of the simulation box.

    Returns:
        Nk (float): Number of modes in the k bin.
    """
    Vs = 4 * np.pi**2 * k**2 * dk * (1 + 1 / 12 * (dk / k) ** 2)
    kf = 2 * np.pi / Lbox
    Vk = kf**3
    Nk = Vs / Vk
    return Nk


@memoize_numpy_arrays
def get_linP_interp(cosmo, zs, camb_results, camb_kmax_Mpc=30):
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

    if camb_results is None:
        camb_results = camb_cosmo.get_camb_results(
            cosmo, zs=zs, camb_kmax_Mpc=camb_kmax_Mpc
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

    Attributes:
        omega_m (float): Matter density parameter.
        omega_b (float): Baryon density parameter.
        h (float): Reduced Hubble constant.
        n_s (float): Spectral index of the primordial power spectrum.
        sigma_8 (float): RMS amplitude of linear matter fluctuations in spheres of radius 8 Mpc/h.
        gamma (float): Slope of the mass-temperature relation in the hydrodynamical simulations.
        T0 (float): Temperature of the intergalactic medium at mean density.
        gamma_lambda (float): Redshift dependence of the temperature-density relation.
        P0_He (float): Normalization of the helium photoionization rate.
        gamma_He (float): Power-law index of the helium photoionization rate.
        default_q1 (float): Default value for the small scales correction parameter q1.
        default_q2 (float): Default value for the small scales correction parameter q2.
        default_kvav (float): Default value for the small scales correction parameter kvav.
        default_av (float): Default value for the small scales correction parameter av.
        default_bv (float): Default value for the small scales correction parameter bv.
        default_kp (float): Default value for the small scales correction parameter kp.
        default_beta_p1d (float): Default value for the redshift distortion parameter beta_p1d.
        default_b_p1d (float): Default value for the linear bias parameter b_p1d.

    Methods:
        __init__(self, omega_m, omega_b, h, n_s, sigma_8, gamma, T0, gamma_lambda, P0_He, gamma_He,
                 default_q1, default_q2, default_kvav, default_av, default_bv, default_kp,
                 default_beta_p1d, default_b_p1d): Initializes the ArinyoModel instance.
        linP_Mpc(self, z, k): Computes the linear power spectrum at a given redshift and wavenumber.
        P1D_Mpc(self, z, k_par, k_perp_min=0.001, k_perp_max=100, n_k_perp=99, parameters={}):
            Computes P1D for specified values of k_par by integrating over k_perp.
        P1D_Mpc_smooth(self, z, k_par, k3d_smooth, k_perp_min=0.001, k_perp_max=100, n_k_perp=99, parameters={}):
            Computes P1D for specified values of k_par with a smoothing kernel.
        rat_P3D(self, z, k, mu, parameters={}): Computes the ratio of the non-linear to linear power spectrum.
        P3D_Mpc(self, z, k, mu, parameters={}): Computes the non-linear power spectrum.
        small_scales_correction(self, z, k, mu, parameters={}): Computes the small-scales correction to delta_flux biasing.
        _P3D_kperp2(self, z, ln_k_perp, k_par, parameters={}): Function to be integrated to compute P1D.
        _P1D_lnkperp(self, z, ln_k_perp, kpars, parameters={}): Function to be integrated to compute P1D.
        _rat_P1D_lnkperp(self, z, ln_k_perp, kpars, parameters={}): Function to be integrated to compute P1D.
        _P1D_lnkperp_fast(self, z, ln_k_perp, kpars, parameters={}): Computes P1D by integrating P3D in terms of ln(k_perp).
        _P1D_lnkperp_fast_smooth(self, z, ln_k_perp, kpars, k3d_smooth, parameters={}):
            Computes P1D by integrating P3D in terms of ln(k_perp) with smoothing.
        rel_err_P1d(self, z, k_par, Lbox, res_los, nskewers, n_k_perp=1000, parameters={}):
            Computes the relative error of P1D.
    """

    def __init__(
        self,
        camb_pk_interp=None,
        cosmo=None,
        zs=None,
        camb_results=None,
        default_bias=-0.18,
        default_beta=1.3,
        default_q1=0.4,
        default_q2=0.0,
        default_kvav=0.58,
        default_av=0.29,
        default_bv=1.55,
        default_kp=10.5,
        camb_kmax_Mpc=100.0,
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

        if camb_pk_interp is None:
            # get a linear power interpolator
            try:
                self.linP_interp = get_linP_interp(
                    cosmo, zs, camb_results, camb_kmax_Mpc=camb_kmax_Mpc
                )
            except:
                raise ValueError(
                    "If camb_pk_interp is not provided, cosmo and zs must be"
                )
        else:
            self.linP_interp = camb_pk_interp
            self.linP_interp.P = types.MethodType(P_camb, self.linP_interp)

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

    @memoize_numpy_arrays
    def linP_Mpc(self, z, k_Mpc):
        """
        Get the linear power spectrum at the input redshift and wavenumber.

        Parameters:
            z (float): Redshift.
            k_Mpc (float): Wavenumber in Mpc^-1.

        Returns:
            linP (float): Linear power spectrum value.
        """

        return self.linP_interp.P(z, k_Mpc, grid=False)

    def P3D_Mpc_check(self, z, k, mu, parameters={}):
        """
        Compute the model for the 3D flux power spectrum in units of Mpc^3.

        Parameters:
            z (float): Redshift.
            k (float): Wavenumber.
            mu (float): Cosine of the angle between the line-of-sight and the wavevector.
            parameters (dict, optional): Additional parameters for the model. Defaults to {}.

        Returns:
            float: Computed value of the 3D flux power spectrum.
        """

        # evaluate linear power at input (z,k)
        linP = self.linP_Mpc(z, k)

        # model large-scales biasing for delta_flux(k)
        lowk_bias = self.lowk_biasing(mu, parameters)

        # model small-scales correction (D_NL in Arinyo-i-Prats 2015)
        D_NL = self.small_scales_correction(z, k, mu, parameters)

        return linP * lowk_bias**2 * D_NL

    def P3D_Mpc(self, z, k, mu, pp):
        """
        Compute the model for the 3D flux power spectrum in units of Mpc^3.

        Parameters:
            z (float): Redshift.
            k (float): Wavenumber.
            mu (float): Cosine of the angle between the line-of-sight and the wavevector.
            parameters (dict, optional): Additional parameters for the model. Defaults to {}.

        Returns:
            float: Computed value of the 3D flux power spectrum.
        """

        # evaluate linear power at input (z,k)
        linP = self.linP_Mpc(z, k)

        # model large-scales biasing for delta_flux(k)
        lowk_bias = pp["bias"] * (1 + pp["beta"] * mu**2)

        # model small-scales correction (D_NL in Arinyo-i-Prats 2015)
        delta2 = (1 / (2 * np.pi**2)) * k**3 * linP
        nonlin = delta2 * (pp["q1"] + pp["q2"] * delta2)
        vel = k ** pp["av"] / pp["kvav"] * mu ** pp["bv"]
        press = (k / pp["kp"]) ** 2

        D_NL = np.exp(nonlin * (1 - vel) - press)

        return linP * lowk_bias**2 * D_NL

    def rat_P3D(self, z, k, mu, parameters={}):
        """
        Compute the model for the ratio of the 3D flux power spectrum and Plin (no units).

        Parameters:
            z (float): Redshift.
            k (float): Wavenumber.
            mu (float): Cosine of the angle between the line-of-sight and the wavevector.
            parameters (dict, optional): Additional parameters for the model. Defaults to {}.

        Returns:
            float: Computed value of the ratio of the 3D flux power spectrum and Plin.
        """

        # model large-scales biasing for delta_flux(k)
        lowk_bias = self.lowk_biasing(mu, parameters)

        # model small-scales correction (D_NL in Arinyo-i-Prats 2015)
        D_NL = self.small_scales_correction(z, k, mu, parameters)

        return lowk_bias**2 * D_NL

    def lowk_biasing(self, mu, parameters={}):
        """
        Compute the model for the large-scales biasing of delta_flux.

        Parameters:
            mu (float): Cosine of the angle between the line-of-sight and the wavevector.
            parameters (dict, optional): Additional parameters for the model. Defaults to {}.

        Returns:
            float: Computed value of the large-scale biasing of delta_flux.
        """

        # extract bias and beta from dictionary with parameter values
        list_check = ["beta", "bias"]
        pp = self.check_params(list_check, parameters)

        linear_rsd = 1 + pp["beta"] * mu**2

        return pp["bias"] * linear_rsd

    def small_scales_correction(self, z, k, mu, parameters={}):
        """
        Compute the small-scale correction to delta_flux biasing.

        Parameters:
            z (float): Redshift.
            k (float): Wavenumber.
            mu (float): Cosine of the angle between the line-of-sight and the wavevector.
            parameters (dict, optional): Additional parameters for the model. Defaults to {}.

        Returns:
            float: Computed value of the small-scale correction to delta_flux biasing.
        """

        delta2 = self.delta2(z, k)
        nonlin = self.small_nonlin(delta2, parameters)
        vel = self.small_vel(k, mu, parameters)
        press = self.small_press(k, parameters)

        d1 = np.exp(nonlin * (1 - vel) - press)

        return d1

    def delta2(self, z, k):
        # get linear power (required to get delta squared)
        # evaluate linear power at input (z,k)
        linP = self.linP_Mpc(z, k)

        delta2 = (1 / (2 * np.pi**2)) * k**3 * linP
        return delta2

    def small_nonlin(self, delta2, parameters={}):
        # extract parameters from dictionary of parameter values
        list_check = ["q1", "q2"]
        pp = self.check_params(list_check, parameters)
        res = delta2 * (pp["q1"] + pp["q2"] * delta2)
        return res

    def small_vel(self, k, mu, parameters={}):
        # extract parameters from dictionary of parameter values
        list_check = ["av", "kvav", "bv"]
        pp = self.check_params(list_check, parameters)
        vel = k ** pp["av"] / pp["kvav"] * mu ** pp["bv"]
        return vel

    def small_press(self, k, parameters={}):
        # extract parameters from dictionary of parameter values
        list_check = ["kp"]
        pp = self.check_params(list_check, parameters)
        press = (k / pp["kp"]) ** 2
        return press

    def check_params(self, list_check, parameters):
        parameters_check = {}
        for key in list_check:
            if key in parameters:
                parameters_check[key] = parameters[key]
            else:
                parameters_check[key] = self.default_params[key]
        return parameters_check

    def dp3d_dbias_p3d(self, parameters={}):
        list_check = ["bias"]
        pp = self.check_params(list_check, parameters)
        res_p3d = 2 / pp["bias"]
        return res_p3d

    def dp3d_dbeta_p3d(self, mu, parameters={}):
        list_check = ["beta"]
        pp = self.check_params(list_check, parameters)
        res = 2 * mu**2 / (1 + pp["beta"] * mu**2)
        return res

    def dp3d_dq1_p3d(self, k, mu, delta2, parameters={}):
        res = delta2 * (1 - self.small_vel(k, mu, parameters=parameters))
        return res

    def dp3d_dq2_p3d(self, k, mu, delta2, parameters={}):
        res = delta2**2 * (1 - self.small_vel(k, mu, parameters=parameters))
        return res

    def dp3d_dav_p3d(self, k, mu, delta2, parameters={}):
        nonlin = self.small_nonlin(delta2, parameters=parameters)
        nonvel = self.small_vel(k, mu, parameters=parameters)
        res = -nonlin * nonvel * np.log(k)
        return res

    def dp3d_dkvav_p3d(self, k, mu, delta2, parameters={}):
        list_check = ["kvav"]
        pp = self.check_params(list_check, parameters)
        nonlin = self.small_nonlin(delta2, parameters=parameters)
        nonvel = self.small_vel(k, mu, parameters=parameters)
        res = nonlin * nonvel / pp["kvav"]
        return res

    def dp3d_dbv_p3d(self, k, mu, delta2, parameters={}):
        nonlin = self.small_nonlin(delta2, parameters=parameters)
        nonvel = self.small_vel(k, mu, parameters=parameters)
        res = -nonlin * nonvel * np.log(mu)
        return res

    def dp3d_dkp_p3d(self, k, parameters={}):
        list_check = ["kp"]
        pp = self.check_params(list_check, parameters)
        nonpress = self.small_press(k, parameters=parameters)
        res = 2 * nonpress / pp["kp"]
        return res

    def _P3D_kperp2(self, z, ln_k_perp, k_par, parameters={}):
        """
        Function to be integrated to compute P1D.

        Parameters:
            z (float): Redshift.
            ln_k_perp (float): Natural logarithm of the perpendicular wavenumber.
            k_par (float): Parallel wavenumber.
            parameters (dict, optional): Additional parameters for the model. Defaults to {}.

        Returns:
            float: Computed value of the integrated function.
        """

        # compute k and mu from ln_k_perp and k_par
        k_perp = np.exp(ln_k_perp)
        k = np.sqrt(k_par**2 + k_perp**2)
        mu = k_par / k

        # get P3D
        p3d = self.P3D_Mpc(z, k, mu, parameters)

        return (1 / (2 * np.pi)) * k_perp**2 * p3d

    def _P1D_lnkperp(self, z, ln_k_perp, kpars, parameters={}):
        """
        Compute P1D by integrating P3D in terms of ln(k_perp).

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

        # for each value of k_par, integrate P3D over ln(k_perp) to get P1D
        p1d = np.empty_like(kpars)
        for i in range(kpars.size):
            # get function to be integrated
            p3d_fix_k_par = self._P3D_kperp2(z, ln_k_perp, kpars[i], parameters)
            # perform numerical integration
            p1d[i] = simpson(p3d_fix_k_par, ln_k_perp, dx=dlnk)

        return p1d

    def _P1D_lnkperp_fast(self, z, ln_k_perp, kpars, parameters={}):
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

        p3d_fix_k_par = self.P3D_Mpc(z, k, mu, parameters) * fact

        # perform numerical integration
        p1d = simpson(p3d_fix_k_par, ln_k_perp, dx=dlnk, axis=1)

        return p1d

    def _rat_P1D_lnkperp_fast(self, z, ln_k_perp, kpars, parameters={}):
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
        p3d_fix_k_par = self.rat_P3D(z, k, mu, parameters) * fact

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

    def P1D_Mpc(
        self,
        z,
        k_par,
        k_perp_min=0.001,
        k_perp_max=100,
        n_k_perp=99,
        parameters={},
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

        p1d = self._P1D_lnkperp_fast(z, ln_k_perp, k_par, parameters)

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

    def rel_err_P1d(
        self,
        z,
        k_par,
        Lbox,
        res_los,
        nskewers,
        n_k_perp=1000,
        parameters={},
    ):
        """
        Computes the relative error of P1D.

        Parameters:
            z (float): Redshift.
            k_par (array-like): Array or list of values for which P1D is to be computed.
            Lbox (float): Size of the simulation box.
            res_los (float): Resolution of the LOS direction.
            nskewers (int): Number of skewers used for the measurement.
            n_k_perp (int, optional): Number of points in the k_perp integration. Defaults to 1000.
            parameters (dict, optional): Additional parameters for the model. Defaults to {}.

        Returns:
            tuple: A tuple containing:
                - sigma_both (array-like): Estimated error combining cosmic variance and sampling variance.
                - sigma_p1d__p1d (array-like): Estimated error from cosmic variance.
                - sampling_sigma__p1d (array-like): Estimated error from sampling variance.
        """
        n_k_par = k_par.shape[0]

        # array of k_perp that fit within simulation box (important!)
        nk_max = int(Lbox / res_los / 2)
        k_perp_min = 2 * np.pi / Lbox
        k_perp_max = k_perp_min * nk_max
        ln_k_perp = np.linspace(
            np.log(k_perp_min), np.log(k_perp_max), n_k_perp
        )
        k_perp = np.exp(ln_k_perp)

        # get p3d and number of modes as a function of k_par and k_per
        p3d = np.zeros((n_k_par, n_k_perp))
        nmod = np.zeros((n_k_par, n_k_perp))
        for ii in range(n_k_perp):
            k = np.sqrt(k_par**2 + k_perp[ii] ** 2)
            dk = k[1:] - k[:-1]
            dk = np.concatenate([dk, np.atleast_1d(dk[-1])])
            mu = k_par / k
            p3d[:, ii] = self.P3D_Mpc(z, k, mu)
            nmod[:, ii] = get_nmod(k, dk, Lbox)

        ## estimate error from cosmic variance (new, JCM)

        # sigma_p3d = np.sqrt(2/Nmodes) * P
        sigma_p3d = np.sqrt(2 / nmod) * p3d

        # p1d = 1/(2*pi) int_kmin^kmax dkper kper p3d
        # I am not sure whether should be pi or 2*pi, depends on the convention?
        # we use 2*pi above so let's go with that
        p1d = self.P1D_Mpc(z, k_par)

        # in practise, we only need to evaluate the derivatives and error at kper=kmax
        # dp1d/dp3d = dP1d/dkpar (dkpar/dp3d)_kmin^kmax
        # dp1d_dkpar = (1/pi) int_kmin^kmax dkper kper dp3d/dkpar
        dp3d_dkpar = np.gradient(p3d, k_par, axis=0)
        dp1d_dkpar = simpson(dp3d_dkpar * k_perp, k_perp, axis=1) / (2 * np.pi)
        dp1d_dp3d = dp1d_dkpar / dp3d_dkpar[:, -1]

        # err_p1d = |dp1d/dp3d| * sigma_p3d_kmin^kmax
        sigma_p1d = np.abs(dp1d_dp3d) * sigma_p3d[:, -1]

        # err_p1d__p1d
        sigma_p1d__p1d = sigma_p1d / p1d

        ## estimate error from sampling variance (Zhan et al. 2005)
        sampling_sigma__p1d = (k_par[:] * 0 + 1) / np.sqrt(nskewers)

        ## combine both
        sigma_both = np.sqrt(sigma_p1d__p1d**2 + sampling_sigma__p1d**2)

        return sigma_both, sigma_p1d__p1d, sampling_sigma__p1d
