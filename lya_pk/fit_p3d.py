import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import emcee
import copy

from ForestFlow.utils import purge_chains, init_chains, params_numpy2dict
from ForestFlow.plot_routines import plot_template


class FitPk(object):
    """
    Fit measured P3D with Arinyo model.

    Attributes:
        data (dict): Measured P3D data.
        model (object): Theoretical model for 3D flux power.
        fit_type (str): Fit type ('p3d' or 'both').
        k3d_max (float): Maximum 3D wavenumber for the fit.
        k1d_max (float): Maximum 1D wavenumber for the fit.
        noise_3d (float): Noise level for 3D flux power.
        noise_1d (float): Noise level for 1D flux power.
        priors (dict): Prior information for the fit.
        units (str): Units of the data.

    Methods:
        __init__: Initializes the FitPk object.
        get_model_3d: Computes the model for the 3D flux power spectrum.
        get_model_1d: Computes the model for the 1D flux power spectrum.
        get_chi2: Computes the chi-squared for a particular P3D model.
        get_log_like: Computes the log likelihood ignoring determinant.
        _log_like: Function passed to scipy minimizer to compute the log likelihood.
        maximize_likelihood: Run minimizer and return the best-fit values.
        log_like_emcee: Compute the log likelihood for emcee sampling.
        explore_likelihood: Run emcee to explore the likelihood and return the best-fitting chain.
        old_smooth_err_pkmu: Return P3D(k, mu) errors estimated from simulation.
        old_estimate_err_p1d: Returns P1D(k, mu) errors estimated from simulation.
        plot_fits: Compare data and best-fitting model.
    """

    def __init__(
        self,
        data,
        model,
        fit_type="p3d",
        k3d_max=10,
        k1d_max=10,
        noise_3d=0.05,
        noise_1d=0.05,
        priors=None,
        verbose=False,
    ):
        """
        Setup P3D flux power model and measurement.

        Args:
            data (dict): Measured P3D data.
            model (object): Theoretical model for 3D flux power.
            fit_type (str, optional): Fit type ('p3d' or 'both'). Defaults to 'p3d'.
            k3d_max (float, optional): Maximum 3D wavenumber for the fit. Defaults to 10.
            k1d_max (float, optional): Maximum 1D wavenumber for the fit. Defaults to 10.
            noise_3d (float, optional): Noise level for 3D flux power. Defaults to 0.05.
            noise_1d (float, optional): Noise level for 1D flux power. Defaults to 0.05.
            priors (dict, optional): Prior information for the fit. Defaults to None.

        Modifies:
            data (dict): Updates the data with standard deviations and fit indices.

        Returns:
            None
        """

        # store data and model
        self.verbose = verbose
        self.data = copy.deepcopy(data)
        self.model = model
        self.priors = priors
        self.units = self.data["units"]

        # p3d or both
        self.fit_type = fit_type

        # relative errors with noise floor
        self.data["std_p3d_sta"] = self.data["std_p3d"] * 1
        self.data["std_p3d_sys"] = noise_3d * self.data["p3d"]
        self.data["std_p3d"] = np.sqrt(
            self.data["std_p3d_sta"] ** 2 + self.data["std_p3d_sys"] ** 2
        )
        # k_Mpc < fit_k_Mpc_max
        self.ind_fit3d = (
            np.isfinite(self.data["std_p3d"])
            & (self.data["std_p3d"] != 0)
            & np.isfinite(self.data["p3d"])
            & (self.data["k3d"] < k3d_max)
        )
        self.data["std_p3d"][~self.ind_fit3d] = np.inf

        # same for p1d
        if fit_type == "both":
            # relative errors with noise floor
            self.data["std_p1d_sta"] = self.data["std_p1d"] * 1
            self.data["std_p1d_sys"] = noise_1d * self.data["p1d"]
            self.data["std_p1d"] = np.sqrt(
                self.data["std_p1d_sta"] ** 2 + self.data["std_p1d_sys"] ** 2
            )
            # k_Mpc < fit_k_Mpc_max
            self.ind_fit1d = (
                np.isfinite(self.data["std_p1d"])
                & (self.data["std_p1d"] != 0)
                & np.isfinite(self.data["p1d"])
                & (self.data["k1d"] < k1d_max)
            )
            self.data["std_p1d"][~self.ind_fit1d] = np.inf

    def get_model_3d(self, parameters={}):
        """
        Computes the model for the 3D flux power spectrum.

        Args:
            parameters (dict, optional): Model parameters. Defaults to {}.

        Returns:
            ndarray: Model for the 3D flux power spectrum.
        """

        # identify (k,mu) for bins included in fit
        p3d = self.model.P3D_Mpc(
            self.data["z"][0],
            self.data["k3d"],
            self.data["mu3d"],
            parameters,
        )
        if self.units == "N":
            p3d *= self.data["k3d"] ** 3 / 2 / np.pi**2
        return p3d

    def get_model_1d(
        self,
        parameters={},
        k_perp_min=0.001,
        k_perp_max=60,
        n_k_perp=40,
    ):
        """
        Computes the model for the 1D flux power spectrum.

        Args:
            parameters (dict, optional): Model parameters. Defaults to {}.
            k_perp_min (float, optional): Minimum perpendicular wavenumber. Defaults to 0.001.
            k_perp_max (float, optional): Maximum perpendicular wavenumber. Defaults to 60.
            n_k_perp (int, optional): Number of perpendicular wavenumbers. Defaults to 40.

        Returns:
            ndarray: Model for the 1D flux power spectrum.
        """
        p1d = self.model.P1D_Mpc(
            self.data["z"][0],
            self.data["k1d"],
            parameters=parameters,
            k_perp_min=k_perp_min,
            k_perp_max=k_perp_max,
            n_k_perp=n_k_perp,
        )
        if self.units == "N":
            p1d *= self.data["k1d"] / np.pi
        return p1d

    def get_chi2(self, parameters={}, return_npoints=False):
        """
        Compute chi squared for a particular P3D model.

        Args:
            parameters (dict, optional): Dictionary with parameters to use. Defaults to {}.
            return_npoints (bool, optional): Whether to return the number of data points used. Defaults to False.

        Returns:
            float or tuple: Chi squared value or tuple containing chi squared and number of data points.
        """

        # get P3D measurement for bins included in fit
        data_p3d = self.data["p3d"][self.ind_fit3d]

        # compute model for these wavenumbers
        th_p3d = self.get_model_3d(parameters=parameters)[self.ind_fit3d]

        # get absolute error
        err_p3d = self.data["std_p3d"][self.ind_fit3d]

        # compute chi2
        chi2 = np.sum(((data_p3d - th_p3d) / err_p3d) ** 2) / np.sum(
            self.ind_fit3d
        )

        if self.fit_type == "both":
            # get P1D measurement for bins included in fit
            data_p1d = self.data["p1d"][self.ind_fit1d]

            # compute model for these wavenumbers
            th_p1d = self.get_model_1d(parameters=parameters)[self.ind_fit1d]

            # compute absolute error
            err_p1d = self.data["std_p1d"][self.ind_fit1d]

            # compute chi2
            chi2_p1d = np.sum(((data_p1d - th_p1d) / err_p1d) ** 2) / np.sum(
                self.ind_fit1d
            )
            chi2 += chi2_p1d

        if return_npoints:
            npoints = len(data)
            return chi2, npoints
        else:
            return chi2

    def get_log_like(self, parameters={}):
        """
        Compute log likelihood (ignoring determinant).

        Args:
            parameters (dict, optional): Dictionary with parameters to use. Defaults to {}.

        Returns:
            float: Log likelihood value.
        """

        return -0.5 * self.get_chi2(parameters, return_npoints=False)

    def _log_like(self, values, parameter_names):
        """
        Function passed to scipy minimizer to compute the log likelihood.

        Args:
            values (array): Array of initial values of parameters.
            parameter_names (list): List of parameter names (should have the same size as values).

        Returns:
            float: Log likelihood value.
        """

        Np = len(values)
        assert Np == len(parameter_names), "inconsistent inputs in _log_like"

        # create dictionary with parameters that models can understand
        # also check if all parameters within priors
        out_priors = 0
        parameters = {}
        for ii in range(Np):
            parameters[parameter_names[ii]] = values[ii]
            if self.priors is not None:
                if (values[ii] < self.priors[parameter_names[ii]][0]) | (
                    values[ii] > self.priors[parameter_names[ii]][1]
                ):
                    out_priors += 1

        if out_priors != 0:
            return -np.inf
        else:
            return self.get_log_like(parameters)

    def maximize_likelihood(self, parameters):
        """
        Run minimizer and return the best-fit values.

        Args:
            parameters (dict): Dictionary of parameters.

        Returns:
            tuple: Tuple containing the optimization results and the best-fit parameters.
        """

        ndim = len(parameters)
        names = list(parameters.keys())
        params_in = np.array(list(parameters.values()))

        # lambda function to minimize
        minus_log_like = lambda *args: -self._log_like(*args)

        chi2_in = self.get_chi2(params_numpy2dict(params_in))

        for it in range(10):
            if it != 0:
                chi2_in = chi2_out * 1

            results = minimize(
                minus_log_like,
                params_in,
                args=(names),
                method="BFGS",
            )

            params_in = results.x
            chi2_out = self.get_chi2(params_numpy2dict(params_in))

            if np.abs(chi2_in - chi2_out) < 0.1:
                break

        if self.verbose:
            print("iterations", it)

        best_fit_parameters = params_numpy2dict(params_in)

        return results, best_fit_parameters

    def log_like_emcee(self, params):
        """
        Compute the log likelihood for emcee sampling.

        Args:
            params (array): Array of parameter values.

        Returns:
            float: Log likelihood value.
        """

        out_priors = 0
        parameters = {}
        for ii in range(self.ndim):
            parameters[self.names[ii]] = params[ii]
            if (params[ii] < self.priors[self.names[ii]][0]) | (
                params[ii] > self.priors[self.names[ii]][1]
            ):
                out_priors += 1

        if out_priors != 0:
            return -np.inf
        else:
            return -0.5 * self.get_chi2(parameters, return_npoints=False)

    def explore_likelihood(
        self,
        parameters,
        seed=0,
        nwalkers=20,
        nsteps=100,
        nburn=0,
        plot=False,
        attraction=0.3,
    ):
        """
        Run emcee to explore the likelihood and return the best-fitting chain.

        Args:
            parameters (dict): Dictionary of parameters.
            seed (int): Seed for random number generation (default: 0).
            nwalkers (int): Number of walkers in the ensemble (default: 20).
            nsteps (int): Number of steps to run the sampler (default: 100).
            nburn (int): Number of burn-in steps to discard (default: 0).
            plot (bool): Flag to plot the ln(probability) for each walker (default: False).
            attraction (float): Attraction parameter for the initial walker positions (default: 0.3).

        Returns:
            tuple: Tuple containing the ln(probability) and chain arrays.
        """

        self.ndim = len(parameters)
        self.names = list(parameters.keys())
        values = np.array(list(parameters.values()))

        # generate random initial value
        ini_values = init_chains(
            parameters,
            nwalkers,
            self.priors,
            seed=seed,
            attraction=attraction,
        )

        sampler = emcee.EnsembleSampler(
            nwalkers,
            self.ndim,
            self.log_like_emcee,
        )
        sampler.run_mcmc(ini_values, nsteps)

        chain = sampler.get_chain(discard=nburn)
        lnprob = sampler.get_log_prob(discard=nburn)

        # print(chain.shape, lnprob.shape)
        # (nsteps, nwalkers, ndim) (nsteps, nwalkers)

        if plot:
            for ii in range(nwalkers):
                plt.plot(lnprob[:, ii])
            plt.show()

        minval = np.nanmedian(lnprob, axis=0)
        minval = np.nanmax(minval) - 5
        keep = purge_chains(lnprob, minval=minval)

        chain = chain[:, keep, :].reshape(-1, self.ndim)
        lnprob = lnprob[:, keep].reshape(-1)

        return lnprob, chain

    def old_smooth_err_pkmu(self, kmax=10, order=2):
        """
        Return P3D(k, mu) errors estimated from simulation.

        Args:
            kmax (float): Maximum value of k (default: 10).
            order (int): Order of the polynomial fit (default: 2).

        Returns:
            None
        """

        self.data["fcov3d"] = np.diag(self.data["cov3d"]).reshape()
        fit_epk3d = np.zeros((self.data["k3d"].shape[1], order + 1))
        for ii in range(self.data["k3d"].shape[1]):
            _ = np.isfinite(self.data["k3d"][:, ii]) & (
                self.data["k3d"][:, ii] < kmax
            )
            logk = np.log10(self.data["k3d"][_, ii])
            self.fit_epk3d[ii, :] = np.polyfit(
                logk, np.log10(sigma_pkmu[_, ii]), order
            )
            pfit = np.poly1d(self.fit_epk3d[ii, :])

            _ = np.isfinite(self.data["k3d"][:, ii])
            logk = np.log10(self.data["k3d"][_, ii])
            noise_floor = noise_3d * self.data["p3d"][_, ii]
            self.err_p3d[_, ii] = 10 ** pfit(logk) + noise_floor

    def old_estimate_err_p1d(self, sigma_pk1d, order=3, kmax=40, noise_1d=0.05):
        """
        Returns P1D(k, mu) errors estimated from simulation

        Parameters:
            sigma_pk1d (array): Array of P1D errors.
            order (int): Polynomial fit order (default: 3).
            kmax (float): Maximum value of k (default: 40).
            noise_1d (float): Noise level for P1D (default: 0.05).

        Returns:
            array: Array of P1D(k, mu) errors.

        Note:
            - The method checks if "k1d" key exists in self.data. If not, it returns "No k_Mpc_1d key in data".
            - The method calculates the P1D(k, mu) errors based on the provided inputs and stores the result in self.err_p1d.
        """

        if "k1d" in self.data:
            pass
        else:
            return "No k_Mpc_1d key in data"

        self.err_p1d = np.zeros_like(self.data["p1d"])
        _ = (
            (self.data["k1d"] > 0)
            & (self.data["k1d"] < kmax)
            & (sigma_pk1d != 0)
        )
        logk = np.log10(self.data["k1d"][_])
        self.fit_epk1d = np.polyfit(logk, np.log10(sigma_pk1d[_]), order)
        pfit = np.poly1d(self.fit_epk1d)

        _ = self.data["k1d"] > 0
        logk = np.log10(self.data["k1d"][_])
        noise_floor = noise_1d * self.data["p1d"][_]
        self.err_p1d[_] = 10 ** pfit(logk) + noise_floor
        self.err_p1d[0] = self.err_p1d[1] * 2

    def plot_fits(
        self,
        parameters,
        error_fit_3d=None,
        error_fit_1d=None,
        save_fig=None,
        err_bar_all=False,
        plot_emu=False,
    ):
        """
        Compare data and best-fitting model.

        Parameters:
            parameters (dict): Dictionary of fitting parameters.
            error_fit_3d (array, optional): Array of 3D fitting errors (default: None).
            error_fit_1d (array, optional): Array of 1D fitting errors (default: None).
            save_fig (str, optional): File path to save the figure (default: None).
            err_bar_all (bool, optional): Flag to enable error bars for all data points (default: False).

        Note:
            - This method compares the data and the best-fitting model.
            - It plots the comparison using subplots for 3D fitting, 1D fitting, and cosmic variance errors.
            - The `parameters` argument should be a dictionary of fitting parameters required to compute the model.
            - The `error_fit_3d` and `error_fit_1d` arrays provide the fitting errors for the 3D and 1D data, respectively.
            - If `error_fit_3d` is provided and `err_bar_all` is True, error bars will be shown for all data points in the 3D fitting plot.
            - If `error_fit_1d` is provided and `err_bar_all` is True, error bars will be shown for all data points in the 1D fitting plot.
            - If `save_fig` is provided, the plot will be saved to the specified file path.

        Returns:
            None
        """

        fig, ax = plt.subplots(3, sharex=True, figsize=(8, 6))

        # compute best-fitting model
        p3d_best = self.get_model_3d(parameters=parameters)

        p1d_best = self.get_model_1d(parameters=parameters)

        # compute linear power spectrum (for plotting)
        linp = (
            self.model.linP_Mpc(z=self.data["z"][0], k_Mpc=self.data["k3d"])
            * self.data["k3d"] ** 3
            / 2
            / np.pi**2
        )

        # iterate over wedges
        nmus = self.data["k3d"].shape[1]
        mubins = np.linspace(0, 1, nmus + 1)
        mu_use = np.linspace(0, nmus - 1, 4, dtype=int)
        for imu, ii in enumerate(mu_use):
            col = "C" + str(imu)

            mutag = (
                str(np.round(mubins[ii], 2))
                + r"$\leq\mu\leq$"
                + str(np.round(mubins[ii + 1], 2))
            )

            ## PF/Plin ##
            if imu == 0:
                if plot_emu == False:
                    lab0a = r"$X=$Arinyo"
                    lab0b = r"$X=$Data"
                else:
                    lab0a = r"$X=$Emulated P3D"
                    lab0b = r"$X=$Data"
            else:
                lab0a = None
                lab0b = None

            # only plot when data is not nan
            mask = self.ind_fit3d[:, ii]

            # Pbest/Plin #
            rat = p3d_best[mask, ii] / linp[mask, ii]

            if error_fit_3d is not None:
                err = error_fit_3d[mask, ii] / linp[mask, ii]
                ax[0].errorbar(
                    self.data["k3d"][mask, ii],
                    rat,
                    err,
                    color=col,
                    ls="-",
                    label=lab0a,
                )
            else:
                ax[0].plot(
                    self.data["k3d"][mask, ii],
                    rat,
                    color=col,
                    ls="-",
                    label=lab0a,
                )

            # Pdata/Plin #

            av_err = self.data["std_p3d"][mask, ii]
            yy0 = (self.data["p3d"][mask, ii] - av_err) / linp[mask, ii]
            yy1 = (self.data["p3d"][mask, ii] + av_err) / linp[mask, ii]
            ax[0].fill_between(
                self.data["k3d"][mask, ii],
                yy0,
                yy1,
                color=col,
                alpha=0.25,
            )
            rat = self.data["p3d"][mask, ii] / linp[mask, ii]
            ax[0].plot(
                self.data["k3d"][mask, ii],
                rat,
                color=col,
                label=lab0b,
                ls=":",
                marker=".",
            )

            ## PF/PFmodel ##
            rat = self.data["p3d"][mask, ii] / p3d_best[mask, ii]

            if (error_fit_3d is not None) & err_bar_all:
                err = error_fit_3d[mask, ii] / p3d_best[mask, ii]

                ax[1].errorbar(
                    self.data["k3d"][mask, ii],
                    rat,
                    err,
                    color=col,
                    label=mutag,
                )
            else:
                ax[1].plot(
                    self.data["k3d"][mask, ii],
                    rat,
                    color=col,
                    label=mutag,
                )

        ###
        # plot cosmic variance errors
        ii = 0
        iax = 1
        err_sta = self.data["std_p3d_sta"][mask, ii] / p3d_best[mask, ii]
        ax[iax].fill_between(
            self.data["k3d"][mask, ii],
            -err_sta + 1,
            y2=err_sta + 1,
            color="k",
            alpha=0.25,
        )
        err_sys = self.data["std_p3d_sys"][mask, ii] / p3d_best[mask, ii]
        ax[iax].fill_between(
            self.data["k3d"][mask, ii],
            -err_sys + 1,
            y2=err_sys + 1,
            color="k",
            alpha=0.1,
            hatch="/",
        )

        ## P1D ##
        mask = self.ind_fit1d
        rat = self.data["p1d"][mask] / p1d_best[mask]

        if (error_fit_1d is not None) & err_bar_all:
            err = error_fit_1d[mask] / p1d_best[mask]
            ax[2].errorbar(
                self.data["k1d"][mask],
                rat,
                err,
                alpha=0.2,
            )
        else:
            ax[2].plot(self.data["k1d"][mask], rat, "C0")

        # plot cosmic variance errors
        err_sta = self.data["std_p1d_sta"][mask] / p1d_best[mask]
        ax[2].fill_between(
            self.data["k1d"][mask],
            -err_sta + 1,
            y2=err_sta + 1,
            color="k",
            alpha=0.25,
        )
        err_sys = self.data["std_p1d_sys"][mask] / p1d_best[mask]
        ax[2].fill_between(
            self.data["k1d"][mask],
            -err_sys + 1,
            y2=err_sys + 1,
            color="k",
            alpha=0.1,
            hatch="/",
        )

        ax[iax].set_ylim([0.7, 1.3])
        # plot expected precision lines
        ax[iax].axhline(1, color="k", ls=":")
        ax[iax].axhline(1.1, color="k", ls="--")
        ax[iax].axhline(0.9, color="k", ls="--")
        # ax[iax].axvline(x=kmax_3d, color="k")

        ax[iax].set_xscale("log")
        if plot_emu == True:
            ylab = r"$P_\mathrm{F}^\mathrm{Emu}/P_\mathrm{F}^\mathrm{Arinyo}$"
        else:
            ylab = r"$P_\mathrm{F}^\mathrm{Data}/P_\mathrm{F}^\mathrm{Arinyo}$"

        plot_template(
            ax[iax],
            legend_loc="upper right",
            ylabel=ylab,
            ftsize=19,
            ftsize_legend=13,
            legend=0,
            legend_columns=1,
        )

        ####
        iax = 0
        # ax[iax].set_ylim([-0.005, 0.08])
        # ax[iax].axvline(x=kmax_3d, color="k")
        ax[iax].set_xscale("log")

        plot_template(
            ax[iax],
            legend_loc="upper right",
            ylabel=r"$P_\mathrm{F}^X/P_\mathrm{L}$",
            ftsize=19,
            ftsize_legend=17,
            legend=0,
            legend_columns=1,
        )

        ####
        iax = 2
        # plot expected precision lines
        ax[iax].axhline(1, color="k", ls=":")
        ax[iax].axhline(1.01, color="k", ls="--")
        ax[iax].axhline(0.99, color="k", ls="--")
        ax[iax].set_ylim([0.9, 1.1])
        # ax[iax].axvline(x=kmax_1d, color="k")
        ax[iax].set_xscale("log")

        if plot_emu == True:
            ylab = r"$P_{1D}^\mathrm{Emu}/P_{1D}^\mathrm{Arinyo}$"
        else:
            ylab = r"$P_{1D}^\mathrm{Data}/P_{1D}^\mathrm{Arinyo}$"

        plot_template(
            ax[iax],
            #     legend_loc="upper right",
            xlabel=r"$k\,\left[\mathrm{Mpc}^{-1}\right]$",
            ylabel=ylab,
            ftsize=19,
            #     ftsize_legend=17,
            #     legend=0,
            #     legend_columns=1,
        )

        ax[iax].set_xlim(self.data["k3d"][0, 0] * 0.9, 25)
        plt.tight_layout()

        if save_fig is not None:
            plt.savefig(save_fig)

    def plot_compare_smooth(
        self,
        parameters1,
        parameters2=None,
        error_fit_3d=None,
        error_fit_1d=None,
        save_fig=None,
        err_bar_all=False,
    ):
        """
        Compare data and best-fitting model.

        Parameters:
            parameters (dict): Dictionary of fitting parameters.
            error_fit_3d (array, optional): Array of 3D fitting errors (default: None).
            error_fit_1d (array, optional): Array of 1D fitting errors (default: None).
            save_fig (str, optional): File path to save the figure (default: None).
            err_bar_all (bool, optional): Flag to enable error bars for all data points (default: False).

        Note:
            - This method compares the data and the best-fitting model.
            - It plots the comparison using subplots for 3D fitting, 1D fitting, and cosmic variance errors.
            - The `parameters` argument should be a dictionary of fitting parameters required to compute the model.
            - The `error_fit_3d` and `error_fit_1d` arrays provide the fitting errors for the 3D and 1D data, respectively.
            - If `error_fit_3d` is provided and `err_bar_all` is True, error bars will be shown for all data points in the 3D fitting plot.
            - If `error_fit_1d` is provided and `err_bar_all` is True, error bars will be shown for all data points in the 1D fitting plot.
            - If `save_fig` is provided, the plot will be saved to the specified file path.

        Returns:
            None
        """

        fig, ax = plt.subplots(
            2,
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
            figsize=(8, 6),
        )

        # compute best-fitting model
        p3d_best1 = self.get_model_3d(parameters=parameters1)
        p1d_best1 = self.get_model_1d(parameters=parameters1)
        if parameters2 is not None:
            p3d_best2 = self.get_model_3d(parameters=parameters2)
            p1d_best2 = self.get_model_1d(parameters=parameters2)

        # iterate over wedges
        nmus = self.data["k3d"].shape[1]
        mubins = np.linspace(0, 1, nmus + 1)
        mu_use = np.linspace(0, nmus - 1, 4, dtype=int)
        for imu, ii in enumerate(mu_use):
            col = "C" + str(imu)

            mutag = (
                str(np.round(mubins[ii], 2))
                + r"$\leq\mu\leq$"
                + str(np.round(mubins[ii + 1], 2))
            )

            ## P2/P1 ##
            if imu == 0:
                if plot_emu == False:
                    lab0a = r"$X=$Fit"
                    lab0b = r"$X=$Emulator"
                else:
                    lab0a = r"$X=$Emulated P3D"
                    lab0b = r"$X=$Data"
            else:
                lab0a = None
                lab0b = None

            # only plot when data is not nan
            mask = self.ind_fit3d[:, ii]

            # Pbest/Plin #
            ax[0].plot(
                self.data["k3d"][mask, ii],
                p3d_best1[mask, ii],
                color=col,
                ls="-",
                label=lab0a,
            )
            ax[0].plot(
                self.data["k3d"][mask, ii],
                p3d_best2[mask, ii],
                color=col,
                ls="-",
                label=lab0a,
            )

            ax[1].plot(
                self.data["k3d"][mask, ii],
                p3d_best2[mask, ii] / p3d_best1[mask, ii],
                color=col,
                ls="-",
                label=lab0a,
            )

            # if error_fit_3d is not None:
            #     err = error_fit_3d[mask, ii] / linp[mask, ii]
            #     ax[0].errorbar(
            #         self.data["k3d"][mask, ii],
            #         rat,
            #         err,
            #         color=col,
            #         ls="-",
            #         label=lab0a,
            #     )
            # else:
            #     ax[0].plot(
            #         self.data["k3d"][mask, ii],
            #         rat,
            #         color=col,
            #         ls="-",
            #         label=lab0a,
            #     )

        ###
        # plot cosmic variance errors
        ii = 0
        iax = 1
        err_sta = self.data["std_p3d_sta"][mask, ii] / p3d_best[mask, ii]
        ax[iax].fill_between(
            self.data["k3d"][mask, ii],
            -err_sta + 1,
            y2=err_sta + 1,
            color="k",
            alpha=0.25,
        )
        err_sys = self.data["std_p3d_sys"][mask, ii] / p3d_best[mask, ii]
        ax[iax].fill_between(
            self.data["k3d"][mask, ii],
            -err_sys + 1,
            y2=err_sys + 1,
            color="k",
            alpha=0.1,
            hatch="/",
        )

        iax = 0
        ax[iax].set_xscale("log")
        ax[iax].set_ylim([0.7, 1.3])
        # plot expected precision lines
        ax[iax].axhline(1, color="k", ls=":")
        ax[iax].axhline(1.1, color="k", ls="--")
        ax[iax].axhline(0.9, color="k", ls="--")
        # ax[iax].axvline(x=kmax_3d, color="k")

        ax[iax].set_xscale("log")
        ylab = r"$P_\mathrm{F}^\mathrm{Data}/P_\mathrm{F}^\mathrm{Arinyo}$"
        plot_template(
            ax[iax],
            legend_loc="upper right",
            ylabel=ylab,
            ftsize=19,
            ftsize_legend=13,
            legend=0,
            legend_columns=1,
        )

        ####
        iax = 0
        # ax[iax].set_ylim([-0.005, 0.08])
        # ax[iax].axvline(x=kmax_3d, color="k")

        plot_template(
            ax[iax],
            legend_loc="upper right",
            ylabel=r"$P_\mathrm{F}^X/P_\mathrm{L}$",
            ftsize=19,
            ftsize_legend=17,
            legend=0,
            legend_columns=1,
        )

        ####
        iax = 1
        # plot expected precision lines
        ax[iax].axhline(1, color="k", ls=":")
        ax[iax].axhline(1.01, color="k", ls="--")
        ax[iax].axhline(0.99, color="k", ls="--")
        ax[iax].set_ylim([0.97, 1.03])
        # ax[iax].axvline(x=kmax_1d, color="k")
        ax[iax].set_xscale("log")

        plot_template(
            ax[iax],
            #     legend_loc="upper right",
            xlabel=r"$k\,\left[\mathrm{Mpc}^{-1}\right]$",
            ylabel=r"$P_{1D}^\mathrm{Data}/P_{1D}^\mathrm{Arinyo}$",
            ftsize=19,
            #     ftsize_legend=17,
            #     legend=0,
            #     legend_columns=1,
        )

        ax[iax].set_xlim(self.data["k3d"][0, 0] * 0.9, 25)
        plt.tight_layout()

        if save_fig is not None:
            plt.savefig(save_fig)
