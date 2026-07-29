import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from scipy.optimize import minimize
from dataclasses import dataclass
from forestflow.model_p3d_arinyo import ArinyoModel
from lace.cosmo import cosmology


@dataclass(slots=True)
class FitData:

    # cosmology
    z: float
    linear: dict

    # model
    power_model: ArinyoModel

    # params
    ini_params: dict

    # 1D
    k1d: np.ndarray
    p1d: np.ndarray
    std_p1d: np.ndarray

    # 3D
    k3d: np.ndarray
    mu3d: np.ndarray
    p3d: np.ndarray
    std_p3d: np.ndarray


def _get_err_p1d(x, alpha=4, xmin=0.1, xmax=5, ymin=1, ymax=4):
    t = (x - xmin) / (xmax - xmin)
    return ymin + ymax * t**alpha


def _get_err_p3d(x, alpha=0.2, xmin=0.1, xmax=5, ymin=3, ymax=20):
    t = (x - xmin) / (xmax - xmin)
    return ymax - (ymax - ymin) * t**alpha


class ArinyoFitter:
    """
    Fit the Arinyo model to P3D and P1D measurements.
    """

    PARAM_NAMES = (
        "bias",
        "bias_eta",
        "q1",
        "q2",
        "kvav",
        "av",
        "bv",
        "kp",
    )

    def __init__(
        self,
        zlist=np.arange(2.0, 4.501, 0.25),
        kmin_3d=0.7,
        kmax_3d=4.5,
        kmin_1d=0.3,
        kmax_1d=7.0,
        n_k_bins=20,
        n_mu_bins=16,
        boxsize=67.5,
        nrkbin=12,
        nrmubin=8,
        bounds=None,
    ):

        self.zlist = np.asarray(zlist)

        self.kmin_3d = kmin_3d
        self.kmax_3d = kmax_3d

        self.kmin_1d = kmin_1d
        self.kmax_1d = kmax_1d

        self.n_k_bins = n_k_bins
        self.n_mu_bins = n_mu_bins
        self.boxsize = boxsize

        self.nrkbin = nrkbin
        self.nrmubin = nrmubin

        if bounds is None:
            bounds = [
                (-1.0, -0.01),  # bias
                (-0.5, -0.01),  # bias_eta
                (0.0, 5.0),  # q1
                (-2.0, 2.0),  # q2
                (0.2, 15.5),  # kvav
                (0.0, 2.0),  # av
                (1.0, 5.0),  # bv
                (4.0, 50.0),  # kp
            ]

        self.bounds = bounds

        # Fine grid used for P3D binning
        self._prepare_model_grid()

    def _prepare_model_grid(self, kmax=20.0):
        """
        Build the fine (k, mu) grid used to evaluate the Arinyo model before
        averaging into the simulation bins.
        """

        lnk_max = np.log(kmax)
        lnk_min = np.log(2.0 * np.pi / self.boxsize)

        lnk_bin_max = lnk_max + (lnk_max - lnk_min) / (self.n_k_bins - 1)

        lnk_bin_edges = np.linspace(
            lnk_min,
            lnk_bin_max,
            self.n_k_bins + 1,
        )

        self.k_bin_edges = np.exp(lnk_bin_edges)
        self.mu_bin_edges = np.linspace(
            0.0,
            1.0,
            self.n_mu_bins + 1,
        )

        # Keep only bins inside the fitting range
        ind = np.where(
            (self.k_bin_edges >= self.kmin_3d) & (self.k_bin_edges < self.kmax_3d)
        )[0]

        self.k_bin_edges_fit = self.k_bin_edges[ind[0] - 1 : ind[-1] + 2]

        # Fine grid
        fine_k = np.geomspace(
            self.k_bin_edges_fit[0],
            self.k_bin_edges_fit[-1],
            len(self.k_bin_edges_fit) * self.nrkbin,
        )

        fine_mu = np.linspace(
            0.0,
            1.0,
            self.n_mu_bins * self.nrmubin,
        )

        self.fine_k, self.fine_mu = np.meshgrid(
            fine_k,
            fine_mu,
            indexing="ij",
        )

        # Mean coordinates inside each coarse bin
        self.k_mean, _, _, _ = binned_statistic_2d(
            self.fine_k.ravel(),
            self.fine_mu.ravel(),
            self.fine_k.ravel(),
            statistic="mean",
            bins=[self.k_bin_edges_fit, self.mu_bin_edges],
        )

        self.mu_mean, _, _, _ = binned_statistic_2d(
            self.fine_k.ravel(),
            self.fine_mu.ravel(),
            self.fine_mu.ravel(),
            statistic="mean",
            bins=[self.k_bin_edges_fit, self.mu_bin_edges],
        )

        ik = np.digitize(self.fine_k.ravel(), self.k_bin_edges_fit) - 1
        imu = np.digitize(self.fine_mu.ravel(), self.mu_bin_edges) - 1

        nk = len(self.k_bin_edges_fit) - 1
        nmu = len(self.mu_bin_edges) - 1

        mask = (ik >= 0) & (ik < nk) & (imu >= 0) & (imu < nmu)

        self._bin_index = ik[mask] * nmu + imu[mask]
        self._fine_mask = mask

        self._nbins = nk * nmu
        self._bin_counts = np.bincount(
            self._bin_index,
            minlength=self._nbins,
        )

        self._p3d_shape = (
            len(self.k_bin_edges_fit) - 1,
            nmu,
        )

        self._bin_norm = 1.0 / self._bin_counts

    def prepare_simulation(self, sim):
        """
        Prepare one simulation for fitting.
        """

        linear, power_model = self._build_model(sim)

        k3d, mu3d, p3d, std_p3d = self._prepare_p3d(sim)
        k1d, p1d, std_p1d = self._prepare_p1d(sim)

        self.data = FitData(
            z=sim["z"],
            linear=linear,
            power_model=power_model,
            k1d=k1d,
            p1d=p1d,
            std_p1d=std_p1d,
            k3d=k3d,
            mu3d=mu3d,
            p3d=p3d,
            std_p3d=std_p3d,
            ini_params=sim["Arinyo_min"],
        )

        return

    def _build_model(self, sim):
        """
        Build the cosmology, Arinyo model and linear theory.
        """

        cosmo = cosmology.Cosmology(sim["cosmo_params"])
        power_model = ArinyoModel(cosmo)
        linear = power_model.linear_theory(self.zlist)

        return linear, power_model

    def _prepare_p3d(self, sim):
        """
        Extract the fitted 3D power spectrum.
        """

        mask = (sim["k3d_Mpc"][:, 0] >= self.kmin_3d) & (
            sim["k3d_Mpc"][:, 0] < self.kmax_3d
        )

        k3d = sim["k3d_Mpc"][mask]
        mu3d = sim["mu3d"][mask]
        p3d = sim["p3d_Mpc"][mask]

        std_p3d = (
            _get_err_p3d(
                k3d,
                xmin=self.kmin_3d,
                xmax=self.kmax_3d,
            )
            * 0.01
        )

        return k3d, mu3d, p3d, std_p3d

    def _prepare_p1d(self, sim):
        """
        Extract the fitted 1D power spectrum.
        """

        mask = (sim["k_Mpc"] >= self.kmin_1d) & (sim["k_Mpc"] < self.kmax_1d)

        k1d = sim["k_Mpc"][mask]
        p1d = sim["p1d_Mpc"][mask]

        std_p1d = (
            _get_err_p1d(
                k1d,
                xmin=self.kmin_1d,
                xmax=self.kmax_1d,
            )
            * 0.01
        )

        return k1d, p1d, std_p1d

    def params_to_dict(self, params):
        """
        Convert a parameter vector into an Arinyo parameter dictionary.
        """

        params = np.asarray(params)

        return {name: value for name, value in zip(self.PARAM_NAMES, params)}

    def params_from_dict(self, params):
        """
        Convert an Arinyo parameter dictionary into a parameter vector.
        """

        return np.array(
            [params[name] for name in self.PARAM_NAMES],
            dtype=float,
        )

    def predict(self, params):
        """
        Evaluate the Arinyo model for the current simulation.

        Parameters
        ----------
        params : array-like
            Parameter vector.

        Returns
        -------
        p3d : ndarray
            Model P3D evaluated on the simulation grid.

        p1d : ndarray
            Model P1D.
        """

        ari_par = self.params_to_dict(params)

        # Evaluate on the fine grid
        fine_p3d = self.data.power_model.P3D_Mpc_k_mu(
            self.data.linear,
            self.data.z,
            self.fine_k,
            self.fine_mu,
            ari_par,
        )

        # Average into the simulation bins
        # p3d, _, _, _ = binned_statistic_2d(
        #     self.fine_k.ravel(),
        #     self.fine_mu.ravel(),
        #     fine_p3d.ravel(),
        #     statistic="mean",
        #     bins=[self.k_bin_edges_fit, self.mu_bin_edges],
        # )

        p3d = self._bin_p3d(fine_p3d)

        # Evaluate P1D
        p1d = self.data.power_model.P1D_Mpc(
            self.data.linear,
            self.data.z,
            self.data.k1d,
            ari_par,
        )

        return p3d, p1d

    def _bin_p3d(self, fine_p3d):
        """
        Average a fine-grid P3D onto the coarse (k,mu) grid.
        """

        bin_sum = np.bincount(
            self._bin_index,
            weights=fine_p3d.ravel()[self._fine_mask],
            minlength=self._nbins,
        )

        return (bin_sum * self._bin_norm).reshape(self._p3d_shape)

    def chi2(self, params):
        """
        Chi-square objective function.
        """

        p3d, p1d = self.predict(params)

        chi2_3d = np.nanmean(((p3d / self.data.p3d - 1.0) / self.data.std_p3d) ** 2)

        chi2_1d = np.nanmean(((p1d / self.data.p1d - 1.0) / self.data.std_p1d) ** 2)

        return chi2_3d + chi2_1d

    def residuals(self, params):
        """
        Return fractional residuals.
        """

        p3d, p1d = self.predict(params)

        return (
            p3d / self.data.p3d - 1,
            p1d / self.data.p1d - 1,
        )

    def fit(
        self,
        x0=None,
        bounds=None,
        method="L-BFGS-B",
        maxiter=500,
        ftol=1e-2,
        xatol=1e-5,
        **kwargs,
    ):
        """
        Fit the Arinyo model to the current simulation.

        Parameters
        ----------
        x0 : array-like, optional
            Initial guess. If None, uses the true simulation parameters.
        bounds : sequence, optional
            scipy.optimize bounds.
        method : str
            Optimization method.
        maxiter : int
            Maximum number of iterations.

        Returns
        -------
        OptimizeResult
        """

        if x0 is None:
            x0 = self.params_from_dict(self.data.ini_params.copy())

        result = minimize(
            self.chi2,
            x0,
            method=method,
            bounds=bounds,
            options={
                "disp": True,
                "maxiter": maxiter,
                "maxfev": maxiter,
                "fatol": ftol,
                "xatol": xatol,
                **kwargs,
            },
        )

        self.result = result
        self.best_params = result.x.copy()
        self.best_chi2 = result.fun

        return result

    def fit_iterative(
        self,
        niter=20,
        ftol=1e-2,
    ):
        """
        Alternate L-BFGS-B and Nelder-Mead until convergence.
        """

        x = self.params_from_dict(self.data.ini_params.copy())
        best = np.inf

        for i in range(niter):

            if i == 0:
                method = "L-BFGS-B"
                maxiter = 500
            else:
                method = "Nelder-Mead"
                maxiter = 1000

            result = self.fit(
                x0=x,
                bounds=self.bounds,
                method=method,
                maxiter=maxiter,
                ftol=ftol,
            )

            improvement = best - result.fun

            print(f"Iter {i:2d}   " f"chi2={result.fun:.4f}   " f"Δ={improvement:.4f}")

            if improvement < ftol:
                break

            best = result.fun
            x = result.x.copy()

        return self.result

    def plot_fit(
        self,
        params=None,
        normalized=True,
        figsize=(8, 8),
    ):
        """
        Compare the fitted model to the simulation.
        """

        if params is None:
            params = self.best_params

        p3d_model, p1d_model = self.predict(params)

        fig, ax = plt.subplots(
            2,
            1,
            figsize=figsize,
            constrained_layout=True,
        )

        # ---------- P3D ----------
        color = 0
        for imu in range(0, self.data.p3d.shape[1], 2):

            x = self.data.k3d[:, imu]

            if normalized:
                factor = x**3 / (2 * np.pi**2)
            else:
                factor = 1.0

            ax[0].plot(
                x,
                factor * self.data.p3d[:, imu],
                color=f"C{color}",
            )

            ax[0].plot(
                x,
                factor * p3d_model[:, imu],
                "--",
                color=f"C{color}",
            )

            color += 1

        # ---------- P1D ----------
        if normalized:
            factor = self.data.k1d / np.pi
        else:
            factor = 1.0

        ax[1].plot(
            self.data.k1d,
            factor * self.data.p1d,
            lw=2,
            label="Simulation",
        )

        ax[1].plot(
            self.data.k1d,
            factor * p1d_model,
            "--",
            lw=2,
            label="Model",
        )

        ax[0].set_ylabel(
            r"$\Delta^2_{\rm F}(k,\mu)$" if normalized else r"$P_{\rm F}(k,\mu)$"
        )
        ax[1].set_ylabel(r"$\Delta^2_{\rm 1D}$" if normalized else r"$P_{\rm 1D}$")

        ax[1].set_xlabel(r"$k\ [{\rm Mpc}^{-1}]$")

        ax[1].legend()

        return fig, ax

    def plot_residuals(
        self,
        params=None,
        figsize=(8, 8),
    ):
        """
        Plot fractional residuals.
        """

        if params is None:
            params = self.best_params

        res3d, res1d = self.residuals(params)

        fig, ax = plt.subplots(
            2,
            1,
            figsize=figsize,
            constrained_layout=True,
        )

        # ---------- P3D ----------
        for imu in range(self.data.p3d.shape[1]):

            ax[0].plot(
                self.data.k3d[:, imu],
                res3d[:, imu],
                color=f"C{imu}",
            )

        ax[0].fill_between(
            self.data.k3d[:, 0],
            -0.05,
            0.05,
            color="k",
            alpha=0.15,
        )

        ax[0].fill_between(
            self.data.k3d[:, 0],
            -self.data.std_p3d[:, 0],
            self.data.std_p3d[:, 0],
            color="k",
            alpha=0.3,
        )

        # ---------- P1D ----------
        ax[1].plot(
            self.data.k1d,
            res1d,
            lw=2,
        )

        ax[1].fill_between(
            self.data.k1d,
            -0.01,
            0.01,
            color="k",
            alpha=0.15,
        )

        ax[1].fill_between(
            self.data.k1d,
            -self.data.std_p1d,
            self.data.std_p1d,
            color="k",
            alpha=0.3,
        )

        ax[0].set_xscale("log")
        ax[1].set_xscale("log")

        ax[0].set_ylabel(r"$P_{3D}^{\rm model}/P_{3D}^{\rm sim}-1$")
        ax[1].set_ylabel(r"$P_{1D}^{\rm model}/P_{1D}^{\rm sim}-1$")

        ax[1].set_xlabel(r"$k\ [{\rm Mpc}^{-1}]$")
        ax[1].set_ylim(-0.05, 0.05)
        ax[0].set_ylim(-0.15, 0.15)

        return fig, ax
