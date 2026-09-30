"""Implementation backend for :mod:`forestflow.model_fits`.

Users should import :class:`ArinyoFitter` from ``forestflow.model_fits``.  This
module remains at its historical location to avoid breaking local workflows.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
import warnings
from typing import Any
from numpy.typing import ArrayLike, NDArray

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from forestflow.model.arinyo import ArinyoModel
from forestflow.statistics.p3d import P3D_Mpc_k_mu_hybrid_averaged
from forestflow.statistics.rebin_p3d import get_P3D_k_mu_modes
from .data import FitData
from .errors import _get_err_p1d, _get_err_p3d
from lace.cosmo import cosmology
from lace.archive.gadget_archive import MPG_SIM_REDSHIFTS


class ArinyoFitter:
    """
    Fit the Arinyo model to P3D and P1D measurements.
    """

    # Keep the linear-theory grid exactly aligned with the MP-Gadget archive.
    # Importing this module-level declaration does not construct or load an
    # archive.
    DEFAULT_ZLIST = MPG_SIM_REDSHIFTS

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
        zlist: Any | None = None,
        kmin_3d: float | None = 0.01,
        kmax_3d: float | None = 4.5,
        kmin_1d: float | None = 0.01,
        kmax_1d: float | None = 7.0,
        n_k_bins: int | None = 20,
        n_mu_bins: int | None = 16,
        boxsize: float | None = 67.5,
        bounds: Any | None = None,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        zlist : object
            Zlist used by the calculation.
        kmin_3d : float, optional
            Kmin 3d used by the calculation.
        kmax_3d : float, optional
            Maximum three-dimensional wavenumber included in the fit.
        kmin_1d : float, optional
            Kmin 1d used by the calculation.
        kmax_1d : float, optional
            Maximum one-dimensional wavenumber included in the fit.
        n_k_bins : int, optional
            N k bins used by the calculation.
        n_mu_bins : int, optional
            N mu bins used by the calculation.
        boxsize : float, optional
            Boxsize used by the calculation.
        bounds : object, optional
            Lower and upper bounds for each parameter.
        """
        self.zlist = (
            self.DEFAULT_ZLIST.copy()
            if zlist is None
            else np.asarray(zlist, dtype=float)
        )

        self.kmin_3d = kmin_3d
        self.kmax_3d = kmax_3d

        self.kmin_1d = kmin_1d
        self.kmax_1d = kmax_1d

        self.n_k_bins = n_k_bins
        self.n_mu_bins = n_mu_bins
        self.boxsize = boxsize

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

        self._prepare_mpg_bin_geometry()

    def _prepare_mpg_bin_geometry(self, kmax_iMpc: float = 20.0) -> None:
        """Define native MP-Gadget P3D bin geometry for hybrid averaging."""

        lnk_max = np.log(kmax_iMpc)
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

        # Simulation spectra are selected by their radial-bin centres in
        # ``_prepare_p3d``. Select the model bins by the same convention;
        # selecting edges includes one extra bin whenever a cut lies between
        # two edges.
        k_bin_centres = np.sqrt(self.k_bin_edges[:-1] * self.k_bin_edges[1:])
        ind = np.flatnonzero(
            (k_bin_centres >= self.kmin_3d) & (k_bin_centres < self.kmax_3d)
        )
        if len(ind) == 0:
            raise ValueError("P3D scale cuts do not select any model radial bins")
        self._model_k_indices = ind
        self.k_bin_edges_fit = self.k_bin_edges[ind[0] : ind[-1] + 2]

        self._p3d_shape = (
            len(self.k_bin_edges_fit) - 1,
            len(self.mu_bin_edges) - 1,
        )

    def _initial_parameters_from_simulation(
        self,
        sim: Mapping[str, Any],
        standard_simulations: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return local parameters or the matching standard-postprocessing fit."""
        if "Arinyo_min" in sim:
            return dict(sim["Arinyo_min"])

        # Corrected P3D post-processing may not yet have a fit stored for a
        # snapshot. Its standard-postprocessing counterpart is then the only
        # approved initialization source; do not substitute mpg_central.
        if standard_simulations is None and "sim_label" in sim:
            if not hasattr(self, "_standard_postproc_simulations"):
                from forestflow.archive.gadget_archive import GadgetArchive3D

                standard_archive = GadgetArchive3D(postproc="Cabayol23")
                # ``get_training_data`` regenerates measurements without the
                # cached Arinyo fits.  The constructor-attached
                # ``training_data`` is the authoritative fitted collection.
                self._standard_postproc_simulations = [
                    snapshot
                    for snapshot in standard_archive.training_data
                    if snapshot["sim_label"] == sim["sim_label"]
                ]
            standard_simulations = self._standard_postproc_simulations
        if standard_simulations is not None:
            # These are the fields assigned by LaCE's GadgetArchive when it
            # expands a post-processing file into individual measurements.
            # ``ind_rescaling`` identifies the optical-depth rescaling.
            identity_keys = (
                "sim_label",
                "ind_snap",
                "ind_phase",
                "ind_axis",
                "ind_rescaling",
                "z",
            )

            def same_identity_value(left, right):
                """Compare archive identifiers without coercing string fields."""
                try:
                    return bool(np.isclose(float(left), float(right)))
                except (TypeError, ValueError):
                    return left == right

            matches = list(standard_simulations)
            for key in identity_keys:
                if key not in sim:
                    continue
                matches = [
                    candidate
                    for candidate in matches
                    if key in candidate
                    and same_identity_value(candidate[key], sim[key])
                ]
            if len(matches) == 1 and "Arinyo_min" in matches[0]:
                return dict(matches[0]["Arinyo_min"])
            if len(matches) > 1:
                raise ValueError(
                    "Standard-postprocessing fallback is ambiguous for "
                    f"simulation {sim.get('sim_label', '<unknown>')}"
                )

        raise KeyError(
            "Simulation has no Arinyo_min and no matching standard "
            "Cabayol23 post-processing fit"
        )

    def prepare_simulation(
        self,
        sim: Mapping[str, Any],
        *,
        standard_simulations: Sequence[Mapping[str, Any]] | None = None,
        is_mpg: bool = True,
    ) -> None:
        """Prepare one simulation for fitting.

        The snapshot's ``Arinyo_min`` values initialize the fit when present.
        For corrected measurements without a stored fit, the matching snapshot
        from standard ``Cabayol23`` post-processing is used. Supply
        ``standard_simulations`` to reuse snapshots already loaded from that
        archive; otherwise they are loaded and cached on first use. A missing
        match raises an error rather than using MPG-central parameters.

        Parameters
        ----------
        is_mpg : bool, default=True
            Treat the input as an MP-Gadget measurement and evaluate P3D with
            finite-volume hybrid averaging. Set this explicitly to ``False``
            for a non-MP-Gadget measurement; the fit then evaluates P3D at
            supplied bin centres and emits a warning because this can bias
            small-scale results.
        """

        linear, power_model = self._build_model(sim)

        k3d, mu3d, p3d, std_p3d = self._prepare_p3d(sim)
        k1d, p1d, std_p1d = self._prepare_p1d(sim)
        ini_params = self._initial_parameters_from_simulation(
            sim,
            standard_simulations=standard_simulations,
        )

        self._direct_p3d = not is_mpg
        self._p3d_averaging = "hybrid" if is_mpg else "centres"
        if is_mpg:
            self._prepare_mpg_hybrid_geometry()
        else:
            self._warn_non_mpg_centre_evaluation()
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
            ini_params=ini_params,
        )

        return

    def prepare_measurements(
        self,
        *,
        z: float,
        cosmo_params: Mapping[str, Any],
        k3d_Mpc: ArrayLike,
        mu3d: ArrayLike,
        p3d_Mpc: ArrayLike,
        std_p3d: ArrayLike,
        k1d_Mpc: ArrayLike,
        p1d_Mpc: ArrayLike,
        std_p1d: ArrayLike,
        ini_params: Mapping[str, Any] | None = None,
    ) -> None:
        """Prepare arbitrary P3D and P1D measurements for a direct Arinyo fit.

        Unlike :meth:`prepare_simulation`, this method does not assume an
        archive schema or rebin P3D. The supplied P3D coordinates and relative
        uncertainties are used directly, making it suitable for an external
        simulation or measurement such as Astrid. Wavenumbers are in Mpc^-1;
        P3D and P1D are in Mpc^3 and Mpc, respectively.
        """
        k3d_Mpc = np.asarray(k3d_Mpc, dtype=float)
        mu3d = np.asarray(mu3d, dtype=float)
        p3d_Mpc = np.asarray(p3d_Mpc, dtype=float)
        std_p3d = np.asarray(std_p3d, dtype=float)
        k1d_Mpc = np.asarray(k1d_Mpc, dtype=float)
        p1d_Mpc = np.asarray(p1d_Mpc, dtype=float)
        std_p1d = np.asarray(std_p1d, dtype=float)
        if not (k3d_Mpc.shape == mu3d.shape == p3d_Mpc.shape == std_p3d.shape):
            raise ValueError(
                "P3D coordinates, values, and uncertainties must share a shape"
            )
        if not (k1d_Mpc.shape == p1d_Mpc.shape == std_p1d.shape):
            raise ValueError(
                "P1D coordinates, values, and uncertainties must share a shape"
            )
        if np.any(std_p3d <= 0) or np.any(std_p1d <= 0):
            raise ValueError("relative uncertainties must be positive")

        power_model = ArinyoModel(cosmology.Cosmology(cosmo_params))
        linear = power_model.linear.get_linear_theory(z)
        defaults = power_model.default_params if ini_params is None else ini_params
        self._direct_p3d = True
        self._p3d_averaging = "centres"
        self._warn_non_mpg_centre_evaluation()
        self.data = FitData(
            z=float(z),
            linear=linear,
            power_model=power_model,
            ini_params=dict(defaults),
            k1d=k1d_Mpc,
            p1d=p1d_Mpc,
            std_p1d=std_p1d,
            k3d=k3d_Mpc,
            mu3d=mu3d,
            p3d=p3d_Mpc,
            std_p3d=std_p3d,
        )

    @staticmethod
    def _warn_non_mpg_centre_evaluation() -> None:
        """Warn when finite-volume MP-Gadget geometry is unavailable."""
        warnings.warn(
            "NON-MP-GADGET P3D FIT: evaluating the model at supplied bin "
            "centres instead of finite-volume mode averages. This is an "
            "approximation and can introduce appreciable small-scale errors; "
            "provide MP-Gadget geometry or use is_mpg=True when applicable.",
            UserWarning,
            stacklevel=3,
        )

    def _prepare_mpg_hybrid_geometry(self) -> None:
        """Construct and cache only the MP-Gadget modes needed by this fit.

        The discrete lattice grows cubically with its maximum wavenumber.  It
        is therefore essential to stop at the final fitted bin edge rather
        than construct the archive's complete 20 iMpc lattice.
        """
        # Use the exact native rows selected from the simulation data, not
        # approximate analytic bin centres.  This remains valid when a user
        # changes either P3D scale cut.
        data_indices = np.asarray(self._selected_data_k_indices, dtype=int)
        if data_indices.size == 0 or np.any(np.diff(data_indices) != 1):
            raise ValueError(
                "MP-Gadget P3D scale cuts must select a contiguous set of "
                "native radial bins."
            )
        self._model_k_indices = data_indices
        self.k_bin_edges_fit = self.k_bin_edges[data_indices[0] : data_indices[-1] + 2]

        cache_key = (
            float(self.boxsize),
            int(self.n_k_bins),
            int(self.n_mu_bins),
            tuple(data_indices),
        )
        if not hasattr(self, "_mpg_hybrid_geometry_cache"):
            self._mpg_hybrid_geometry_cache = {}
        if cache_key in self._mpg_hybrid_geometry_cache:
            self._mpg_k_mu_modes = self._mpg_hybrid_geometry_cache[cache_key]
            return

        # Preserve the native 20-iMpc parent bin definition, but generate
        # lattice vectors only through the highest bin participating in the
        # current fit (typically ~4.9 iMpc for kmax_3d=4.5).
        all_modes = get_P3D_k_mu_modes(
            self.k_bin_edges_fit[-1],
            Lbox_Mpc=self.boxsize,
            k_grid_max_iMpc=20.0,
            n_k_bins=self.n_k_bins,
            n_mu_bins=self.n_mu_bins,
        )
        selected = {}
        for new_index, old_index in enumerate(self._model_k_indices):
            for mu_index in range(self.n_mu_bins):
                old_key = f"{old_index}_{mu_index}"
                if old_key + "_k" in all_modes:
                    new_key = f"{new_index}_{mu_index}"
                    selected[new_key + "_k"] = all_modes[old_key + "_k"]
                    selected[new_key + "_mu"] = all_modes[old_key + "_mu"]
        self._mpg_hybrid_geometry_cache[cache_key] = selected
        self._mpg_k_mu_modes = selected

    def _build_model(self, sim: Any) -> tuple[Any, ...]:
        """
        Build the cosmology, Arinyo model and linear theory.

        Parameters
        ----------
        sim : object
            Sim used by the calculation.

        Returns
        -------
        tuple
            Result produced when the function is used to build the cosmology, arinyo model and linear theory.
        """

        cosmo = cosmology.Cosmology(sim["cosmo_params"])
        power_model = ArinyoModel(cosmo)
        linear = power_model.linear.get_linear_theory(self.zlist)

        return linear, power_model

    def _prepare_p3d(self, sim: Any) -> tuple[Any, ...]:
        """
        Extract the fitted 3D power spectrum.

        Parameters
        ----------
        sim : object
            Sim used by the calculation.

        Returns
        -------
        tuple
            Result produced when the function is used to extract the fitted 3d power spectrum.
        """

        mask = (sim["k3d_Mpc"][:, 0] >= self.kmin_3d) & (
            sim["k3d_Mpc"][:, 0] < self.kmax_3d
        )
        # For MP-Gadget the rows retain native radial-bin order.  Keep these
        # indices so the hybrid model uses exactly the same bin selection as
        # the measurement, whose mode-weighted centres differ slightly from
        # ideal logarithmic centres.
        self._selected_data_k_indices = np.flatnonzero(mask)

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

    def _prepare_p1d(self, sim: Any) -> tuple[Any, ...]:
        """
        Extract the fitted 1D power spectrum.

        Parameters
        ----------
        sim : object
            Sim used by the calculation.

        Returns
        -------
        tuple
            Result produced when the function is used to extract the fitted 1d power spectrum.
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

    def params_to_dict(self, params: Mapping[str, Any]) -> Any:
        """
        Convert a parameter vector into an Arinyo parameter dictionary.

        Parameters
        ----------
        params : dict
            Model parameter values.

        Returns
        -------
        object
            Result produced when the function is used to convert a parameter vector into an arinyo parameter dictionary.
        """

        params = np.asarray(params)

        return {name: value for name, value in zip(self.PARAM_NAMES, params)}

    def params_from_dict(self, params: Mapping[str, Any]) -> Any:
        """
        Convert an Arinyo parameter dictionary into a parameter vector.

        Parameters
        ----------
        params : dict
            Model parameter values.

        Returns
        -------
        object
            Result produced when the function is used to convert an arinyo parameter dictionary into a parameter vector.
        """

        return np.array(
            [params[name] for name in self.PARAM_NAMES],
            dtype=float,
        )

    def predict(self, params: ArrayLike) -> tuple[Any, ...]:
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

        if self._p3d_averaging == "hybrid":
            p3d = P3D_Mpc_k_mu_hybrid_averaged(
                self.data.linear,
                self.data.z,
                self.data.power_model.P3D_Mpc_k_mu,
                ari_par,
                k_mu_modes=self._mpg_k_mu_modes,
                k_iMpc_edges=self.k_bin_edges_fit,
                mu_edges=self.mu_bin_edges,
            )
            if p3d.shape != self.data.p3d.shape:
                raise ValueError(
                    "MP-Gadget hybrid P3D geometry does not match the selected "
                    "measurement grid; call prepare_simulation(..., is_mpg=False) "
                    "only for a non-MP-Gadget dataset."
                )
        else:
            p3d = self.data.power_model.P3D_Mpc_k_mu(
                self.data.linear,
                self.data.z,
                self.data.k3d,
                self.data.mu3d,
                ari_par,
            )

        # Evaluate P1D
        p1d = self.data.power_model.P1D_Mpc(
            self.data.linear,
            self.data.z,
            self.data.k1d,
            ari_par,
        )

        return p3d, p1d

    def save_results(
        self,
        filename: str | Path,
        *,
        snapshots: Sequence[Mapping[str, Any]],
        initial_chi2: ArrayLike,
        chi2: ArrayLike,
        success: ArrayLike,
        message: ArrayLike,
        arinyo: Mapping[str, ArrayLike],
        simulation_label: str,
        postproc: str,
    ) -> Path:
        """Save a portable collection of Arinyo fits and archive identities.

        This is shared by notebooks and batch scripts.  ``snapshots`` defines
        the archive identity columns; all result arrays must have one entry
        per snapshot.  Parameter names always remain ordinary API names.
        """
        snapshots = list(snapshots)
        n_snapshots = len(snapshots)
        arrays = {
            "initial_chi2": np.asarray(initial_chi2),
            "chi2": np.asarray(chi2),
            "success": np.asarray(success),
            "message": np.asarray(message, dtype=object),
        }
        for name, values in arrays.items():
            if values.shape != (n_snapshots,):
                raise ValueError(f"{name} must have one value per snapshot")
        arinyo_output = {}
        for name in self.PARAM_NAMES:
            if name not in arinyo:
                raise KeyError(f"arinyo is missing parameter {name!r}")
            values = np.asarray(arinyo[name], dtype=float)
            if values.shape != (n_snapshots,):
                raise ValueError(f"arinyo[{name!r}] must have one value per snapshot")
            arinyo_output[name] = values

        output = Path(filename)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 2,
            "simulation_label": simulation_label,
            "postproc": postproc,
            "p3d_averaging": getattr(self, "_p3d_averaging", "hybrid"),
            "parameter_names": self.PARAM_NAMES,
            "z": np.asarray([item["z"] for item in snapshots]),
            "ind_snap": np.asarray([item.get("ind_snap") for item in snapshots]),
            "ind_phase": np.asarray([item.get("ind_phase") for item in snapshots]),
            "ind_axis": np.asarray([item.get("ind_axis") for item in snapshots]),
            "ind_rescaling": np.asarray(
                [item.get("ind_rescaling") for item in snapshots]
            ),
            **arrays,
            "Arinyo": arinyo_output,
        }
        np.save(output, payload)
        return output

    def _warn_if_final_parameters_near_bounds(self, fraction: float = 0.01) -> None:
        """Print a warning for final parameters within a bound-range fraction."""
        for name, value, (lower, upper) in zip(
            self.PARAM_NAMES, self.best_params, self.bounds
        ):
            span = upper - lower
            if span <= 0:
                continue
            distance_lower = value - lower
            distance_upper = upper - value
            if distance_lower <= fraction * span:
                print(
                    f"WARNING: final {name}={value:.6g} is within "
                    f"{fraction:.1%} of its lower bound ({lower:.6g})."
                )
            elif distance_upper <= fraction * span:
                print(
                    f"WARNING: final {name}={value:.6g} is within "
                    f"{fraction:.1%} of its upper bound ({upper:.6g})."
                )

    def chi2(self, params: Mapping[str, Any]) -> Any:
        """
        Chi-square objective function.

        Parameters
        ----------
        params : dict
            Model parameter values.

        Returns
        -------
        object
            Result produced when the function is used to chi-square objective function.
        """

        # Invalid trial points can overflow the nonlinear model. They are not
        # valid minima, so return an infinite objective instead of propagating
        # NaNs into the optimizer's numerical derivatives.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            p3d, p1d = self.predict(params)
            chi2_3d = np.nanmean(((p3d / self.data.p3d - 1.0) / self.data.std_p3d) ** 2)
            chi2_1d = np.nanmean(((p1d / self.data.p1d - 1.0) / self.data.std_p1d) ** 2)
        chi2 = chi2_3d + chi2_1d
        return float(chi2) if np.isfinite(chi2) else 1e100

    def residuals(self, params: Mapping[str, Any]) -> tuple[Any, ...]:
        """
        Return fractional residuals.

        Parameters
        ----------
        params : dict
            Model parameter values.

        Returns
        -------
        tuple
            Result produced when the function is used to return fractional residuals.
        """

        p3d, p1d = self.predict(params)

        return (
            p3d / self.data.p3d - 1,
            p1d / self.data.p1d - 1,
        )

    def fit(
        self,
        x0: ArrayLike | None = None,
        bounds: Sequence[Any] | None = None,
        method: str = "L-BFGS-B",
        maxiter: int = 500,
        ftol: Any = 1e-2,
        xatol: Any = 1e-5,
        **kwargs: Mapping[str, Any],
    ) -> Any:
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

        Other Parameters
        ----------------
        ftol : object
            Relative function-value convergence tolerance.
        xatol : object
            Absolute parameter convergence tolerance.
        kwargs : dict
            Additional keyword arguments forwarded to the underlying calculation.
        """

        if x0 is None:
            x0 = self.params_from_dict(self.data.ini_params.copy())

        options = {"maxiter": maxiter, **kwargs}
        if method == "Nelder-Mead":
            options.update(maxfev=maxiter, fatol=ftol, xatol=xatol)
        elif method == "L-BFGS-B":
            options["ftol"] = ftol

        result = minimize(
            self.chi2,
            x0,
            method=method,
            bounds=bounds,
            options=options,
        )

        self.result = result
        self.best_params = result.x.copy()
        self.best_chi2 = result.fun

        return result

    def fit_iterative(
        self,
        niter: int | None = 20,
        ftol: float | None = 1e-2,
    ) -> Any:
        """
        Alternate L-BFGS-B and Nelder-Mead until convergence.

        Parameters
        ----------
        niter : int, optional
            Niter used by the calculation.
        ftol : float, optional
            Ftol used by the calculation.

        Returns
        -------
        object
            Result produced when the function is used to alternate l-bfgs-b and nelder-mead until convergence.
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

        self._warn_if_final_parameters_near_bounds()
        return self.result

    def plot_fit(
        self,
        params: Any | None = None,
        normalized: bool | None = True,
        figsize: Sequence[Any] | None = (8, 8),
    ) -> tuple[Any, ...]:
        """
        Compare the fitted model to the simulation.

        Parameters
        ----------
        params : object, optional
            Model parameter values.
        normalized : bool, optional
            Normalized used by the calculation.
        figsize : tuple, optional
            Figsize used by the calculation.

        Returns
        -------
        tuple
            Result produced when the function is used to compare the fitted model to the simulation.
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

            # The values at each point are the bin averages used in the
            # likelihood; connecting them only aids visual interpretation.
            ax[0].plot(
                x,
                factor * self.data.p3d[:, imu],
                color=f"C{color}",
                label="Simulation bin averages" if color == 0 else None,
            )

            ax[0].plot(
                x,
                factor * p3d_model[:, imu],
                "--",
                color=f"C{color}",
                label="Hybrid model bin averages" if color == 0 else None,
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

        ax[0].legend()
        ax[1].legend()

        return fig, ax

    def plot_residuals(
        self,
        params: Any | None = None,
        figsize: Sequence[Any] | None = (8, 8),
    ) -> tuple[Any, ...]:
        """
        Plot fractional residuals.

        Parameters
        ----------
        params : object, optional
            Model parameter values.
        figsize : tuple, optional
            Figsize used by the calculation.

        Returns
        -------
        tuple
            Result produced when the function is used to plot fractional residuals.
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

            # These residuals use the same hybrid bin averages as chi2;
            # connecting them only aids visual interpretation.
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
