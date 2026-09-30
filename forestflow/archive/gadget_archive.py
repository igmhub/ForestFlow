"""
forestflow.archive.gadget_archive

Utilities to load and process Arinyo/Gadget archives for ForestFlow.

This module provides `GadgetArchive3D`, which extends
`lace.archive.gadget_archive.GadgetArchive` with helpers to load
training/testing data and Arinyo minimizer fits (both individual
and redshift-parameterized), and to compute simple priors for
Arinyo / IGM parameters.
"""

from typing import Any

import numpy as np
import os
from lace.archive.gadget_archive import GadgetArchive
import forestflow
from forestflow.utils import params_numpy2dict_minimizerz
from forestflow.statistics.rebin_p3d import MPG_P3D_BINNING, get_P3D_k_mu_bin_edges


class GadgetArchive3D(GadgetArchive):
    """
    Archive helpers for 3D Gadget simulations.

    Extends `GadgetArchive` with methods to load training and testing
    data, attach Arinyo minimizer results (individual snapshots and
    joint/redshift-parameterized fits), and to compute summary priors
    for Arinyo and IGM parameters over redshift ranges.
    """

    def __init__(
        self,
        base_folder: Any | None = None,
        file_errors: Any | None = None,
        postproc: str | None = "Cabayol23",
        kp_Mpc: Any | None = None,
        average: str | None = "both",
        addcentral: bool | None = False,
    ) -> None:
        """
        Archive class for 3D simulations

        It calls the Lace GadgetArchive class and adds the Arinyo parameters

        Parameters
        ----------
        base_folder : object, optional
            Base folder used by the calculation.
        file_errors : object, optional
            File errors used by the calculation.
        postproc : str, optional
            Postproc used by the calculation.
        kp_Mpc : object, optional
            Kp mpc used by the calculation.
        average : str, optional
            Average used by the calculation.
        addcentral : bool, optional
            Addcentral used by the calculation.
        """

        if base_folder == None:
            self.base_folder = os.path.dirname(forestflow.__path__[0])
        else:
            self.base_folder = base_folder

        if file_errors == None:
            file_errors = os.path.join(self.base_folder, "data", "std_pnd_mpg.npz")

        err_pnd = np.load(file_errors)
        self.rel_err_p1d = err_pnd["std_p1d"]
        self.rel_err_p3d = err_pnd["std_p3d"]

        self.P3D_binning = MPG_P3D_BINNING.copy()

        self.emu_params = [
            "Delta2_p",
            "n_p",
            "mF",
            "sigT_Mpc",
            "gamma",
            "kF_Mpc",
        ]

        super().__init__(postproc=postproc, kp_Mpc=kp_Mpc)

        self.training_data = self.get_training_data(average=average)

        # mcmc chains, only computed for both
        if average == "both":
            self.add_Arinyo_minimizer_indiv(
                self.training_data, sim_label="mpg_hypercube"
            )
            # self.add_Arinyo_minimizer_joint(
            #     self.training_data, sim_label="mpg_hypercube"
            # )
            # self.add_Arinyo_minimizer_indiv_lowk(
            #     self.training_data, sim_label="mpg_hypercube"
            # )
            self.add_arinyo_fixp3d(self.training_data)

        if addcentral:
            central_data = self.get_testing_data("mpg_central")
            self.training_data.extend(central_data)

    def get_P3D_k_mu_bin_edges(self, k_max_iMpc=None):
        """Return the native MP-Gadget P3D bin edges used by this archive."""
        return get_P3D_k_mu_bin_edges(k_max_iMpc, **self.P3D_binning)

    def get_training_data(
        self,
        simulation_label: str | list[str] | None = None,
        emu_params: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return MPG training snapshots with ForestFlow's standard inputs.

        Parameters
        ----------
        simulation_label
            One hypercube simulation label (for example ``"mpg_0"``), a list
            of labels, or ``None`` for all training simulations. The legacy
            positional form ``get_training_data(emu_params, ...)`` remains
            supported when its first argument is a list of parameter names.
        emu_params
            Required emulator-input fields. Defaults to the standard
            ForestFlow inputs: ``Delta2_p``, ``n_p``, ``mF``, ``sigT_Mpc``,
            ``gamma``, and ``kF_Mpc``.
        **kwargs
            Selection options accepted by LaCE's base archive method, such as
            ``average``, ``val_scaling``, and redshift or simulation cuts.
        """
        # Preserve the inherited positional API for downstream callers that
        # supplied the emulator input list as the first argument.
        if (
            isinstance(simulation_label, list)
            and emu_params is None
            and not all(label in self.list_sim_cube for label in simulation_label)
        ):
            emu_params = simulation_label
            simulation_label = None

        # During ``__init__`` this method constructs ``training_data`` from
        # the parent archive. Afterwards the default public call merely
        # selects from that already loaded, fitted collection; do not read and
        # rebuild the full archive again just to select one MPG simulation.
        use_cache = hasattr(self, "training_data") and emu_params is None and not kwargs

        if emu_params is None:
            emu_params = self.emu_params
        if not isinstance(emu_params, list):
            raise TypeError("emu_params must be a list or None")

        if simulation_label is None:
            selected_labels = None
        elif isinstance(simulation_label, str):
            selected_labels = [simulation_label]
        elif isinstance(simulation_label, list) and all(
            isinstance(label, str) for label in simulation_label
        ):
            selected_labels = simulation_label
        else:
            raise TypeError(
                "simulation_label must be a string, list of strings, or None"
            )

        if selected_labels is not None:
            invalid_labels = [
                label for label in selected_labels if label not in self.list_sim_cube
            ]
            if invalid_labels:
                raise ValueError(
                    "simulation_label must name MPG hypercube simulations; "
                    f"invalid value(s): {invalid_labels}"
                )

        training_data = (
            self.training_data
            if use_cache
            else super().get_training_data(emu_params, **kwargs)
        )
        if selected_labels is None:
            return training_data
        return [
            snapshot
            for snapshot in training_data
            if snapshot["sim_label"] in selected_labels
        ]

    def get_testing_data(
        self,
        sim_label: str,
        ind_rescaling: int | None = 0,
        kmax_3d: float | None = 5,
        kmax_1d: float | None = 4,
    ) -> dict[str, Any]:
        """
        Return testing data augmented with Arinyo minimizer fits.

        Loads testing data for the given `sim_label` (via
        the parent `GadgetArchive.get_testing_data`) and attaches both
        individual and joint Arinyo minimizer results to each snapshot
        entry in the returned archive structure.

        Parameters
        ----------
        sim_label : str
            Label of the simulation to load.
        ind_rescaling : int, optional
            Index of the optical-depth rescaling to select.
        kmax_3d : float, optional
            Maximum three-dimensional wavenumber used by the fit.
        kmax_1d : float, optional
            Maximum one-dimensional wavenumber used by the fit.

        Returns
        -------
        list of dict
            Testing snapshots augmented with ``Arinyo_min`` and
            ``Arinyo_minz`` entries where available.
        """
        testing_data = super().get_testing_data(sim_label, ind_rescaling=ind_rescaling)
        self.add_Arinyo_minimizer_indiv(testing_data, sim_label, kmax_3d, kmax_1d)
        # self.add_Arinyo_minimizer_joint(testing_data, sim_label)
        # self.add_Arinyo_minimizer_indiv_lowk(testing_data, sim_label=sim_label)
        self.add_arinyo_fixp3d(testing_data)

        return testing_data

    def get_central_seed_average(self) -> list[dict[str, Any]]:
        """Return mean-flux-consistent averages of central and seed snapshots.

        The two MP-Gadget realizations have the same cosmology and native
        Fourier grid.  Scalar IGM summaries are averaged arithmetically,
        while P1D and P3D are first converted to absolute flux power using
        ``mF**2``, averaged, and converted back using the averaged ``mF``.
        The returned snapshots are labelled ``"mpg_central_seed"`` and retain
        the central fit as their initial condition.  If its corrected combined
        fit file exists, it is attached as ``arinyo_fixp3d``.
        """
        central = self.get_testing_data("mpg_central")
        seed = self.get_testing_data("mpg_seed")
        identity_fields = ("z", "ind_snap", "ind_phase", "ind_axis", "ind_rescaling")

        def identity_value(value: Any) -> Any:
            if isinstance(value, np.generic):
                value = value.item()
            return round(float(value), 10) if isinstance(value, float) else value

        def identity(snapshot: dict[str, Any]) -> tuple[Any, ...]:
            return tuple(identity_value(snapshot[field]) for field in identity_fields)

        seed_by_identity = {identity(snapshot): snapshot for snapshot in seed}
        if len(seed_by_identity) != len(seed):
            raise ValueError("mpg_seed contains duplicate snapshot identities")
        combined = []
        average_fields = ("mF", "T0", "gamma", "sigT_Mpc", "kF_Mpc")
        grid_fields = ("k_Mpc", "k3d_Mpc", "mu3d")
        reported_coordinate_mismatch = False
        for central_snapshot in central:
            snapshot_identity = identity(central_snapshot)
            try:
                seed_snapshot = seed_by_identity.pop(snapshot_identity)
            except KeyError as error:
                raise KeyError(
                    "mpg_seed has no snapshot matching mpg_central identity "
                    f"{snapshot_identity}"
                ) from error
            for field in grid_fields:
                central_grid = np.asarray(central_snapshot[field])
                seed_grid = np.asarray(seed_snapshot[field])
                if central_grid.shape != seed_grid.shape:
                    raise ValueError(
                        f"mpg_central and mpg_seed have incompatible {field} "
                        f"shapes for snapshot {snapshot_identity}: "
                        f"{central_grid.shape} != {seed_grid.shape}"
                    )
                # The two realizations share the same Fourier lattice and bin
                # definitions. Their archived coordinates can nevertheless
                # differ because they are power/mode-weighted reported bin
                # centres. The combination follows the historical definition:
                # combine corresponding cells and retain central coordinates.
                reported_coordinate_mismatch |= not np.allclose(
                    central_grid, seed_grid, equal_nan=True
                )

            snapshot = central_snapshot.copy()
            # A combined measurement must never inherit central's corrected
            # fit. ``add_arinyo_fixp3d`` below attaches only the dedicated
            # mpg_central_seed result when that file exists.
            snapshot.pop("arinyo_fixp3d", None)
            for field in average_fields:
                snapshot[field] = 0.5 * (central_snapshot[field] + seed_snapshot[field])
            if snapshot["mF"] == 0:
                raise ValueError(f"Combined mean flux is zero for {snapshot_identity}")
            for field in ("p1d_Mpc", "p3d_Mpc"):
                snapshot[field] = (
                    central_snapshot["mF"] ** 2 * central_snapshot[field]
                    + seed_snapshot["mF"] ** 2 * seed_snapshot[field]
                ) / (2.0 * snapshot["mF"] ** 2)
            snapshot["sim_label"] = "mpg_central_seed"
            combined.append(snapshot)
        if seed_by_identity:
            raise KeyError(
                "mpg_seed contains snapshots with no mpg_central counterpart: "
                f"{sorted(seed_by_identity)}"
            )
        if reported_coordinate_mismatch:
            print(
                "NOTE: central and seed store different mode-weighted bin "
                "centres; combined powers retain the central coordinates."
            )

        self.add_arinyo_fixp3d(combined)
        return combined

    def add_arinyo_fixp3d(self, archive: list[dict[str, Any]]) -> None:
        """Attach corrected-postprocessing Arinyo fits as ``arinyo_fixp3d``.

        Fits are matched to archive entries through their complete saved
        snapshot identity, rather than through list order.  This retains the
        corrected-fit parameters alongside the legacy ``Arinyo_min`` and
        ``Arinyo_lowk`` fields.
        """
        fit_folder = os.path.join(
            self.base_folder, "data", "best_arinyo", "cabayol23_fixp3d"
        )
        identity_fields = ("z", "ind_snap", "ind_phase", "ind_axis", "ind_rescaling")

        def identity_value(value: Any) -> Any:
            """Normalize numpy scalars while retaining string-valued axes."""
            if isinstance(value, np.generic):
                value = value.item()
            return round(float(value), 10) if isinstance(value, float) else value

        def snapshot_identity(source: dict[str, Any]) -> tuple[Any, ...]:
            return tuple(identity_value(source[field]) for field in identity_fields)

        labels = {snapshot["sim_label"] for snapshot in archive}
        for sim_label in labels:
            fit_file = os.path.join(fit_folder, f"Arinyo_fit_{sim_label}.npy")
            if not os.path.isfile(fit_file):
                continue
            fit = np.load(fit_file, allow_pickle=True).item()
            if fit.get("simulation_label") != sim_label:
                raise ValueError(
                    f"Corrected Arinyo fit {fit_file} is labelled "
                    f"{fit.get('simulation_label')!r}, not {sim_label!r}."
                )
            if fit.get("postproc") != "Cabayol23_fixp3d":
                raise ValueError(
                    f"Corrected Arinyo fit {fit_file} has the wrong postproc."
                )
            missing = [field for field in identity_fields if field not in fit]
            if missing:
                raise KeyError(f"Corrected Arinyo fit {fit_file} lacks {missing}.")

            rows = {}
            for index in range(len(fit["z"])):
                identity = snapshot_identity(
                    {field: fit[field][index] for field in identity_fields}
                )
                if identity in rows:
                    raise ValueError(
                        f"Duplicate snapshot identity in {fit_file}: {identity}."
                    )
                rows[identity] = index

            for snapshot in archive:
                if snapshot["sim_label"] != sim_label:
                    continue
                identity = snapshot_identity(snapshot)
                if identity not in rows:
                    raise KeyError(
                        f"No corrected Arinyo fit in {fit_file} for snapshot {identity}."
                    )
                row = rows[identity]
                snapshot["arinyo_fixp3d"] = {
                    name: float(values[row]) for name, values in fit["Arinyo"].items()
                }

    def add_Arinyo_minimizer_indiv(
        self,
        archive: Any,
        sim_label: Any | None = None,
        kmax_3d: int | None = 5,
        kmax_1d: int | None = 4,
    ) -> Any:
        """
        Arinyo fits considering each snapshot separately

        Parameters
        ----------
        archive : object
            Simulation archive containing the requested data.
        sim_label : object, optional
            Sim label used by the calculation.
        kmax_3d : int, optional
            Maximum three-dimensional wavenumber included in the fit.
        kmax_1d : int, optional
            Maximum one-dimensional wavenumber included in the fit.

        Returns
        -------
        object
            Result produced when the function is used to arinyo fits considering each snapshot separately.
        """

        def get_flag_out(
            ind_sim: Any, kmax_3d: int | float, kmax_1d: int | float
        ) -> Any:
            """
            Return flag out.

            Parameters
            ----------
            ind_sim : object
                Ind sim used by the calculation.
            kmax_3d : int or float
                Maximum three-dimensional wavenumber included in the fit.
            kmax_1d : int or float
                Maximum one-dimensional wavenumber included in the fit.

            Returns
            -------
            object
                Result produced when the function is used to return flag out.
            """
            flag = (
                "fit_sim_label_"
                + str(ind_sim)
                + "_kmax3d_"
                + str(kmax_3d)
                + "_kmax1d_"
                + str(kmax_1d)
            )
            return flag

        if sim_label == "mpg_hypercube":
            ii = 0
            for isim in range(30):
                ind_sim = archive[ii]["sim_label"]
                flag = get_flag_out(ind_sim, kmax_3d, kmax_1d)
                file = os.path.join(
                    self.base_folder,
                    "data",
                    "best_arinyo",
                    "minimizer",
                    flag + ".npz",
                )
                data = np.load(file, allow_pickle=True)
                best_params = data["best_params"]
                ind_snap = data["ind_snap"]
                val_scaling = data["val_scaling"]

                nelem = len(best_params)
                for jj in range(nelem):
                    if ind_sim != archive[ii]["sim_label"]:
                        raise ValueError("sim_label does not match")

                    ind = np.argwhere(
                        (ind_snap == archive[ii]["ind_snap"])
                        & (val_scaling == archive[ii]["val_scaling"])
                    )[0, 0]

                    archive[ii]["Arinyo_min"] = best_params[ind]
                    archive[ii]["Arinyo_min"]["bias"] = -np.abs(
                        archive[ii]["Arinyo_min"]["bias"]
                    )
                    # bias_eta = bias * beta / fz
                    archive[ii]["Arinyo_min"]["bias_eta"] = (
                        archive[ii]["Arinyo_min"]["bias"]
                        * archive[ii]["Arinyo_min"]["beta"]
                        / archive[ii]["f_p"]
                    )
                    archive[ii]["Arinyo_min"]["q1"] = np.abs(
                        archive[ii]["Arinyo_min"]["q1"]
                    )
                    archive[ii]["Arinyo_min"]["q2"] = np.abs(
                        archive[ii]["Arinyo_min"]["q2"]
                    )
                    ii += 1
        else:
            flag = get_flag_out(sim_label, kmax_3d, kmax_1d)
            file = os.path.join(
                self.base_folder,
                "data",
                "best_arinyo",
                "minimizer",
                flag + ".npz",
            )
            data = np.load(file, allow_pickle=True)
            best_params = data["best_params"]
            ind_snap = data["ind_snap"]
            val_scaling = data["val_scaling"]

            nelem = len(best_params)
            for ii in range(nelem):
                ind = np.argwhere(
                    (ind_snap == archive[ii]["ind_snap"])
                    & (val_scaling == archive[ii]["val_scaling"])
                )[0, 0]
                archive[ii]["Arinyo_min"] = best_params[ind]
                archive[ii]["Arinyo_min"]["bias"] = -np.abs(
                    archive[ii]["Arinyo_min"]["bias"]
                )
                # bias_eta = bias * beta / fz
                archive[ii]["Arinyo_min"]["bias_eta"] = (
                    archive[ii]["Arinyo_min"]["bias"]
                    * archive[ii]["Arinyo_min"]["beta"]
                    / archive[ii]["f_p"]
                )
                archive[ii]["Arinyo_min"]["q1"] = np.abs(
                    archive[ii]["Arinyo_min"]["q1"]
                )
                archive[ii]["Arinyo_min"]["q2"] = np.abs(
                    archive[ii]["Arinyo_min"]["q2"]
                )

    def add_Arinyo_minimizer_indiv_lowk(
        self, archive: Any, sim_label: str | None = "mpg_hypercube"
    ) -> None:
        """
        Add Arinyo minimizer indiv lowk.

        Parameters
        ----------
        archive : object
            Simulation archive containing the requested data.
        sim_label : str, optional
            Sim label used by the calculation.
        """
        # The seed simulation has no standalone low-k Arinyo fit yet.  Its
        # measurements remain untouched; only its initial conditions use the
        # already available central-simulation low-k fit.
        source_label = "mpg_central" if sim_label == "mpg_seed" else sim_label
        name_out = "Arinyo_fit_" + source_label + "_lowk.npy"
        file = os.path.join(
            self.base_folder, "data", "best_arinyo", "minimizer_lowk", name_out
        )
        if sim_label == "mpg_seed":
            print(
                "Using mpg_central low-k Arinyo fit only as the initial "
                "condition for mpg_seed."
            )
        data = np.load(file, allow_pickle=True).item()
        n_available = next(iter(data["Arinyo"].values())).shape[0]
        if len(archive) > n_available:
            raise ValueError(
                f"Low-k Arinyo fit {file} has {n_available} snapshots, but "
                f"{sim_label} requested {len(archive)}."
            )

        for isim in range(len(archive)):
            archive[isim]["Arinyo_lowk"] = {}
            for par in data["Arinyo"]:
                archive[isim]["Arinyo_lowk"][par] = data["Arinyo"][par][isim, 0]
            archive[isim]["Arinyo_lowk"]["beta"] = (
                archive[isim]["f_p"]
                * archive[isim]["Arinyo_lowk"]["bias_eta"]
                / archive[isim]["Arinyo_lowk"]["bias"]
            )

    def add_Arinyo_minimizer_joint(
        self,
        archive: Any,
        sim_label: Any | None = None,
        kmax_3d: int | None = 3,
        kmax_1d: int | None = 3,
    ) -> Any:
        """
        Fits parameterizing the redshift dependence of the Arinyo params

        Parameters
        ----------
        archive : object
            Simulation archive containing the requested data.
        sim_label : object, optional
            Sim label used by the calculation.
        kmax_3d : int, optional
            Maximum three-dimensional wavenumber included in the fit.
        kmax_1d : int, optional
            Maximum one-dimensional wavenumber included in the fit.

        Returns
        -------
        object
            Result produced when the function is used to fits parameterizing the redshift dependence of the arinyo params.
        """

        def get_flag_out(
            ind_sim: Any, val_scaling: Any, kmax_3d: int | float, kmax_1d: int | float
        ) -> Any:
            """
            Return flag out.

            Parameters
            ----------
            ind_sim : object
                Ind sim used by the calculation.
            val_scaling : object
                Val scaling used by the calculation.
            kmax_3d : int or float
                Maximum three-dimensional wavenumber included in the fit.
            kmax_1d : int or float
                Maximum one-dimensional wavenumber included in the fit.

            Returns
            -------
            object
                Result produced when the function is used to return flag out.
            """
            flag = (
                "fit_sim_label_"
                + str(ind_sim)
                + "_val_scaling_"
                + str(np.round(val_scaling, 2))
                + "_kmax3d_"
                + str(kmax_3d)
                + "_kmax1d_"
                + str(kmax_1d)
            )
            return flag

        def paramz_to_paramind(z: int | float, paramz: Any) -> Any:
            """
            Convert to paramind.

            Parameters
            ----------
            z : int or float
                Redshift.
            paramz : object
                Paramz used by the calculation.

            Returns
            -------
            object
                Result produced when the function is used to convert to paramind.
            """
            paramind = []
            for ii in range(len(z)):
                param = {}
                for key in paramz:
                    param[key] = 10 ** np.poly1d(paramz[key])(z[ii])
                paramind.append(param)
            return paramind

        if sim_label == "mpg_hypercube":
            nsim = 30
            arr_val_scaling = [0.9, 0.95, 1.0, 1.05, 1.1]
        else:
            nsim = 1
            arr_val_scaling = [1.0]
        z = self.list_sim_redshifts.copy()

        id_sim_label = []
        id_val_scaling = []
        id_z = []
        arr_params = []

        for isim in range(nsim):
            if sim_label == "mpg_hypercube":
                ind_sim = "mpg_" + str(isim)
            else:
                ind_sim = sim_label
            for val_scaling in arr_val_scaling:
                flag = get_flag_out(ind_sim, val_scaling, kmax_3d, kmax_1d)
                file = os.path.join(
                    self.base_folder,
                    "data",
                    "best_arinyo",
                    "minimizer_z",
                    flag + ".npz",
                )
                data = np.load(file, allow_pickle=True)
                best_params = paramz_to_paramind(z, data["best_params"].item())
                for iz in range(len(z)):
                    id_sim_label.append(ind_sim)
                    id_val_scaling.append(np.round(val_scaling, 2))
                    id_z.append(np.round(z[iz], 2))
                    arr_params.append(best_params[iz])

        id_sim_label = np.array(id_sim_label)
        id_val_scaling = np.array(id_val_scaling)
        id_z = np.array(id_z)

        for ii in range(len(archive)):
            _ = np.argwhere(
                (archive[ii]["sim_label"] == id_sim_label)
                & (np.round(archive[ii]["z"], 2) == id_z)
                & (np.round(archive[ii]["val_scaling"], 2) == id_val_scaling)
            )[0, 0]
            archive[ii]["Arinyo_minz"] = params_numpy2dict_minimizerz(arr_params[_])

    def get_Arinyo_priors(
        self,
        zmin: float,
        zmax: float,
        type_fit: str | None = "Arinyo_min",
        return_all: bool | None = False,
    ) -> dict[str, Any]:
        """
        Compute summary priors for Arinyo fit parameters.

        Aggregates Arinyo fit parameter values from `self.training_data`
        within the redshift interval [zmin, zmax] and returns simple
        summary statistics (mean, std, min, max) for each parameter.

        Parameters
        ----------
        zmin, zmax : float
            Inclusive redshift bounds.
        type_fit : str, optional
            Key containing fitted parameters in each training-data entry.
        return_all : bool, optional
            Whether to also return the flattened parameter samples.

        Returns
        -------
        dict or tuple of dict
            Summary statistics, optionally paired with the raw flattened
            parameter samples.
        """
        # redshifts to be used
        ind_z = np.argwhere(
            (self.list_sim_redshifts >= zmin - 1e-3)
            & (self.list_sim_redshifts <= zmax + 1e-3)
        )[:, 0]
        print("Using data from redshifts:", self.list_sim_redshifts[ind_z])

        Nsim = len(self.list_sim_cube)
        Nz = len(ind_z)
        Nscalings = len(self.scalings_avail)

        # mapping from redshift to index
        conv_z_ind = {}
        for ii in range(Nz):
            conv_z_ind[self.list_sim_redshifts[ind_z[ii]]] = ii

        # mapping sim_label to index
        conv_sim_ind = {}
        for ii in range(len(self.list_sim_cube)):
            conv_sim_ind[self.list_sim_cube[ii]] = ii

        # create dict to store params
        data_priors = {}
        for par in self.training_data[0][type_fit]:
            data_priors[par] = np.zeros((Nsim, Nscalings, Nz))

        # fill dict
        for ii in range(len(self.training_data)):
            if self.training_data[ii]["z"] not in conv_z_ind:
                continue
            else:
                indz = conv_z_ind[self.training_data[ii]["z"]]
                indsim = conv_sim_ind[self.training_data[ii]["sim_label"]]
                indscal = self.training_data[ii]["ind_rescaling"]
            for par in self.training_data[ii][type_fit]:
                data_priors[par][indsim, indscal, indz] = self.training_data[ii][
                    type_fit
                ][par]

        for par in self.training_data[0][type_fit]:
            data_priors[par] = data_priors[par].reshape(-1)

        out_priors = {}
        out_priors["mean"] = {}
        out_priors["std"] = {}
        out_priors["min"] = {}
        out_priors["max"] = {}
        for par in data_priors:
            if par == "zs":
                continue

            if par == "bias":
                use_dat = -np.abs(data_priors[par])
            else:
                use_dat = data_priors[par]

            out_priors["mean"][par] = np.mean(use_dat)
            out_priors["std"][par] = np.std(use_dat)
            out_priors["min"][par] = np.min(use_dat)
            out_priors["max"][par] = np.max(use_dat)

        if return_all:
            return out_priors, data_priors
        else:
            return out_priors

    def get_IGM_priors(
        self,
        zmin: float,
        zmax: float,
        return_all: bool | None = False,
        IGM_params: str | None = None,
    ) -> dict[str, Any]:
        """
        Compute priors for a set of IGM parameters over redshift.

        Gathers the requested `IGM_params` from `self.training_data`
        within the redshift interval [zmin, zmax] and returns simple
        summary statistics (mean, std, min, max) for each parameter.

        Parameters
        ----------
        zmin, zmax : float
            Inclusive redshift bounds.
        return_all : bool, optional
            Whether to also return the flattened parameter samples.
        IGM_params : iterable of str, optional
            Parameters to include. By default, the standard cosmological and
            intergalactic-medium emulator inputs are used.

        Returns
        -------
        dict or tuple of dict
            Summary statistics, optionally paired with the raw flattened
            parameter samples.
        """

        if IGM_params is None:
            IGM_params = {
                "Delta2_p",
                "n_p",
                "mF",
                "sigT_Mpc",
                "gamma",
                "kF_Mpc",
            }

        # redshifts to be used
        ind_z = np.argwhere(
            (self.list_sim_redshifts >= zmin - 1e-3)
            & (self.list_sim_redshifts <= zmax + 1e-3)
        )[:, 0]
        print("Using data from redshifts:", self.list_sim_redshifts[ind_z])

        Nsim = len(self.list_sim_cube)
        Nz = len(ind_z)
        Nscalings = len(self.scalings_avail)

        # mapping from redshift to index
        conv_z_ind = {}
        for ii in range(Nz):
            conv_z_ind[self.list_sim_redshifts[ind_z[ii]]] = ii

        # mapping sim_label to index
        conv_sim_ind = {}
        for ii in range(len(self.list_sim_cube)):
            conv_sim_ind[self.list_sim_cube[ii]] = ii

        # create dict to store params
        data_priors = {}
        for par in IGM_params:
            data_priors[par] = np.zeros((Nsim, Nscalings, Nz))

        # fill dict
        for ii in range(len(self.training_data)):
            if self.training_data[ii]["z"] not in conv_z_ind:
                continue
            else:
                indz = conv_z_ind[self.training_data[ii]["z"]]
                indsim = conv_sim_ind[self.training_data[ii]["sim_label"]]
                indscal = self.training_data[ii]["ind_rescaling"]
            for par in IGM_params:
                data_priors[par][indsim, indscal, indz] = self.training_data[ii][par]

        for par in IGM_params:
            data_priors[par] = data_priors[par].reshape(-1)

        out_priors = {}
        out_priors["mean"] = {}
        out_priors["std"] = {}
        out_priors["min"] = {}
        out_priors["max"] = {}
        for par in data_priors:
            if par == "zs":
                continue

            if par == "bias":
                use_dat = -np.abs(data_priors[par])
            else:
                use_dat = data_priors[par]

            out_priors["mean"][par] = np.mean(use_dat)
            out_priors["std"][par] = np.std(use_dat)
            out_priors["min"][par] = np.min(use_dat)
            out_priors["max"][par] = np.max(use_dat)

        if return_all:
            return out_priors, data_priors
        else:
            return out_priors
