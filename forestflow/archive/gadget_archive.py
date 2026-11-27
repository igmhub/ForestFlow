"""
forestflow.archive.gadget_archive

Utilities to load and process Arinyo/Gadget archives for ForestFlow.

This module provides `GadgetArchive3D`, which extends
`lace.archive.gadget_archive.GadgetArchive` with helpers to load
training/testing data and Arinyo minimizer fits (both individual
and redshift-parameterized), and to summarize the Arinyo fits and emulator
inputs of the training simulations.
"""

from typing import Any

import numpy as np
import os
import warnings
from lace.archive.gadget_archive import GadgetArchive
import forestflow
from forestflow.statistics.rebin_p3d import MPG_P3D_BINNING, get_P3D_k_mu_bin_edges

# MP-Gadget snapshot convention used by ForestFlow fitting and archive tools.
# It is owned here rather than imported from LaCE so ForestFlow supports the
# released LaCE API as well as its development branch.
MPG_SIM_REDSHIFTS = np.arange(2.0, 4.6, 0.25)


class GadgetArchive3D(GadgetArchive):
    """
    Extend the LaCE MP-Gadget archive with ForestFlow P3D products.

    The archive owns native MP-Gadget P3D binning, relative-error products,
    ForestFlow emulator inputs, and optional per-snapshot Arinyo fits. Its
    snapshots use inverse comoving-Mpc wavenumbers and comoving-volume P3D.
    """

    def __init__(
        self,
        base_folder: Any | None = None,
        file_errors: Any | None = None,
        postproc: str | None = "Cabayol23_fixp3d",
        kp_Mpc: Any | None = None,
        average: str | None = "both",
        addcentral: bool | None = False,
    ) -> None:
        """
        Initialize an MP-Gadget P3D archive and attach standard fit data.

        Parameters
        ----------
        base_folder : str or path-like, optional
            ForestFlow repository root containing distributed archive products.
        file_errors : str or path-like, optional
            ``.npz`` file containing relative P1D/P3D error templates.
        postproc : str, default: "Cabayol23_fixp3d"
            Post-processing used by the calculation. Defaults to the corrected
            ``"Cabayol23_fixp3d"`` MP-Gadget P3D archive.
        kp_Mpc : float, optional
            Comoving linear-power pivot in ``1 / Mpc`` forwarded to LaCE.
        average : str, default: "both"
            Phase/axis averaging selection used to construct training data.
        addcentral : bool, default: False
            Append held-out central MP-Gadget snapshots to ``training_data``.
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
            self.add_arinyo_fixp3d(self.training_data)

        if addcentral:
            central_data = self.get_testing_data("mpg_central")
            self.training_data.extend(central_data)

    def get_P3D_k_mu_bin_edges(self, k_max_iMpc=None):
        """
        Return native MP-Gadget P3D bin edges used by this archive.

        Parameters
        ----------
        k_max_iMpc : float, optional
            Upper comoving wavenumber in ``1 / Mpc``. The standard maximum is
            used when omitted.

        Returns
        -------
        tuple of ndarray
            One-dimensional k and direction-cosine bin-edge arrays.
        """
        return get_P3D_k_mu_bin_edges(k_max_iMpc, **self.P3D_binning)

    def get_training_data(
        self,
        simulation_label: str | list[str] | None = None,
        emu_params: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Return MPG training snapshots with ForestFlow's standard inputs.

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

        Returns
        -------
        list of dict
            Training snapshots in inherited archive order, optionally filtered
            to requested simulation labels.

        Raises
        ------
        TypeError
            If labels or emulator parameters have unsupported types.
        ValueError
            If a requested label is not an MP-Gadget hypercube simulation.
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
            ``arinyo_fixp3d`` entries where available.
        """
        testing_data = super().get_testing_data(sim_label, ind_rescaling=ind_rescaling)
        self.add_Arinyo_minimizer_indiv(testing_data, sim_label, kmax_3d, kmax_1d)
        self.add_arinyo_fixp3d(testing_data)

        return testing_data

    def get_central_seed_average(self) -> list[dict[str, Any]]:
        """
        Return mean-flux-consistent averages of central and seed snapshots.

        The two MP-Gadget realizations have the same cosmology and native
        Fourier grid.  Scalar IGM summaries are averaged arithmetically,
        while P1D and P3D are first converted to absolute flux power using
        ``mF**2``, averaged, and converted back using the averaged ``mF``.
        The returned snapshots are labelled ``"mpg_central_seed"`` and retain
        the central fit as their initial condition.  If its corrected combined
        fit file exists, it is attached as ``arinyo_fixp3d``.

        Returns
        -------
        list of dict
            Combined central/seed snapshots, preserving central coordinates.

        Raises
        ------
        KeyError
            If either realization has missing or unmatched snapshot identities.
        ValueError
            If matching snapshots have incompatible grid shapes or zero mean
            flux.
        """
        central = self.get_testing_data("mpg_central")
        seed = self.get_testing_data("mpg_seed")
        identity_fields = ("z", "ind_snap", "ind_phase", "ind_axis", "ind_rescaling")

        def identity_value(value: Any) -> Any:
            """
            Normalize a scalar field for stable snapshot-identity matching.

            Parameters
            ----------
            value : object
                Snapshot identity-field value.

            Returns
            -------
            object
                Python scalar with float values rounded to ten decimals.
            """
            if isinstance(value, np.generic):
                value = value.item()
            return round(float(value), 10) if isinstance(value, float) else value

        def identity(snapshot: dict[str, Any]) -> tuple[Any, ...]:
            """
            Build the complete matching key for one archive snapshot.

            Parameters
            ----------
            snapshot : dict
                Archive snapshot containing all identity fields.

            Returns
            -------
            tuple
                Normalized redshift, snapshot, phase, axis, and rescaling key.
            """
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
        """
        Attach corrected-postprocessing Arinyo fits as ``arinyo_fixp3d``.

        Fits are matched to archive entries through their complete saved
        snapshot identity, rather than through list order.  This retains the
        corrected-fit parameters alongside the legacy ``Arinyo_min`` field.

        Parameters
        ----------
        archive : list of dict
            Snapshots mutated in place with matching ``arinyo_fixp3d``
            parameter mappings when corrected fit files are available.
            A ``UserWarning`` is issued for each simulation label without a
            corrected fit file.

        Raises
        ------
        KeyError
            If a fit lacks identity fields or a snapshot has no matching fit.
        ValueError
            If a corrected fit has incompatible metadata or duplicate rows.
        """
        fit_folder = os.path.join(
            self.base_folder, "data", "best_arinyo", "cabayol23_fixp3d"
        )
        identity_fields = ("z", "ind_snap", "ind_phase", "ind_axis", "ind_rescaling")

        def identity_value(value: Any) -> Any:
            """
            Normalize one identity component while retaining string axes.

            Parameters
            ----------
            value : object
                Snapshot or fit identity-field value.

            Returns
            -------
            object
                Python scalar with floats rounded to ten decimals.
            """
            if isinstance(value, np.generic):
                value = value.item()
            return round(float(value), 10) if isinstance(value, float) else value

        def snapshot_identity(source: dict[str, Any]) -> tuple[Any, ...]:
            """
            Build a normalized identity key from snapshot-like data.

            Parameters
            ----------
            source : dict
                Mapping containing required identity fields.

            Returns
            -------
            tuple
                Normalized identity components in archive matching order.
            """
            return tuple(identity_value(source[field]) for field in identity_fields)

        labels = {snapshot["sim_label"] for snapshot in archive}
        for sim_label in labels:
            fit_file = os.path.join(fit_folder, f"Arinyo_fit_{sim_label}.npy")
            if not os.path.isfile(fit_file):
                warnings.warn(
                    f"No corrected Arinyo fit found for '{sim_label}' ({fit_file}); "
                    "its snapshots will have no 'arinyo_fixp3d' entry.",
                    UserWarning,
                    stacklevel=3,
                )
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
        Attach independently fitted Arinyo parameters to archive snapshots.

        Parameters
        ----------
        archive : sequence of dict
            Snapshots mutated in place with ``Arinyo_min`` mappings.
        sim_label : str, optional
            Simulation label. ``"mpg_hypercube"`` loads all 30 hypercube fit
            files; another label loads its single fit product.
        kmax_3d, kmax_1d : float, default: 5, 4
            P3D/P1D fitting cuts in ``1 / Mpc`` encoded in fit filenames.

        Raises
        ------
        ValueError
            If an archive entry disagrees with its expected hypercube label.
        """

        def get_flag_out(
            ind_sim: Any, kmax_3d: int | float, kmax_1d: int | float
        ) -> Any:
            """
            Build the stored filename stem for one independent fit.

            Parameters
            ----------
            ind_sim : str
                Simulation label encoded in the fit product name.
            kmax_3d : int or float
                Maximum three-dimensional wavenumber included in the fit.
            kmax_1d : int or float
                Maximum one-dimensional wavenumber included in the fit.

            Returns
            -------
            str
                Independent-fit product filename stem.
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

    def _summarize_training_values(
        self, zmin: float, zmax: float, values: Any, names: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Summarize per-snapshot training values over a redshift interval.

        Parameters
        ----------
        zmin, zmax : float
            Inclusive redshift bounds.
        values : callable
            ``values(snapshot)`` returns the mapping holding the parameters.
        names : iterable of str
            Parameters to summarize.

        Returns
        -------
        summary, samples : dict
            ``summary`` has ``mean``, ``std``, ``min`` and ``max`` mappings;
            ``samples`` the flattened (simulation, rescaling, redshift) values.
            ``bias`` is summarized with a negative sign.
        """
        ind_z = np.argwhere(
            (self.list_sim_redshifts >= zmin - 1e-3)
            & (self.list_sim_redshifts <= zmax + 1e-3)
        )[:, 0]
        print("Using data from redshifts:", self.list_sim_redshifts[ind_z])

        conv_z_ind = {self.list_sim_redshifts[iz]: ii for ii, iz in enumerate(ind_z)}
        conv_sim_ind = {label: ii for ii, label in enumerate(self.list_sim_cube)}
        shape = (len(self.list_sim_cube), len(self.scalings_avail), len(ind_z))
        samples = {par: np.zeros(shape) for par in names}

        for snapshot in self.training_data:
            if snapshot["z"] not in conv_z_ind:
                continue
            index = (
                conv_sim_ind[snapshot["sim_label"]],
                snapshot["ind_rescaling"],
                conv_z_ind[snapshot["z"]],
            )
            for par in names:
                samples[par][index] = values(snapshot)[par]

        summary = {"mean": {}, "std": {}, "min": {}, "max": {}}
        for par in samples:
            samples[par] = samples[par].reshape(-1)
            if par == "zs":
                continue
            use = -np.abs(samples[par]) if par == "bias" else samples[par]
            summary["mean"][par] = np.mean(use)
            summary["std"][par] = np.std(use)
            summary["min"][par] = np.min(use)
            summary["max"][par] = np.max(use)
        return summary, samples

    def get_arinyo_summary(
        self,
        zmin: float,
        zmax: float,
        type_fit: str | None = "arinyo_fixp3d",
        return_all: bool | None = False,
    ) -> dict[str, Any]:
        """
        Summarize the Arinyo fits of the training simulations.

        Not to be confused with :func:`forestflow.priors.get_arinyo_priors`,
        which reads priors derived from the DESI P1D posterior.

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
            Summary statistics (mean, std, min, max), optionally paired with
            the raw flattened parameter samples.
        """
        summary, samples = self._summarize_training_values(
            zmin,
            zmax,
            lambda snapshot: snapshot[type_fit],
            list(self.training_data[0][type_fit]),
        )
        return (summary, samples) if return_all else summary

    def get_input_summary(
        self,
        zmin: float,
        zmax: float,
        return_all: bool | None = False,
        IGM_params: Any | None = None,
    ) -> dict[str, Any]:
        """
        Summarize the emulator inputs of the training simulations.

        Not to be confused with :func:`forestflow.priors.get_IGM_priors`,
        which reads priors derived from the DESI P1D posterior.

        Parameters
        ----------
        zmin, zmax : float
            Inclusive redshift bounds.
        return_all : bool, optional
            Whether to also return the flattened parameter samples.
        IGM_params : iterable of str, optional
            Parameters to include; by default the six emulator inputs.

        Returns
        -------
        dict or tuple of dict
            Summary statistics (mean, std, min, max), optionally paired with
            the raw flattened parameter samples.
        """
        if IGM_params is None:
            IGM_params = self.emu_params
        summary, samples = self._summarize_training_values(
            zmin, zmax, lambda snapshot: snapshot, list(IGM_params)
        )
        return (summary, samples) if return_all else summary
