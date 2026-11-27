"""
P1D interface built from a ForestFlow P3D emulator.

This adapter is owned by ForestFlow.  It turns predicted Arinyo parameters and
linear theory into P1D predictions for likelihood clients such as cup1d.
"""

from collections.abc import Mapping
import numpy as np
from threadpoolctl import threadpool_limits

from .loading import load_emulator
from .labels import DEFAULT_EMULATOR_KEY


class P1DEmulator:
    """
    Adapt a ForestFlow P3D bundle into a stateful P1D emulator.

    The cosmology and linear theory are attached lazily, allowing this module
    to be imported without initializing the LaCE cosmology stack.
    """

    def __init__(
        self,
        name_emu=DEFAULT_EMULATOR_KEY,
        compile_model=True,
        *,
        sampler="sobol_antithetic",
        statistic="mean",
        aggregation_space="transformed",
        draw_policy="nested",
        seed=0,
        Nrealizations=None,
    ):
        """
        Load a named P3D bundle.

        Parameters
        ----------
        name_emu : str, default=DEFAULT_EMULATOR_KEY
            Emulator label, e.g. ``"forest_mpg_cinn"`` (cINN) or
            ``"forest_mpg_gp"`` (Gaussian process); see
            :func:`forestflow.emulator.load_emulator`.
        compile_model : bool, default=True
            Whether to compile the neural-network model when supported;
            ignored by emulators that are not neural networks.
        sampler, statistic, aggregation_space, draw_policy, seed, Nrealizations
            Latent-sampling options of a cINN (``sampling_options``, passed to
            its ``evaluate``); ignored by deterministic emulators (GP).
            ``Nrealizations=None`` uses the bundle default.
        """

        self.emulator = load_emulator(name_emu, compile_model=compile_model)

        self.emu_params = self.emulator.input_labels
        self.emulator_label = name_emu
        # cup1d uses these attributes only when it constructs a mock from a
        # simulation P1D. Reuse the exact CH24 MPG smoothing calibration used
        # by ``lace_mpg`` so both validation paths prepare the same mock.
        self._mock_smoothing_emulator = None

        self.cosmo_params_dict = None
        self.model_Arinyo = None
        self.linear = None
        self._linear_cosmology_parameters = None
        self.sampling_options = {
            "sampler": sampler,
            "statistic": statistic,
            "aggregation_space": aggregation_space,
            "draw_policy": draw_policy,
            "seed": seed,
            "Nrealizations": Nrealizations,
        }

    def _get_mock_smoothing_emulator(self):
        """Return LaCE's MPG mock-smoothing calibration on demand."""

        if self._mock_smoothing_emulator is None:
            from lace.emulator import emulator_manager

            self._mock_smoothing_emulator = emulator_manager.set_emulator(
                emulator_label="CH24_mpgcen_gpr"
            )
        return self._mock_smoothing_emulator

    @property
    def emu_type(self):
        """LaCE smoothing representation used for synthetic MPG mocks."""

        return self._get_mock_smoothing_emulator().emu_type

    @property
    def kmax_Mpc(self):
        """Maximum wavenumber of the MPG mock-smoothing calibration."""

        return self._get_mock_smoothing_emulator().kmax_Mpc

    @property
    def input_norm(self):
        """Normalization grid used by the MPG mock-smoothing calibration."""

        return self._get_mock_smoothing_emulator().input_norm

    @property
    def norm_imF(self):
        """LaCE MPG mean-flux normalization callable for mock smoothing."""

        return self._get_mock_smoothing_emulator().norm_imF

    @property
    def func_poly(self):
        """LaCE MPG polynomial callable, retaining its fitted signature."""

        return self._get_mock_smoothing_emulator().func_poly

    def set_sampling_options(self, **options):
        """Update the P3D latent-sampling settings."""

        allowed = set(self.sampling_options)
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(f"Unknown sampling options: {sorted(unknown)}")
        self.sampling_options.update(options)

    def set_cosmology(self, cosmology):
        """
        Set the fiducial cosmology of subsequent Arinyo and P1D predictions.

        Parameters
        ----------
        cosmology : mapping or lace.cosmo.base_cosmology.BaseCosmology
            Cosmological parameters, or the LaCE cosmology itself. Passing the
            object a caller rescales (e.g. cup1d's fiducial) avoids a second
            CAMB run and enables the fiducial linear-power cache.
        """
        # Kept lazy so a P3D-only ForestFlow installation does not need LaCE.
        from lace.cosmo import cosmology as lace_cosmology

        from forestflow.model import ArinyoModel

        if isinstance(cosmology, Mapping):
            self.cosmo_params_dict = dict(cosmology)
            fid_cosmo = lace_cosmology.Cosmology(cosmo_params_dict=self.cosmo_params_dict)
        else:
            fid_cosmo = cosmology
            self.cosmo_params_dict = dict(fid_cosmo.input_cosmo_params_dict)
        self.model_Arinyo = ArinyoModel(fid_cosmo)
        self.linear = None
        self._linear_cosmology_parameters = None

    def _effective_cosmology_parameters(self, new_cosmo_params):
        """
        Return the complete cosmology represented by one linear grid.
        """

        parameters = dict(self.cosmo_params_dict)
        if new_cosmo_params is not None:
            parameters.update(new_cosmo_params)
        return parameters

    def set_linear_theory(self, z, new_cosmo_params=None, cosmology=None):
        """
        Build or reuse linear theory at the requested redshifts.

        Parameters
        ----------
        z : array_like
            Redshifts requiring linear-theory products.
        new_cosmo_params : mapping, optional
            Cosmological changes relative to the configured fiducial model.
        cosmology : lace.cosmo.base_cosmology.BaseCosmology, optional
            The cosmology itself, instead of ``new_cosmo_params``.
        """

        if self.model_Arinyo is None or self.cosmo_params_dict is None:
            raise RuntimeError("Call set_cosmology before evaluating ForestFlow P1D")
        zuse = np.unique(np.atleast_1d(np.asarray(z, dtype=float)))
        if cosmology is not None:
            self.linear = self.model_Arinyo.linear.get_linear_theory(zuse, cosmology=cosmology)
            self._linear_cosmology_parameters = None
            return
        requested_cosmology = self._effective_cosmology_parameters(
            new_cosmo_params
        )

        if (
            self.linear is not None
            and _same_cosmology(
                self._linear_cosmology_parameters, requested_cosmology
            )
            and np.array_equal(zuse, self.linear.z)
        ):
            return

        self.linear = self.model_Arinyo.linear.get_linear_theory(
            zuse, new_cosmo_params=new_cosmo_params
        )
        self._linear_cosmology_parameters = requested_cosmology

    def emulate_p1d_Mpc(
        self, zs, k_iMpc, emulator_parameters, cosmology_parameters=None,
        cosmologies=None,
    ):
        """
        P1D in Mpc for one cosmology or a batch of cosmologies.

        Parameters
        ----------
        zs : array_like
            Redshifts.
        k_iMpc : ndarray
            ``(n_z, n_k)`` for the linear theory set by
            :meth:`set_linear_theory`, or ``(n_batch, n_z, n_k)`` with one
            cosmology per row (``cosmologies``: LaCE cosmologies, or
            ``cosmology_parameters``: changes to the fiducial).
        emulator_parameters : mapping
            Emulator inputs, ``(n_z,)`` or ``(n_batch, n_z)``.

        Returns
        -------
        ndarray
            P1D in Mpc with the shape of ``k_iMpc``.
        """
        if self.model_Arinyo is None:
            raise RuntimeError("Call set_cosmology before evaluating ForestFlow P1D")
        zs = np.atleast_1d(zs)
        batched = np.asarray(k_iMpc).ndim == 3
        if batched:
            if cosmology_parameters is None and cosmologies is None:
                raise ValueError(
                    "cosmology_parameters or cosmologies is required for batched ForestFlow P1D"
                )
            linear = self.model_Arinyo.linear.get_linear_theory_batch(
                zs, cosmology_parameters, cosmologies=cosmologies
            )
        else:
            if cosmology_parameters is not None or cosmologies is not None:
                raise ValueError(
                    "cosmology_parameters/cosmologies need (n_batch, n_z, n_k) wavenumbers"
                )
            if self.linear is None:
                raise RuntimeError("Call set_linear_theory before evaluating ForestFlow P1D")
            linear = self.linear
        arinyo = self.evaluate_arinyo(emulator_parameters)
        return self.model_Arinyo.P1D_Mpc(linear, zs, k_iMpc, arinyo)

    def evaluate_arinyo(self, emulator_parameters):
        """
        Arinyo coefficients for emulator inputs of any shape.

        The one Arinyo evaluation of ForestFlow's P1D (used by
        :meth:`emulate_p1d_Mpc` and by external callers such as the Cobaya
        theory of lya_interface). A sampling emulator (cINN) draws the same
        latent block (index 0) for every input row, so a row's prediction does
        not depend on the other rows of the call (batch, redshift order or
        chunking).

        Parameters
        ----------
        emulator_parameters : mapping
            Emulator inputs (``self.emu_params``), all of one shape: a scalar,
            ``(n_z,)`` or ``(n_batch, n_z)``.

        Returns
        -------
        dict
            Arinyo coefficients keyed by output label, with the shape of the
            inputs.
        """
        values = {name: np.asarray(emulator_parameters[name], dtype=float) for name in self.emu_params}
        shape = values[self.emu_params[0]].shape
        rows = [
            {name: values[name].reshape(-1)[index] for name in self.emu_params}
            for index in range(values[self.emu_params[0]].size)
        ]
        if getattr(self.emulator, "uses_sampling", True):
            with threadpool_limits(limits=1):
                output = self.emulator.evaluate(
                    rows, latent_indices=np.zeros(len(rows), dtype=int), **self.sampling_options
                )
        else:
            # deterministic emulators (GP) have no latent draws to configure
            output = self.emulator.evaluate(rows)
        return {
            name: np.asarray(output[name], dtype=float).reshape(shape)
            for name in self.emulator.output_labels
        }

    @property
    def kp_iMpc(self):
        """
        Pivot wavenumber stored in the ForestFlow emulator metadata.
        """
        return self.emulator.kp_iMpc

    @property
    def kmax_3d_iMpc(self):
        """
        Maximum P3D fitting cut stored in the emulator metadata.
        """
        return self.emulator.kmax_3d_iMpc

    @property
    def kmax_1d_iMpc(self):
        """
        Maximum P1D fitting cut stored in the emulator metadata.
        """
        return self.emulator.kmax_1d_iMpc

    @property
    def zmax(self):
        """
        Maximum redshift stored in the ForestFlow emulator metadata.
        """
        return self.emulator.zmax

    @property
    def list_sim_cube(self):
        """
        Training simulations stored in the ForestFlow emulator metadata.
        """
        return self.emulator.list_sim_cube

    @property
    def kp_Mpc(self):
        """
        Common emulator API name; its units are inverse Mpc.
        """
        return self.kp_iMpc


def _same_cosmology(first, second):
    """
    Compare effective cosmology mappings, including their key sets.
    """

    if first is None or second is None or first.keys() != second.keys():
        return False
    return all(
        np.array_equal(np.asarray(first[key]), np.asarray(second[key]))
        for key in first
    )
