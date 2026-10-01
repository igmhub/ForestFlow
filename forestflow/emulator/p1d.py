"""P1D interface built from a ForestFlow P3D emulator.

This adapter is owned by ForestFlow.  It turns predicted Arinyo parameters and
linear theory into P1D predictions for likelihood clients such as cup1d.
"""

import numpy as np
from threadpoolctl import threadpool_limits

from .p3d_cinn import P3DEmulator


class P1DEmulator:
    def __init__(self, name_emu="forest_mpg_fix", compile_model=True):

        self.emulator = P3DEmulator(key=name_emu, compile_model=compile_model)

        self.emu_params = self.emulator.input_labels
        self.emulator_label = name_emu

        self.cosmo_params_dict = None
        self.model_Arinyo = None
        self.linear = None
        self._linear_cosmology_parameters = None
        self._prediction_cache = None

    def set_cosmology(self, cosmo_params_dict):
        # Kept lazy so a P3D-only ForestFlow installation does not need LaCE.
        from lace.cosmo import cosmology

        from forestflow.model import ArinyoModel

        self.cosmo_params_dict = dict(cosmo_params_dict)
        fid_cosmo = cosmology.Cosmology(cosmo_params_dict=self.cosmo_params_dict)
        self.model_Arinyo = ArinyoModel(fid_cosmo)
        self.linear = None
        self._linear_cosmology_parameters = None

    def _effective_cosmology_parameters(self, new_cosmo_params):
        """Return the complete cosmology represented by one linear grid."""

        parameters = dict(self.cosmo_params_dict)
        if new_cosmo_params is not None:
            parameters.update(new_cosmo_params)
        return parameters

    def set_linear_theory(self, z, new_cosmo_params=None):

        if self.model_Arinyo is None or self.cosmo_params_dict is None:
            raise RuntimeError("Call set_cosmology before evaluating ForestFlow P1D")
        zuse = np.unique(np.atleast_1d(np.asarray(z, dtype=float)))
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

    def _prediction_key(self, parameters, latent_index=None):
        """Return a stable key for one set of emulator inputs."""

        values = tuple(float(parameters[name]) for name in self.emu_params)
        return (latent_index, values) if latent_index is not None else values

    def _evaluate_emulator(self, emulator_calls, **kwargs):
        """Evaluate ForestFlow efficiently for cup1d's small CPU batches."""

        # A likelihood point is independent work. For the small network
        # batches used here, thread start-up costs more than it saves; sampler
        # or MPI parallelism remains available across likelihood points.
        with threadpool_limits(limits=1):
            return self.emulator.evaluate(emulator_calls, **kwargs)

    def prime_prediction_cache(self, emulator_calls):
        """Evaluate many redshift inputs in one ForestFlow network batch."""

        unique_inputs = {}
        for emulator_call in emulator_calls:
            n_redshifts = np.asarray(emulator_call[self.emu_params[0]]).size
            for index in range(n_redshifts):
                parameters = {
                    name: np.asarray(emulator_call[name]).reshape(-1)[index]
                    for name in self.emu_params
                }
                key = self._prediction_key(parameters)
                unique_inputs.setdefault(key, parameters)

        if not unique_inputs:
            self._prediction_cache = {}
            return

        self._prediction_cache = {}
        items = list(unique_inputs.items())
        for start in range(0, len(items), 128):
            chunk = items[start : start + 128]
            keys = [item[0] for item in chunk]
            predictions = self._evaluate_emulator([item[1] for item in chunk])
            for index, key in enumerate(keys):
                self._prediction_cache[key] = {
                    name: np.asarray(predictions[name]).reshape(-1)[index]
                    for name in self.emulator.output_labels
                }

    def clear_prediction_cache(self):
        """Discard predictions retained for one batched likelihood call."""

        self._prediction_cache = None

    def emulate_p1d_Mpc(
        self, zs, k_iMpc, emulator_parameters, cosmology_parameters=None
    ):
        """Return P1D in Mpc for scalar or leading-batch inputs.

        A two-dimensional ``k_iMpc`` array is one evaluation over redshift. A
        three-dimensional ``(batch, redshift, k)`` array dispatches internally
        to the batched implementation and requires ``cosmology_parameters``.
        Thus callers use one public P1D method regardless of input shape.
        """
        if np.asarray(k_iMpc).ndim == 3:
            if cosmology_parameters is None:
                raise ValueError(
                    "cosmology_parameters is required for batched ForestFlow P1D"
                )
            return self._emulate_p1d_Mpc_batch(
                zs, k_iMpc, emulator_parameters, cosmology_parameters
            )
        if cosmology_parameters is not None:
            raise ValueError(
                "cosmology_parameters is only accepted for batched ForestFlow P1D"
            )
        return self._emulate_p1d_Mpc_scalar(zs, k_iMpc, emulator_parameters)

    def _emulate_p1d_Mpc_scalar(self, zs, k_iMpc, emulator_parameters):
        """Evaluate the scalar/redshift-vector P1D implementation."""
        if self.linear is None:
            raise RuntimeError(
                "Call set_linear_theory before evaluating ForestFlow P1D"
            )
        in_params = {
            name: np.asarray(emulator_parameters[name]) for name in self.emu_params
        }
        list_dicts = []
        nin = in_params[self.emu_params[0]].reshape(-1).size
        for ii in range(nin):
            in_par_only = {}
            for par in self.emu_params:
                in_par_only[par] = in_params[par].reshape(-1)[ii]
            list_dicts.append(in_par_only)
        if self._prediction_cache is None:
            out_emu = self._evaluate_emulator(list_dicts)
        else:
            cached = [
                self._prediction_cache[self._prediction_key(parameters)]
                for parameters in list_dicts
            ]
            out_emu = {
                name: np.asarray([prediction[name] for prediction in cached])
                for name in self.emulator.output_labels
            }

        list_P1D_Mpc = self.model_Arinyo.P1D_Mpc(
            self.linear,
            zs,
            k_iMpc,
            out_emu,
        )

        return list_P1D_Mpc

    def _emulate_p1d_Mpc_batch(
        self, zs, k_iMpc, emulator_parameters, cosmo_params_batch
    ):
        """Evaluate ForestFlow for ``(batch, redshift, k)`` inputs.

        The cINN is evaluated once over flattened batch/redshift rows. Linear
        theory remains one inexpensive rescaling per cosmology because each
        Arinyo integration owns a distinct linear grid.
        """
        if self.model_Arinyo is None:
            raise RuntimeError("Call set_cosmology before evaluating ForestFlow P1D")
        zs = np.atleast_1d(zs)
        n_batch, n_z, _ = np.asarray(k_iMpc).shape
        calls = [
            {
                name: np.asarray(emulator_parameters[name])[ib, iz]
                for name in self.emu_params
            }
            for ib in range(n_batch)
            for iz in range(n_z)
        ]
        output = self._evaluate_emulator(
            calls, latent_indices=np.tile(np.arange(n_z), n_batch)
        )
        arinyo = {
            name: np.asarray(values).reshape(n_batch, n_z)
            for name, values in output.items()
        }
        linear = self.model_Arinyo.linear.get_linear_theory_batch(zs, cosmo_params_batch)
        return self.model_Arinyo.P1D_Mpc(linear, zs, k_iMpc, arinyo)

    @property
    def kp_iMpc(self):
        """Pivot wavenumber stored in the ForestFlow emulator metadata."""
        return self.emulator.kp_iMpc

    @property
    def kmax_3d_iMpc(self):
        """Maximum P3D fitting cut stored in the emulator metadata."""
        return self.emulator.kmax_3d_iMpc

    @property
    def kmax_1d_iMpc(self):
        """Maximum P1D fitting cut stored in the emulator metadata."""
        return self.emulator.kmax_1d_iMpc

    @property
    def zmax(self):
        """Maximum redshift stored in the ForestFlow emulator metadata."""
        return self.emulator.zmax

    @property
    def list_sim_cube(self):
        """Training simulations stored in the ForestFlow emulator metadata."""
        return self.emulator.list_sim_cube

    @property
    def kp_Mpc(self):
        """Common emulator API name; its units are inverse Mpc."""
        return self.kp_iMpc


def _same_cosmology(first, second):
    """Compare effective cosmology mappings, including their key sets."""

    if first is None or second is None or first.keys() != second.keys():
        return False
    return all(
        np.array_equal(np.asarray(first[key]), np.asarray(second[key]))
        for key in first
    )
