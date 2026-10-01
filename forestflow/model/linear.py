"""LaCE-backed linear-theory adapters used by ForestFlow models."""

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from lace.cosmo import cosmology, rescale_cosmology


@dataclass(slots=True)
class LinearTheoryGrid:
    """
    One or more LaCE cosmology objects associated with redshift samples.

    The historical name is retained for API continuity.  This object no longer
    stores or interpolates a duplicate ForestFlow linear-power grid: LaCE is
    the sole implementation of both linear power and growth rate.
    """

    z: np.ndarray
    cosmology: Any | None = None
    cosmologies: tuple[Any, ...] | None = None

    @property
    def is_batched(self) -> bool:
        """Whether this adapter carries one LaCE cosmology per batch row."""
        return self.cosmologies is not None


class LinearTheory:
    """
    Create scalar or batched ForestFlow views of LaCE cosmologies.

    This class owns no Boltzmann calculation or interpolation scheme.  Its
    only role is to package LaCE cosmology objects for ForestFlow's scalar and
    batched model APIs.
    """

    def __init__(self, fiducial_cosmology=None):
        self.fiducial_cosmology = fiducial_cosmology or cosmology.Cosmology()

    def _cosmology_for_parameters(self, new_cosmo_params=None):
        # ``nnu`` changes the radiation content and transfer functions.  Some
        # supported LaCE releases did not include it in ``same_background``,
        # so do not permit its primordial-only rescaling path here.
        if new_cosmo_params is not None and "nnu" in new_cosmo_params:
            print("WARNING: computing CAMB again")
            return cosmology.Cosmology(cosmo_params_dict=new_cosmo_params)
        if self.fiducial_cosmology.same_background(
            cosmo_params=new_cosmo_params
        ):
            return rescale_cosmology.RescaledCosmology(
                self.fiducial_cosmology, new_cosmo_params
            )
        print("WARNING: computing CAMB again")
        return cosmology.Cosmology(cosmo_params_dict=new_cosmo_params)

    def get_linear_theory(self, zs, new_cosmo_params=None, **_ignored):
        """Return a scalar ForestFlow adapter backed directly by LaCE.

        LaCE controls the trusted interpolation extent, including its default
        ``kmax=200 iMpc``.  ForestFlow does not create a second k grid.
        """
        return LinearTheoryGrid(
            z=np.atleast_1d(np.asarray(zs, dtype=float)),
            cosmology=self._cosmology_for_parameters(new_cosmo_params),
        )

    def get_linear_theory_batch(
        self, zs, cosmology_parameters: Sequence[dict[str, Any]]
    ):
        """Return a batched adapter containing one LaCE cosmology per row."""
        return LinearTheoryGrid(
            z=np.atleast_1d(np.asarray(zs, dtype=float)),
            cosmologies=tuple(
                self._cosmology_for_parameters(parameters)
                for parameters in cosmology_parameters
            ),
        )

    @staticmethod
    def _require_scalar_adapter(linear: LinearTheoryGrid):
        if linear.cosmology is None or linear.is_batched:
            raise ValueError("A scalar LaCE cosmology adapter is required")

    def get_linP_Mpc(self, linear, z, k_iMpc):
        """Evaluate LaCE linear power for scalar or redshift-vector requests."""
        self._require_scalar_adapter(linear)
        z = np.asarray(z, dtype=float)
        k_iMpc = np.asarray(k_iMpc, dtype=float)
        if z.ndim == 0 or (z.ndim == 1 and k_iMpc.ndim == 1):
            return linear.cosmology.get_linP_Mpc(z, k_iMpc)
        if z.ndim == 1 and k_iMpc.ndim >= 2 and k_iMpc.shape[0] == len(z):
            return np.asarray(
                [
                    linear.cosmology.get_linP_Mpc(zi, k_iMpc[iz])
                    for iz, zi in enumerate(z)
                ]
            )
        raise NotImplementedError(
            f"Unsupported shapes: z={z.shape}, k_iMpc={k_iMpc.shape}"
        )

    def get_linP_Mpc_batch(self, linear, k_iMpc):
        """Evaluate LaCE linear power for a ``(batch, z, ...)`` request."""
        if not linear.is_batched or linear.cosmologies is None:
            raise ValueError("A batched LaCE cosmology adapter is required")
        k_iMpc = np.asarray(k_iMpc, dtype=float)
        if k_iMpc.ndim < 3:
            raise ValueError("k_iMpc must have shape (batch, z, ...)")
        if len(linear.cosmologies) != k_iMpc.shape[0]:
            raise ValueError("cosmology batch and k_iMpc batch have different lengths")
        if len(linear.z) != k_iMpc.shape[1]:
            raise ValueError("redshift grid and k_iMpc redshift axis have different lengths")
        return np.asarray(
            [
                [
                    cosmo.get_linP_Mpc(z, k_iMpc[ibatch, iz])
                    for iz, z in enumerate(linear.z)
                ]
                for ibatch, cosmo in enumerate(linear.cosmologies)
            ]
        )

    def get_fz(self, linear, z):
        """Evaluate LaCE's growth rate for a scalar adapter."""
        self._require_scalar_adapter(linear)
        return linear.cosmology.get_growth_rate(z)

    def get_fz_batch(self, linear):
        """Evaluate LaCE growth rates for every batched cosmology and z."""
        if not linear.is_batched or linear.cosmologies is None:
            raise ValueError("A batched LaCE cosmology adapter is required")
        return np.asarray(
            [cosmo.get_growth_rate(linear.z) for cosmo in linear.cosmologies]
        )
