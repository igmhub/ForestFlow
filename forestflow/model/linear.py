"""
LaCE-backed linear-theory adapters used by ForestFlow models.
"""

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from lace.cosmo import cosmology, rescale_cosmology


@dataclass(slots=True)
class LinearTheoryGrid:
    """
    One or more LaCE cosmology objects associated with redshift samples.

    The historical name is retained for API continuity. This object no longer
    stores or interpolates a duplicate ForestFlow linear-power grid: LaCE is
    the sole implementation of both linear power and growth rate.
    """

    z: np.ndarray
    cosmology: Any | None = None
    cosmologies: tuple[Any, ...] | None = None

    @property
    def is_batched(self) -> bool:
        """
        Return whether this adapter carries one LaCE cosmology per batch row.

        Returns
        -------
        bool
            True when ``cosmologies`` rather than scalar ``cosmology`` is set.
        """
        return self.cosmologies is not None


class LinearTheory:
    """
    Create scalar or batched ForestFlow views of LaCE cosmologies.

    This class owns no Boltzmann calculation or interpolation scheme.  Its
    only role is to package LaCE cosmology objects for ForestFlow's scalar and
    batched model APIs.
    """

    def __init__(self, fiducial_cosmology=None):
        """
        Initialize the adapter with a fiducial LaCE cosmology.

        Parameters
        ----------
        fiducial_cosmology : lace.cosmo.cosmology.Cosmology, optional
            Reference cosmology used for compatible primordial rescalings.
        """
        self.fiducial_cosmology = fiducial_cosmology or cosmology.Cosmology()

    def _cosmology_for_parameters(self, new_cosmo_params=None):
        """
        Return a rescaled or newly calculated cosmology for parameter changes.

        Parameters
        ----------
        new_cosmo_params : mapping, optional
            Changes relative to the fiducial cosmology. ``nnu`` always forces
            a fresh calculation because it changes transfer functions.

        Returns
        -------
        lace.cosmo.base_cosmology.BaseCosmology
            Compatible rescaling or full cosmology evaluation.
        """
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
        """
        Return a scalar ForestFlow adapter backed directly by LaCE.

        Parameters
        ----------
        zs : array_like
            Redshift grid.
        new_cosmo_params : mapping, optional
            Cosmology changes relative to the fiducial model.
        **_ignored
            Compatibility keywords intentionally ignored.

        Returns
        -------
        LinearTheoryGrid
            Scalar LaCE cosmology adapter.

        Notes
        -----
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
        """
        Return a batched adapter containing one LaCE cosmology per row.

        Parameters
        ----------
        zs : array_like
            Shared redshift grid.
        cosmology_parameters : sequence of mapping
            One physical cosmology parameter mapping per batch row.

        Returns
        -------
        LinearTheoryGrid
            Adapter with one LaCE cosmology per batch element.
        """
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
        """
        Evaluate LaCE linear power for scalar or redshift-vector requests.

        Parameters
        ----------
        linear : LinearTheoryGrid
            Scalar adapter returned by :meth:`get_linear_theory`.
        z : float or ndarray
            Redshift scalar or one-dimensional redshift grid.
        k_iMpc : array_like
            Comoving wavenumbers in ``1 / Mpc``; for a redshift grid the first
            axis must match its length.

        Returns
        -------
        ndarray
            Linear three-dimensional power in ``Mpc**3``.

        Raises
        ------
        ValueError
            If a batched adapter is supplied.
        NotImplementedError
            If redshift and k-array shapes are unsupported.
        """
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
        """
        Evaluate LaCE linear power for a ``(batch, z, ...)`` request.

        Parameters
        ----------
        linear : LinearTheoryGrid
            Batched adapter from :meth:`get_linear_theory_batch`.
        k_iMpc : array_like
            Wavenumbers with leading shape ``(n_batch, n_z, ...)`` in
            ``1 / Mpc``.

        Returns
        -------
        ndarray
            Linear three-dimensional power in ``Mpc**3`` with matching shape.

        Raises
        ------
        ValueError
            If adapter type, batch length, redshift length, or k dimensions do
            not match.
        """
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
        """
        Evaluate LaCE's growth rate for a scalar adapter.

        Parameters
        ----------
        linear : LinearTheoryGrid
            Scalar adapter.
        z : float or array_like
            Redshift coordinate.

        Returns
        -------
        float or ndarray
            Dimensionless logarithmic growth rate.
        """
        self._require_scalar_adapter(linear)
        return linear.cosmology.get_growth_rate(z)

    def get_fz_batch(self, linear):
        """
        Evaluate LaCE growth rates for every batched cosmology and redshift.

        Parameters
        ----------
        linear : LinearTheoryGrid
            Batched adapter.

        Returns
        -------
        ndarray, shape (n_batch, n_z)
            Dimensionless logarithmic growth rates.

        Raises
        ------
        ValueError
            If the adapter is not batched.
        """
        if not linear.is_batched or linear.cosmologies is None:
            raise ValueError("A batched LaCE cosmology adapter is required")
        return np.asarray(
            [cosmo.get_growth_rate(linear.z) for cosmo in linear.cosmologies]
        )
