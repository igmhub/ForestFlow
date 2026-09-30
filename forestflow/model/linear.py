"""Cosmology-backed linear-theory grids for ForestFlow models."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from lace.cosmo import cosmology, rescale_cosmology


@dataclass(slots=True)
class LinearTheoryGrid:
    """Linear power and growth rate sampled on redshift and k_iMpc grids."""

    z: np.ndarray
    logk_iMpc: np.ndarray
    loglinP_Mpc: np.ndarray
    fz: np.ndarray


class LinearTheory:
    """Build and interpolate linear-theory grids for a fiducial cosmology."""

    def __init__(self, fiducial_cosmology=None):
        self.fiducial_cosmology = fiducial_cosmology or cosmology.Cosmology()

    def get_linear_theory(self, zs, k_min_iMpc=1e-3, k_max_iMpc=100.0, new_cosmo_params=None):
        """Return a linear-theory grid, rescaling the fiducial cosmology when possible."""
        if self.fiducial_cosmology.same_background(cosmo_params=new_cosmo_params):
            cosmo = rescale_cosmology.RescaledCosmology(self.fiducial_cosmology, new_cosmo_params)
        else:
            print("WARNING: computing CAMB again")
            cosmo = cosmology.Cosmology(cosmo_params_dict=new_cosmo_params)
        zs = np.atleast_1d(zs)
        logk_iMpc = np.linspace(np.log(k_min_iMpc), np.log(k_max_iMpc), 200)
        return LinearTheoryGrid(
            z=zs,
            logk_iMpc=logk_iMpc,
            loglinP_Mpc=np.log(cosmo.get_linP_Mpc(zs, np.exp(logk_iMpc))),
            fz=cosmo.compute_growth_rate(zs),
        )

    def get_linear_theory_batch(self, zs, cosmology_parameters):
        """Return one stacked linear-theory grid per cosmology mapping."""
        grids = [self.get_linear_theory(zs, new_cosmo_params=params) for params in cosmology_parameters]
        return LinearTheoryGrid(
            z=np.asarray(zs), logk_iMpc=grids[0].logk_iMpc,
            loglinP_Mpc=np.stack([grid.loglinP_Mpc for grid in grids]),
            fz=np.stack([grid.fz for grid in grids]),
        )

    def get_linP_Mpc(self, linear, z, k_iMpc):
        """Interpolate linear power in Mpc cubed at requested redshifts and k_iMpc."""
        z = np.asarray(z, dtype=float)
        k_iMpc = np.asarray(k_iMpc, dtype=float)
        # k_iMpc=0 is allowed by the model; interpolation naturally uses the
        # lowest tabulated linear-power value in that limiting case.
        with np.errstate(divide="ignore"):
            logk_iMpc = np.log(k_iMpc)
        def index(zi):
            matches=np.where(np.isclose(linear.z, zi, atol=1e-3, rtol=0))[0]
            if len(matches)==0:
                raise ValueError(f"Requested z={zi} is not available in the linear theory grid.")
            return matches[0]
        if z.ndim == 0:
            return np.exp(np.interp(logk_iMpc, linear.logk_iMpc, linear.loglinP_Mpc[index(z)]))
        if z.ndim == 1 and k_iMpc.ndim == 1:
            return np.asarray([np.exp(np.interp(logk_iMpc, linear.logk_iMpc, linear.loglinP_Mpc[index(zi)])) for zi in z])
        if z.ndim == 1 and k_iMpc.shape[0] == len(z):
            out=np.empty_like(k_iMpc)
            for iz, zi in enumerate(z):
                out[iz]=np.exp(np.interp(logk_iMpc[iz].ravel(), linear.logk_iMpc, linear.loglinP_Mpc[index(zi)]).reshape(k_iMpc.shape[1:]))
            return out
        raise NotImplementedError(f"Unsupported shapes: z={z.shape}, k_iMpc={k_iMpc.shape}")

    def get_linP_Mpc_batch(self, linear, k_iMpc):
        """Interpolate a ``(batch, z, k, ...)`` request from a stacked grid."""
        k_iMpc=np.asarray(k_iMpc)
        out=np.empty_like(k_iMpc)
        for ib in range(k_iMpc.shape[0]):
            for iz in range(k_iMpc.shape[1]):
                out[ib, iz]=np.exp(np.interp(np.log(k_iMpc[ib, iz]).ravel(), linear.logk_iMpc, linear.loglinP_Mpc[ib, iz]).reshape(k_iMpc.shape[2:]))
        return out

    def get_fz(self, linear, z):
        """Interpolate the linear growth rate at ``z``."""
        return np.interp(np.asarray(z, dtype=float), linear.z, linear.fz)
