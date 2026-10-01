"""Cross-package cosmology-contract regressions for ForestFlow."""

import numpy as np

from lace.cosmo import cosmology, rescale_cosmology

from forestflow.model.linear import LinearTheory


def test_nnu_variation_rebuilds_transfer_functions_instead_of_rescaling():
    """ForestFlow must delegate an ``nnu`` change to a fresh LaCE cosmology."""

    fiducial = cosmology.Cosmology()
    requested = {"nnu": 4.0}
    assert not fiducial.same_background(requested)

    linear_theory = LinearTheory(fiducial)
    varied = linear_theory.get_linear_theory([3.0], requested).cosmology
    fresh = cosmology.Cosmology(cosmo_params_dict=requested)

    assert not isinstance(varied, rescale_cosmology.RescaledCosmology)
    k_iMpc = np.array([0.1, 1.0, 10.0])
    np.testing.assert_allclose(
        varied.get_linP_Mpc(3.0, k_iMpc),
        fresh.get_linP_Mpc(3.0, k_iMpc),
        rtol=1.0e-12,
    )
