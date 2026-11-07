import numpy as np
import pytest

from forestflow.statistics.p3d import P3D_Mpc_k_mu_hybrid_averaged
from forestflow.statistics.rebin_p3d import (
    get_P3D_k_mu_bin_edges,
    get_P3D_k_mu_modes,
    get_P3D_k_mu_modes_from_edges,
)


def _brute_force_counts(k_edges, mu_edges, lbox):
    k_fun = 2 * np.pi / lbox
    nn = int(k_edges[-1] // k_fun) + 1
    grid = np.mgrid[-nn : nn + 1, -nn : nn + 1, -nn : nn + 1] * k_fun
    kk = np.sqrt(np.sum(grid**2, axis=0))
    mu = np.divide(np.abs(grid[2]), kk, out=np.zeros_like(kk), where=kk > 0)
    counts = np.zeros((len(k_edges) - 1, len(mu_edges) - 1), dtype=int)
    for ii in range(counts.shape[0]):
        lower = kk >= k_edges[ii] if ii == 0 else kk > k_edges[ii]
        for jj in range(counts.shape[1]):
            upper_mu = mu <= mu_edges[jj + 1] if jj == counts.shape[1] - 1 else mu < mu_edges[jj + 1]
            counts[ii, jj] = np.sum(
                lower & (kk <= k_edges[ii + 1]) & (mu >= mu_edges[jj]) & upper_mu & (kk > 0)
            )
    return counts


def test_linear_edges_match_brute_force_lattice():
    # ACCEL2-like geometry: linear bins centred on multiples of k_fun
    lbox = 50.0
    k_fun = 2 * np.pi / lbox
    k_edges = (np.arange(1, 12) - 0.5) * k_fun
    mu_edges = np.linspace(0, 1, 5)

    modes, counts = get_P3D_k_mu_modes_from_edges(
        k_edges, mu_edges, Lbox_Mpc=lbox, max_discrete_modes=40
    )

    np.testing.assert_array_equal(counts, _brute_force_counts(k_edges, mu_edges, lbox))
    for ii in range(counts.shape[0]):
        for jj in range(counts.shape[1]):
            key = f"{ii}_{jj}"
            if 0 < counts[ii, jj] <= 40:
                assert len(modes[key + "_k"]) == counts[ii, jj]
                assert np.all((modes[key + "_k"] > k_edges[ii] - 1e-12)
                              & (modes[key + "_k"] <= k_edges[ii + 1]))
            else:
                assert key + "_k" not in modes


def test_logarithmic_edges_match_mpg_modes():
    lbox = 67.5
    k_edges, mu_edges = get_P3D_k_mu_bin_edges(1.0, Lbox_Mpc=lbox, n_k_bins=20, n_mu_bins=16)
    mpg_modes = get_P3D_k_mu_modes(1.0, Lbox_Mpc=lbox, n_k_bins=20, n_mu_bins=16)

    _, counts = get_P3D_k_mu_modes_from_edges(k_edges, mu_edges, Lbox_Mpc=lbox)

    for ii in range(counts.shape[0]):
        for jj in range(counts.shape[1]):
            modes_k = mpg_modes.get(f"{ii}_{jj}_k")
            assert counts[ii, jj] == (0 if modes_k is None else len(modes_k))


def test_hybrid_with_counts_matches_full_mode_dictionary():
    def model(linear, z, k, mu, params):
        return (1 + 2 * mu**2) / (1 + k**2)

    lbox = 50.0
    k_fun = 2 * np.pi / lbox
    k_edges = (np.arange(1, 12) - 0.5) * k_fun
    mu_edges = np.linspace(0, 1, 5)

    # all populated cells sparse: both paths use only exact lattice modes
    modes, counts = get_P3D_k_mu_modes_from_edges(
        k_edges, mu_edges, Lbox_Mpc=lbox, max_discrete_modes=10**6
    )
    kwargs = dict(k_mu_modes=modes, k_iMpc_edges=k_edges, mu_edges=mu_edges,
                  max_discrete_modes=10**6)
    with_counts = P3D_Mpc_k_mu_hybrid_averaged(None, 3.0, model, mode_counts=counts, **kwargs)
    without_counts = P3D_Mpc_k_mu_hybrid_averaged(None, 3.0, model, **kwargs)
    np.testing.assert_allclose(with_counts, without_counts, equal_nan=True)


def test_hybrid_with_counts_requires_sparse_modes():
    def model(linear, z, k, mu, params):
        return np.ones_like(k)

    lbox = 50.0
    k_fun = 2 * np.pi / lbox
    k_edges = (np.arange(1, 5) - 0.5) * k_fun
    mu_edges = np.linspace(0, 1, 5)
    _, counts = get_P3D_k_mu_modes_from_edges(k_edges, mu_edges, Lbox_Mpc=lbox)

    with pytest.raises(ValueError):
        P3D_Mpc_k_mu_hybrid_averaged(
            None, 3.0, model, k_mu_modes={}, k_iMpc_edges=k_edges,
            mu_edges=mu_edges, mode_counts=counts,
        )
