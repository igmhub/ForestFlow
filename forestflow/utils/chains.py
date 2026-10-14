"""
Shared utility helpers.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from numpy.typing import ArrayLike, NDArray
import numpy as np

def purge_chains(ln_prop_chains: ArrayLike, nsplit: int | None=5, abs_diff: int | None=5, minval: Any=-1000) -> Any:
    """
    Flag walkers with stable and sufficiently high log-probability chunks.

    Parameters
    ----------
    ln_prop_chains : ndarray
        Log posterior values with sampling steps on axis zero and walkers on
        the remaining axis.
    nsplit : int, default: 5
        Number of temporal chunks for stability checks.
    abs_diff : float, default: 5
        Maximum permitted chunk-mean deviation from each walker mean.
    minval : float, default: -1000
        Strict lower log-probability threshold for all chunks.

    Returns
    -------
    ndarray of bool
        Per-walker convergence flags.
    """
    # split each walker in nsplit chunks
    split_arr = np.array_split(ln_prop_chains, nsplit, axis=0)
    # compute median of each chunck
    split_med = []
    for ii in range(nsplit):
        split_med.append(split_arr[ii].mean(axis=0))
    # (nwalkers, nchucks)
    split_res = np.array(split_med).T
    # compute median of chunks for each walker ()
    split_res_med = split_res.mean(axis=1)

    # step-dependence convergence
    # check that average logprob does not vary much with step
    # compute difference between chunks and median of each chain
    keep1 = (np.abs(split_res - split_res_med[:, np.newaxis]) < abs_diff).all(axis=1)
    # total-dependence convergence
    # check that average logprob is close to minimum logprob of all chains
    # check that all chunks are above a target minimum value
    keep2 = (split_res > minval).all(axis=1)

    # combine both criteria
    keep = keep1 & keep2

    return keep

def init_chains(
    parameters: Any,
    nwalkers: int | float,
    bounds: Mapping[str, Any],
    seed: int | None=0,
    attraction: int | None=1,
    min_attraction: float | None=0.05,
) -> Any:
    """
    Initialize bounded ensemble-walker positions with Latin hypercubes.

    Parameters
    ----------
    parameters : mapping
        Central parameter values in output-column order.
    nwalkers : int
        Number of ensemble positions.
    bounds : mapping
        Inclusive lower/upper bounds keyed by parameter name.
    seed : int, default: 0
        Latin-hypercube random seed.
    attraction : float, default: 1
        Fraction of each prior interval sampled around the supplied center.
    min_attraction : float, default: 0.05
        Lower clamp applied to ``attraction``.

    Returns
    -------
    ndarray, shape (nwalkers, n_parameters)
        Initial positions clipped back inside the configured bounds.
    """
    from scipy.stats import qmc

    parameter_names = list(parameters.keys())
    parameter_values = np.array(list(parameters.values()))
    nparams = len(parameter_names)

    lhs_sampler = qmc.LatinHypercube(d=nparams, seed=seed)
    design = lhs_sampler.random(n=nwalkers)

    if attraction > 1:
        attraction = 1
    elif attraction < min_attraction:
        attraction = min_attraction

    for ii in range(nparams):
        buse = bounds[parameter_names[ii]]
        lbox = (buse[1] - buse[0]) * attraction

        # design sample using lh as input, attracted to best-fitting solution
        design[:, ii] = (
            lbox * (design[:, ii] - 0.5) + buse[0] * attraction + parameter_values[ii]
        )

        # make sure that samples do not get out of prior range
        _ = design[:, ii] >= buse[1]
        design[_, ii] -= lbox * 0.999
        _ = design[:, ii] <= buse[0]
        design[_, ii] += lbox * 0.999

    return design

def load_Arinyo_chains(
    archive: Any,
    folder_chains: str | None="/pscratch/sd/l/lcabayol/P3D/p3d_fits_new/",
    sim_label: Any | None=None,
    z: Any | None=None,
    chain_samp: int | None=10_000,
    kmax_3d: int | None=3,
    kmax_1d: int | None=3,
    noise_3d: float | None=0.01,
    noise_1d: float | None=0.01,
    training_type: str | None="Arinyo_min_q1_q2",
) -> NDArray[Any]:
    """
    Load and randomly resample legacy on-disk Arinyo posterior chains.

    Parameters
    ----------
    archive : object
        Archive associated with the requested fit. In the all-training-data
        branch, the historical implementation instead reads ``Archive3D``
        from module/global scope.
    folder_chains : path-like, default="/pscratch/sd/l/lcabayol/P3D/p3d_fits_new/"
        Directory containing ``.npz`` files with a ``chain`` array.
    sim_label : str, optional
        Specific simulation label. If omitted, load one resampled chain for
        every training snapshot.
    z : float, optional
        Redshift required when ``sim_label`` is supplied.
    chain_samp : int, default=10000
        Number of draws sampled with replacement from each stored chain.
    kmax_3d, kmax_1d : float, default=3
        P3D/P1D fit cuts encoded into the legacy filename tag in ``Mpc^-1``.
    noise_3d : float, optional
        Relative three-dimensional noise level.
    noise_1d : float, optional
        Relative one-dimensional noise level.
    training_type : {"Arinyo_min_q1_q2"}, optional
        When selected, transform stored q-plus/q-minus coordinates to q1/q2.

    Returns
    -------
    ndarray
        Shape ``(chain_samp, n_parameters)`` for a specific simulation, or
        ``(n_training_snapshots, chain_samp, 8)`` when ``sim_label`` is omitted.

    Raises
    ------
    ValueError
        If a specific ``sim_label`` is provided without ``z``.
    FileNotFoundError
        If the filename constructed from the requested legacy fit settings is
        absent.
    """
    print("Loading Arinyo chains")

    if sim_label == None:
        training_data = Archive3D.training_data

        # Initialize array to store Arinyo model chains
        chains = np.zeros(shape=(len(training_data), chain_samp, 8))

        # Loop over simulations in the training data
        for ind_book in range(0, len(training_data)):
            sim_label = training_data[ind_book]["sim_label"]
            scale_tau = training_data[ind_book]["val_scaling"]
            ind_z = training_data[ind_book]["z"]

            # Construct file tag based on simulation parameters
            tag = (
                "fit_sim_label_"
                + sim_label
                + "_tau"
                + str(np.round(scale_tau, 2))
                + "_z"
                + str(ind_z)
                + "_kmax3d"
                + str(kmax_3d)
                + "_noise3d"
                + str(noise_3d)
                + "_kmax1d"
                + str(kmax_1d)
                + "_noise1d"
                + str(noise_1d)
            )

            # Load Arinyo model chain from file
            file_arinyo = np.load(folder_chains + tag + ".npz")
            chain = file_arinyo["chain"].copy()

            # Ensure non-positive values for the first parameter
            chain[:, 0] = -np.abs(chain[:, 0])

            # Randomly sample from the loaded chain
            idx = np.random.randint(len(chain), size=(chain_samp))
            chain_sampled = chain[idx]
            chains[ind_book] = chain_sampled

        print("Chains loaded")
        return chains

    else:
        if z is None:
            raise ValueError("If sim_label is not None, a redshift must be provided.")

        scale_tau = 1.0
        ind_z = z

        # Construct file tag based on simulation parameters
        # tag = (
        #     "fit_sim"
        #     + sim_label[4:]
        #     + "_tau"
        #     + str(np.round(scale_tau, 2))
        #     + "_z"
        #     + str(ind_z)
        #     + "_kmax3d"
        #     + str(archive.kmax_3d)
        #     + "_noise3d"
        #     + str(archive.noise_3d)
        #     + "_kmax1d"
        #     + str(archive.kmax_1d)
        #     + "_noise1d"
        #     + str(archive.noise_1d)
        # )
        tag = (
            "fit_sim_label_"
            + sim_label
            + "_tau_"
            + str(np.round(scale_tau, 2))
            + "_z_"
            + str(ind_z)
            + "_kmax3d_"
            + str(kmax_3d)
            + "_noise3d_"
            + str(noise_3d)
            + "_kmax1d_"
            + str(kmax_1d)
            + "_noise1d_"
            + str(noise_1d)
        )

        # Load Arinyo model chain from file
        file_arinyo = np.load(folder_chains + tag + ".npz")
        chain = file_arinyo["chain"]

        # Ensure non-positive values for the first parameter
        chain[:, 0] = -np.abs(chain[:, 0])

        if training_type == "Arinyo_min_q1_q2":
            q1 = 0.5 * (chain[:, 2] + chain[:, -1])
            q2 = 0.5 * (chain[:, 2] - chain[:, -1])
            chain[:, 2] = q1
            chain[:, -1] = q2

        # Randomly sample from the loaded chain
        idx = np.random.randint(len(chain), size=(chain_samp))
        chain_sampled = chain[idx]

        print("Chains loaded")
        return chain_sampled
