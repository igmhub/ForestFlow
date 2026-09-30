"""Shared utility helpers."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from numpy.typing import ArrayLike, NDArray
import numpy as np

def purge_chains(ln_prop_chains: ArrayLike, nsplit: int | None=5, abs_diff: int | None=5, minval: Any=-1000) -> Any:
    """
    Purge emcee chains that have not converged

    Parameters
    ----------
    ln_prop_chains : numpy.ndarray
        Ln prop chains used by the calculation.
    nsplit : int, optional
        Nsplit used by the calculation.
    abs_diff : int, optional
        Abs diff used by the calculation.
    minval : object
        Minval used by the calculation.

    Returns
    -------
    object
        Result produced when the function is used to purge emcee chains that have not converged.
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
    Initialize chains.

    Parameters
    ----------
    parameters : object
        Parameters used by the calculation.
    nwalkers : int or float
        Number of ensemble walkers.
    bounds : dict
        Lower and upper bounds for each parameter.
    seed : int, optional
        Seed for the random-number generator.
    attraction : int, optional
        Attraction used by the calculation.
    min_attraction : float, optional
        Min attraction used by the calculation.

    Returns
    -------
    object
        Result produced when the function is used to initialize chains.
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
    Load Arinyo model chains from stored files for all the training LH simulations.

    This function loads Arinyo model chains corresponding to different simulations from saved files.
    It extracts relevant information such as simulation label, scaling factor, redshift, and other parameters
    to construct the file tag for each simulation. The loaded chains are then processed and returned.

    Returns:
        np.array: Array containing Arinyo model chains for all simulations.

    Parameters
    ----------
    archive : object
        Simulation archive containing the requested data.
    folder_chains : str, optional
        Folder chains used by the calculation.
    sim_label : object, optional
        Sim label used by the calculation.
    z : object, optional
        Redshift.
    chain_samp : int, optional
        Chain samp used by the calculation.
    kmax_3d : int, optional
        Maximum three-dimensional wavenumber included in the fit.
    kmax_1d : int, optional
        Maximum one-dimensional wavenumber included in the fit.
    noise_3d : float, optional
        Relative three-dimensional noise level.
    noise_1d : float, optional
        Relative one-dimensional noise level.
    training_type : str, optional
        Training type used by the calculation.
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
