"""
Provide memory, memoization, covariance, and parameter utilities.
"""

from collections.abc import Callable, Mapping, Sequence
from os import PathLike
from typing import Any
from numpy.typing import ArrayLike, NDArray

import numpy as np
import torch
import functools


def print_memory_usage(step_description: str | PathLike[str]) -> None:
    """
    Print the current process memory usage.

    Parameters
    ----------
    step_description : str or pathlib.Path
        Label printed with the memory measurements.
    """
    import psutil

    process = psutil.Process()
    memory_info = process.memory_info()
    print(
        f"{step_description} - RSS: {memory_info.rss / (1024 ** 2):.2f} MB, VMS: {memory_info.vms / (1024 ** 2):.2f} MB"
    )
    if torch.cuda.is_available():
        print(
            f"GPU memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB"
        )
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")


def params_numpy2dict(params: ArrayLike) -> dict[str, Any]:
    """
    Convert a NumPy array of parameters to a dictionary.

    Args:
        params (numpy.ndarray): Array of parameters.

    Returns:
        dict: Dictionary containing the parameters with their corresponding names.
    """
    param_names = [
        "bias",
        "beta",
        "q1",
        "kvav",
        "av",
        "bv",
        "kp",
        "q2",
    ]
    dict_param = {}
    for ii in range(params.shape[0]):
        dict_param[param_names[ii]] = params[ii]
    return dict_param


def params_numpy2dict_minimizer(params: ArrayLike) -> dict[str, Any]:
    """
    Convert a NumPy array of parameters to a dictionary.

    Args:
        params (numpy.ndarray): Array of parameters.

    Returns:
        dict: Dictionary containing the parameters with their corresponding names.
    """
    param_names = [
        "bias",
        "beta",
        "q1",
        "kvav",
        "av",
        "bv",
        "kp",
        "q2",
    ]
    dict_param = {}
    for ii in range(params.shape[0]):
        dict_param[param_names[ii]] = params[ii]

    if "q2" in dict_param.keys():
        q1 = 0.5 * (dict_param["q1"] + dict_param["q2"])
        q2 = 0.5 * (dict_param["q1"] - dict_param["q2"])
        dict_param["q1"] = q1
        dict_param["q2"] = q2
    return dict_param


def params_numpy2dict_minimizerz(params: ArrayLike) -> dict[str, Any]:
    """
    Convert a NumPy array of parameters to a dictionary.

    Args:
        params (numpy.ndarray): Array of parameters.

    Returns:
        dict: Dictionary containing the parameters with their corresponding names.
    """
    dict_param = {}
    for key in params:
        if key == "q1":
            dict_param["q1"] = 0.5 * params[key]
        else:
            dict_param[key] = params[key]
    dict_param["q2"] = dict_param["q1"]

    return dict_param


def transform_arinyo_params(dict_arinyo_params: Mapping[str, Any], fcosmo: Any) -> Any:
    """
    Transform Arinyo parameters.

    Parameters
    ----------
    dict_arinyo_params : dict
        Arinyo parameter mapping.
    fcosmo : object
        Cosmological growth rate.

    Returns
    -------
    object
        Result produced when the function is used to transform arinyo parameters.
    """
    dict_arinyo_params_out = {}
    for key in dict_arinyo_params.keys():
        if key == "beta":
            dict_arinyo_params_out["bias_eta"] = (
                dict_arinyo_params["bias"] * dict_arinyo_params["beta"] / fcosmo
            )
        elif key == "kvav":
            dict_arinyo_params_out["kv"] = dict_arinyo_params["kvav"] ** (
                1 / dict_arinyo_params["av"]
            )
        else:
            dict_arinyo_params_out[key] = dict_arinyo_params[key]
    return dict_arinyo_params_out


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


def memorize(func: Callable[..., Any]) -> Callable[..., Any]:
    # Initialize a dictionary to store the previous arguments and result
    """
    Memoize the requested values.

    Parameters
    ----------
    func : callable
        Function to wrap.

    Returns
    -------
    object
        Result produced when the function is used to memoize the requested values.
    """
    cache = {}

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Convert args and kwargs to a hashable key
        """
        Call the wrapped function, reusing a cached result when available.

        Parameters
        ----------
        args : object
            Args used by the calculation.
        kwargs : object
            Kwargs used by the calculation.

        Returns
        -------
        object
            Result produced when the function is used to call the wrapped function, reusing a cached result when available.
        """
        key = (args, frozenset(kwargs.items()))

        # Check if the same input parameters have been seen before
        if key in cache:
            # If yes, return the cached result
            return cache[key]
        else:
            # If not, call the inner function and cache the result
            result = func(*args, **kwargs)
            cache[key] = result
            return result

    return wrapper


# def memoize_numpy_arrays(func):
#     # Initialize a dictionary to store the previous NumPy arrays and result
#     cache = {}

#     def wrapper(*args, **kwargs):
#         # Convert NumPy arrays to a tuple of their shapes and contents
#         key = tuple(
#             (a.shape, tuple(a.flat)) if isinstance(a, np.ndarray) else a
#             for a in args
#         )

#         # Check if the same input NumPy arrays have been seen before
#         if key in cache:
#             # If yes, return the cached result
#             return cache[key]
#         else:
#             # If not, call the inner function and cache the result
#             result = func(*args, **kwargs)
#             cache[key] = result
#             return result

#     return wrapper


def memoize_numpy_arrays(func: Callable[..., Any], max_history: int | None=2) -> Callable[..., Any]:
    # Initialize a dictionary to store the previous results for each key
    """
    Memoize numpy arrays.

    Parameters
    ----------
    func : callable
        Function to wrap.
    max_history : int, optional
        Maximum number of results retained in the cache.

    Returns
    -------
    object
        Result produced when the function is used to memoize numpy arrays.
    """
    cache = {}

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Convert NumPy arrays to a tuple of their shapes and contents
        """
        Call the wrapped function, reusing a cached result when available.

        Parameters
        ----------
        args : object
            Args used by the calculation.
        kwargs : object
            Kwargs used by the calculation.

        Returns
        -------
        object
            Result produced when the function is used to call the wrapped function, reusing a cached result when available.
        """
        key = tuple(
            (a.shape, tuple(a.flat)) if isinstance(a, np.ndarray) else a for a in args
        )

        # Check if the key is in the cache
        if key in cache:
            # If yes, return the cached result
            return cache[key]
        else:
            # If not, call the inner function and cache the result
            result = func(*args, **kwargs)
            cache[key] = result
            # Trim the history to the specified maximum
            list_keys = list(cache.keys())
            if len(list_keys) > max_history:
                del cache[list_keys[0]]
            return result

    return wrapper


def memoize_pytorch(func: Callable[..., Any]) -> Callable[..., Any]:
    # Initialize a dictionary to store the previous input tensors and result
    """
    Memoize pytorch.

    Parameters
    ----------
    func : callable
        Function to wrap.

    Returns
    -------
    object
        Result produced when the function is used to memoize pytorch.
    """
    cache = {}

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Convert PyTorch tensors to tuples of their shapes and contents
        """
        Call the wrapped function, reusing a cached result when available.

        Parameters
        ----------
        args : object
            Args used by the calculation.
        kwargs : object
            Kwargs used by the calculation.

        Returns
        -------
        object
            Result produced when the function is used to call the wrapped function, reusing a cached result when available.
        """
        args_key = tuple(
            (a.shape, tuple(a.flatten().tolist())) if isinstance(a, torch.Tensor) else a
            for a in args
        )
        kwargs_key = tuple(
            (
                (key, value.shape, tuple(value.flatten().tolist()))
                if isinstance(value, torch.Tensor)
                else (key, value)
            )
            for key, value in kwargs.items()
        )

        # Combine args and kwargs keys into a single key
        key = (args_key, kwargs_key)

        # Check if the same input parameters have been seen before
        if key in cache:
            # If yes, return the cached result
            return cache[key]
        else:
            # If not, call the inner function and cache the result
            result = func(*args, **kwargs)
            cache[key] = result
            return result

    return wrapper


def memorize(func: Callable[..., Any]) -> Callable[..., Any]:
    # Initialize a dictionary to store the previous input parameters and result
    """
    Memoize the requested values.

    Parameters
    ----------
    func : callable
        Function to wrap.

    Returns
    -------
    object
        Result produced when the function is used to memoize the requested values.
    """
    cache = {}

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Convert arguments and keyword arguments to a tuple of their values
        """
        Call the wrapped function, reusing a cached result when available.

        Parameters
        ----------
        args : object
            Args used by the calculation.
        kwargs : object
            Kwargs used by the calculation.

        Returns
        -------
        object
            Result produced when the function is used to call the wrapped function, reusing a cached result when available.
        """
        key = (args, tuple(kwargs.items()))

        # Check if the same input parameters have been seen before
        if key in cache:
            # If yes, return the cached result
            return cache[key]
        else:
            # If not, call the inner function and cache the result
            result = func(*args, **kwargs)
            cache[key] = result
            return result

    return wrapper


def sort_dict(dct: Sequence[Any], keys: Sequence[Any]) -> list[Any]:
    """
    Sort a list of dictionaries based on specified keys.

    Args:
        dct (list): List of dictionaries to be sorted.
        keys (list): List of keys to sort the dictionaries by.

    Returns:
        list: The sorted list of dictionaries.
    """
    for d in dct:
        sorted_d = {
            k: d[k] for k in keys
        }  # create a new dictionary with only the specified keys
        d.clear()  # remove all items from the original dictionary
        d.update(sorted_d)  # update the original dictionary with the sorted dictionary
    return dct


def get_covariance(x: Any, y: Any, return_corr: bool | None=False) -> NDArray[Any]:
    # Calculate the mean and standard deviation along each column
    """
    Return covariance.

    Parameters
    ----------
    x : object
        X used by the calculation.
    y : object
        Y used by the calculation.
    return_corr : bool, optional
        Whether to return the correlation matrix with the covariance.

    Returns
    -------
    object
        Result produced when the function is used to return covariance.
    """
    mean_x = np.mean(x, axis=0)
    std_dev_x = np.std(x, axis=0)
    # Create a mask indicating elements within one standard deviation from the mean
    mask_within_sigma = np.abs(x - mean_x) <= 3 * std_dev_x
    # Apply the mask along each column to preserve the shape
    x = x[mask_within_sigma.all(axis=1)]

    cov = (
        1 / (len(x) - 1) * np.einsum("ij,jk ->ik", (x - y[None, :]).T, (x - y[None, :]))
    )
    corr = np.corrcoef(cov)
    if return_corr:
        return cov, corr
    else:
        return cov


# def params_numpy2dict(
#     array,
#     key_strings=["bias", "beta", "q1", "kvav", "av", "bv", "kp", "q2"],
# ):
#     """
#     Convert a numpy array of parameters to a dictionary.

#     Args:
#         array (numpy.ndarray): Array of parameters.
#         key_strings (list): List of strings for dictionary keys. Default is ["bias", "beta", "q1", "kvav", "av", "bv", "kp", "q2"].

#     Returns:
#         dict: Dictionary with key-value pairs corresponding to parameters.
#     """
#     # Create a dictionary with key strings and array elements
#     array_dict = {}
#     for key, value in zip(key_strings, array):
#         array_dict[key] = value

#     return array_dict


def sigma68(data: ArrayLike) -> NDArray[Any]:
    """
    Compute sigma68.

    Parameters
    ----------
    data : numpy.ndarray
        Input data.

    Returns
    -------
    object
        Result produced when the function is used to compute sigma68.
    """
    return 0.5 * (
        np.nanquantile(data, q=0.84, axis=0) - np.nanquantile(data, q=0.16, axis=0)
    )


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
