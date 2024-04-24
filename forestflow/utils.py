import numpy as np
import torch
import functools
from pyDOE2 import lhs
from lace.cosmo.camb_cosmo import dkms_dMpc


def params_numpy2dict(params):
    """
    Converts a numpy array of parameters to a dictionary.

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


def transform_arinyo_params(dict_arinyo_params, fcosmo):
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


def change_units(in_unit, out_unit, z, cosmo=None, camb_results=None):
    lambda_alpha = 1215.67
    clight = 299792.458
    units = ["kms", "AA", "Mpc"]
    # check in_unit and out_unit in units

    if ((in_unit == "Mpc") | (out_unit == "Mpc")) & (cosmo is None):
        raise ValueError("Please introduce cosmology")

    if (in_unit == "AA") & (out_unit == "kms"):
        k_convert = lambda_alpha * (1 + z) / clight
    elif (in_unit == "kms") & (out_unit == "AA"):
        k_convert = 1 / (lambda_alpha * (1 + z) / clight)
    elif (in_unit == "kms") & (out_unit == "Mpc"):
        k_convert = dkms_dMpc(cosmo, z, camb_results=camb_results)
    elif (in_unit == "Mpc") & (out_unit == "kms"):
        k_convert = 1 / dkms_dMpc(cosmo, z, camb_results=camb_results)
    elif (in_unit == "AA") & (out_unit == "Mpc"):
        AA2kms = lambda_alpha * (1 + z) / clight
        kms2Mpc = dkms_dMpc(cosmo, z, camb_results=camb_results)
        k_convert = AA2kms * kms2Mpc
    elif (in_unit == "Mpc") & (out_unit == "AA"):
        AA2kms = lambda_alpha * (1 + z) / clight
        kms2Mpc = dkms_dMpc(cosmo, z, camb_results=camb_results)
        k_convert = 1 / (AA2kms * kms2Mpc)
    return k_convert


def purge_chains(ln_prop_chains, nsplit=5, abs_diff=5, minval=-1000):
    """Purge emcee chains that have not converged"""
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
    keep1 = (np.abs(split_res - split_res_med[:, np.newaxis]) < abs_diff).all(
        axis=1
    )
    # total-dependence convergence
    # check that average logprob is close to minimum logprob of all chains
    # check that all chunks are above a target minimum value
    keep2 = (split_res > minval).all(axis=1)

    # combine both criteria
    keep = keep1 & keep2

    return keep


def init_chains(
    parameters,
    nwalkers,
    bounds,
    criterion="c",
    seed=0,
    attraction=1,
    min_attraction=0.05,
):
    parameter_names = list(parameters.keys())
    parameter_values = np.array(list(parameters.values()))
    nparams = len(parameter_names)
    design = lhs(
        nparams,
        samples=nwalkers,
        criterion=criterion,
        random_state=seed,
    )

    if attraction > 1:
        attraction = 1
    elif attraction < min_attraction:
        attraction = min_attraction

    for ii in range(nparams):
        buse = bounds[parameter_names[ii]]
        lbox = (buse[1] - buse[0]) * attraction

        # design sample using lh as input, attracted to best-fitting solution
        design[:, ii] = (
            lbox * (design[:, ii] - 0.5)
            + buse[0] * attraction
            + parameter_values[ii]
        )

        # make sure that samples do not get out of prior range
        _ = design[:, ii] >= buse[1]
        design[_, ii] -= lbox * 0.999
        _ = design[:, ii] <= buse[0]
        design[_, ii] += lbox * 0.999

    return design


def memorize(func):
    # Initialize a dictionary to store the previous arguments and result
    cache = {}

    def wrapper(*args, **kwargs):
        # Convert args and kwargs to a hashable key
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


def memoize_numpy_arrays(func, max_history=2):
    # Initialize a dictionary to store the previous results for each key
    cache = {}

    def wrapper(*args, **kwargs):
        # Convert NumPy arrays to a tuple of their shapes and contents
        key = tuple(
            (a.shape, tuple(a.flat)) if isinstance(a, np.ndarray) else a
            for a in args
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


def memoize_pytorch(func):
    # Initialize a dictionary to store the previous input tensors and result
    cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Convert PyTorch tensors to tuples of their shapes and contents
        args_key = tuple(
            (a.shape, tuple(a.flatten().tolist()))
            if isinstance(a, torch.Tensor)
            else a
            for a in args
        )
        kwargs_key = tuple(
            (key, value.shape, tuple(value.flatten().tolist()))
            if isinstance(value, torch.Tensor)
            else (key, value)
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


def memorize(func):
    # Initialize a dictionary to store the previous input parameters and result
    cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Convert arguments and keyword arguments to a tuple of their values
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


def sort_dict(dct, keys):
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
        d.update(
            sorted_d
        )  # update the original dictionary with the sorted dictionary
    return dct


def get_covariance(x, y, return_corr=False):
    
    # Calculate the mean and standard deviation along each column
    mean_x = np.mean(x, axis=0)
    std_dev_x = np.std(x, axis=0)

    # Create a mask indicating elements within one standard deviation from the mean
    mask_within_sigma = np.abs(x - mean_x) <= 3*std_dev_x

    # Apply the mask along each column to preserve the shape (N_after_cleaning, 42)
    x = x[mask_within_sigma.all(axis=1)]


    cov = (
        1 / (len(x) - 1)
        * np.einsum("ij,jk ->ik", (x - y[None, :]).T, (x - y[None, :]))
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


def sigma68(data):
    return 0.5 * (
        np.nanquantile(data, q=0.84, axis=0)
        - np.nanquantile(data, q=0.16, axis=0)
    )


def load_Arinyo_chains(
    archive,
    folder_chains="/pscratch/sd/l/lcabayol/P3D/p3d_fits_new/",
    sim_label=None,
    z=None,
    chain_samp=10_000,
):
    """
    Load Arinyo model chains from stored files for all the training LH simulations.

    This function loads Arinyo model chains corresponding to different simulations from saved files.
    It extracts relevant information such as simulation label, scaling factor, redshift, and other parameters
    to construct the file tag for each simulation. The loaded chains are then processed and returned.

    Returns:
        np.array: Array containing Arinyo model chains for all simulations.
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
                "fit_sim"
                + sim_label[4:]
                + "_tau"
                + str(np.round(scale_tau, 2))
                + "_z"
                + str(ind_z)
                + "_kmax3d"
                + str(archive.kmax_3d)
                + "_noise3d"
                + str(archive.noise_3d)
                + "_kmax1d"
                + str(archive.kmax_1d)
                + "_noise1d"
                + str(archive.noise_1d)
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
            raise ValueError(
                "If sim_label is not None, a redshift must be provided."
            )

        scale_tau = 1.0
        ind_z = z

        # Construct file tag based on simulation parameters
        tag = (
            "fit_sim"
            + sim_label[4:]
            + "_tau"
            + str(np.round(scale_tau, 2))
            + "_z"
            + str(ind_z)
            + "_kmax3d"
            + str(archive.kmax_3d)
            + "_noise3d"
            + str(archive.noise_3d)
            + "_kmax1d"
            + str(archive.kmax_1d)
            + "_noise1d"
            + str(archive.noise_1d)
        )

        # Load Arinyo model chain from file
        file_arinyo = np.load(folder_chains + tag + ".npz")
        chain = file_arinyo["chain"].copy()

        # Ensure non-positive values for the first parameter
        chain[:, 0] = -np.abs(chain[:, 0])

        # Randomly sample from the loaded chain
        idx = np.random.randint(len(chain), size=(chain_samp))
        chain_sampled = chain[idx]

        print("Chains loaded")
        return chain_sampled
