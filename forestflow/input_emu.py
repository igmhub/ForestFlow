import numpy as np
import os
import copy

from lace.cosmo import camb_cosmo
from lace.setup_simulations import read_genic
from lace.emulator import pnd_archive

from forestflow import model_p3d_arinyo
from forestflow.fit_p3d import FitPk


def read_extra_sims():
    """
    Reads additional simulations and appends them to an archive.

    Returns:
        pnd_archive: The updated archive containing the additional simulations.
    """
    option_list = [
        "growth",
        "neutrinos",
        "central",
        "seed",
        "curved",
        "reionization",
        "running",
    ]
    for jj, name_sim in enumerate(option_list):
        _archive = pnd_archive.archivePD(pick_sim=name_sim)
        _archive.average_over_samples(flag="all")
        if jj != 0:
            for ii in range(len(_archive.data)):
                archive.data.append(_archive.data[ii])
            for ii in range(len(_archive.data_av_all)):
                archive.data_av_all.append(_archive.data_av_all[ii])
        else:
            archive = copy.copy(_archive)
    return archive


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
        "d1_q1",
        "d1_kvav",
        "d1_av",
        "d1_bv",
        "d1_kp",
        "d1_q2",
    ]
    dict_param = {}
    for ii in range(params.shape[0]):
        dict_param[param_names[ii]] = params[ii]
    return dict_param


# def add_plin_to_archive(
#     folder_chains,
#     archive,
#     kmax_3d,
#     noise_3d,
#     kmax_1d,
#     noise_1d,
# ):
#     """
#     Adds Plin_nou parameter to the archive for each entry.

#     Args:
#         folder_chains (str): Path to the folder containing Plin_nou chains.
#         archive (list): List of dictionaries representing the archive.
#         kmax_3d (float): Maximum 3D wavenumber for the Plin_nou model.
#         noise_3d (float): Noise level for the Plin_nou model in 3D.
#         kmax_1d (float): Maximum 1D wavenumber for the Plin_nou model.
#         noise_1d (float): Noise level for the Plin_nou model in 1D.

#     Modifies:
#         archive (list): Updated list of dictionaries with Plin_nou parameter added.

#     Returns:
#         None
#     """
#     init = "Plin_nou"
#     for ind_book in range(len(archive)):
#         sim_label = archive[ind_book]["sim_label"]
#         ind_rescaling = archive[ind_book]["scale_tau"]
#         ind_z = archive[ind_book]["z"]

#         tag = get_flag_out(
#             init, sim_label, ind_rescaling, ind_z, kmax_3d, noise_3d, kmax_1d, noise_1d
#         )

#         archive[ind_book]["Plin_nou"] = np.load(folder_chains + tag + ".npy")


def get_flag_out(
    init,
    sim_label,
    scale_tau,
    ind_z,
    kmax_3d,
    noise_3d,
    kmax_1d,
    noise_1d,
):
    """
    Generates a flag string based on input parameters.

    Args:
        sim_label (int): Index of the simulation.
        ind_rescaling (int): Index of the scale tau.
        ind_z (int): Index of the redshift.
        kmax_3d (float): Maximum value of k for 3D power spectrum.
        noise_3d (float): Noise for 3D power spectrum.
        kmax_1d (float): Maximum value of k for 1D power spectrum.
        noise_1d (float): Noise for 1D power spectrum.

    Returns:
        str: The generated flag string.
    """
    flag = (
        init
        + "_sim"
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
    return flag


def get_std_kp1d(sim_label, ind_rescaling, ind_z, err_p1d):
    """
    Calculates the standard deviation of the 1D power spectrum.

    Args:
        sim_label (int): Index of the simulation.
        ind_rescaling (int): Index of the scale tau.
        ind_z (int): Index of the redshift.
        err_p1d (dict): Error information for the 1D power spectrum.

    Returns:
        float: The standard deviation of the 1D power spectrum.
    """
    # temporal hack
    # _sim = np.argwhere(err_p1d["u_sim_label"] == sim_label)[0, 0]
    # _tau = np.argwhere(err_p1d["u_ind_rescaling"] == ind_rescaling)[0, 0]

    _sim = np.argwhere("mpg_" + str(err_p1d["u_sim_label"]) == sim_label)[0, 0]
    _tau = np.argwhere(err_p1d["u_ind_tau"] == ind_rescaling)[0, 0]

    _z = np.argwhere(err_p1d["u_ind_z"] == ind_z)[0, 0]
    av_pk = err_p1d["p1d_sim_tau_z"][_sim, _tau, _z] * err_p1d["k"] / np.pi
    std_kpk = err_p1d["sm_rel_err"] * av_pk
    return std_kpk


def get_std_kp3d(sim_label, ind_rescaling, ind_z, err_p3d, sm_pk=False):
    """
    Calculates the standard deviation of the 3D power spectrum.

    Args:
        sim_label (int): Index of the simulation.
        ind_rescaling (int): Index of the scale tau.
        ind_z (int): Index of the redshift.
        err_p3d (dict): Error information for the 3D power spectrum.
        sm_pk (bool, optional): Boolean flag to use smoothed power spectrum. Defaults to False.

    Returns:
        float: The standard deviation of the 3D power spectrum.
    """
    _sim = np.argwhere("mpg_" + str(err_p3d["u_sim_label"]) == sim_label)[0, 0]
    _tau = np.argwhere(err_p3d["u_ind_tau"] == ind_rescaling)[0, 0]
    _z = np.argwhere(err_p3d["u_ind_z"] == ind_z)[0, 0]
    if sm_pk:
        pk = err_p3d["sm_p3d_sim_tau_z"][_sim, _tau, _z]
    else:
        pk = err_p3d["p3d_sim_tau_z"][_sim, _tau, _z]

    av_pk = pk * err_p3d["k"] ** 3 / 2 / np.pi**2
    std_kpk = err_p3d["sm_rel_err"] * av_pk
    return std_kpk


def get_input_data(
    folder_interp,
    data,
    err_p3d,
    err_p1d,
    kmax_3d,
    noise_3d,
    kmax_1d,
    noise_1d,
):
    """
    Retrieves input data and generates a model for an emulator.

    Args:
        data (dict): Input data dictionary.
        err_p3d (float): Error for 3D power spectrum.
        err_p1d (float): Error for 1D power spectrum.

    Returns:
        tuple: A tuple containing the data dictionary, the model, and the linear power spectrum.
    """
    data_dict = {}
    data_dict["units"] = "N"
    data_dict["z"] = np.atleast_1d(data["z"])
    data_dict["k3d"] = data["k3d_Mpc"]
    data_dict["mu3d"] = data["mu3d"]
    data_dict["p3d"] = data["p3d_Mpc"] * data["k3d_Mpc"] ** 3 / 2 / np.pi**2
    data_dict["std_p3d"] = err_p3d * data_dict["p3d"]

    data_dict["k1d"] = data["k_Mpc"]
    data_dict["p1d"] = data["p1d_Mpc"] * data["k_Mpc"] / np.pi
    data_dict["std_p1d"] = err_p1d * data_dict["p1d"]

    data_dict["emu_params_names"] = np.array(
        [
            "Delta2_p",
            "n_p",
            "mF",
            "sigT_Mpc",
            "gamma",
            "kF_Mpc",
        ]
    )
    data_dict["emu_params"] = np.array(
        [
            data["Delta2_p"],
            data["n_p"],
            data["mF"],
            data["sigT_Mpc"],
            data["gamma"],
            data["kF_Mpc"],
        ]
    )

    flag = data["sim_label"][4:]
    file_plin_inter = folder_interp + "Plin_interp_sim" + flag + ".npy"

    # load linear Plin interpolator
    pk_interp = np.load(file_plin_inter, allow_pickle=True).all()

    model = model_p3d_arinyo.ArinyoModel(camb_pk_interp=pk_interp)

    # linp = (
    #     model.linP_Mpc(z=data_dict["z"][0], k_Mpc=data_dict["k3d"])
    #     * data_dict["k3d"] ** 3
    #     / 2
    #     / np.pi**2
    # )

    return data_dict, model


def data_for_emu_v1(
    folder_best_fits,
    folder_interp,
    archive,
    err_p3d,
    err_p1d,
    kmax_3d,
    noise_3d,
    kmax_1d,
    noise_1d,
):
    """
    Collects data for emulator v1.

    Args:
        folder_best_fits (str): The folder path where the best fit files are located.
        archive (list): List of archive data.
        err_p3d (float): Error for 3D power spectrum.
        err_p1d (float): Error for 1D power spectrum.
        kmax_3d (float): Maximum value of k for 3D power spectrum.
        noise_3d (float): Noise for 3D power spectrum.
        kmax_1d (float): Maximum value of k for 1D power spectrum.
        noise_1d (float): Noise for 1D power spectrum.

    Returns:
        dict: A dictionary containing the collected data.
    """

    nsamples = len(archive)
    n_in_params = 6  # cosmo + IGM
    n_out_params = 8  # Arinyo

    data_emu = {}
    # cosmo + IGM
    data_emu["in_params"] = np.zeros((nsamples, n_in_params))
    # Arinyo
    data_emu["out_params"] = np.zeros((nsamples, n_out_params))
    # function for evaluating loss function
    data_emu["model"] = []

    for ind_book in range(nsamples):
        data_dict, model = get_input_data(
            folder_interp,
            archive[ind_book],
            err_p3d,
            err_p1d,
            kmax_3d,
            noise_3d,
            kmax_1d,
            noise_1d,
        )

        fit = FitPk(
            data_dict,
            model,
            fit_type="both",
            k3d_max=kmax_3d,
            k1d_max=kmax_1d,
            noise_3d=noise_3d,
            noise_1d=noise_1d,
        )

        sim_label = archive[ind_book]["sim_label"][4:]
        val_scaling = archive[ind_book]["val_scaling"]
        zz = archive[ind_book]["z"]

        init = "fit"
        tag = get_flag_out(
            init,
            sim_label,
            val_scaling,
            zz,
            kmax_3d,
            noise_3d,
            kmax_1d,
            noise_1d,
        )
        file_best = np.load(folder_best_fits + tag + ".npz")

        if ind_book == 0:
            data_emu["emu_params_names"] = data_dict["emu_params_names"]

        data_emu["in_params"][ind_book] = data_dict["emu_params"]
        data_emu["out_params"][ind_book] = file_best["best_params"]
        data_emu["model"].append(fit)

    return data_emu
