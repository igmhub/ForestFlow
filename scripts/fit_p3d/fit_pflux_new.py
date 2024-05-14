import numpy as np
import sys, os

from forestflow.model_p3d_arinyo import ArinyoModel
from forestflow.fit_p3d import FitPk
from forestflow.archive import GadgetArchive3D
from forestflow.rebin_p3d import get_p3d_modes


def get_default_params():
    parameters_q = {
        "bias": 0.12,
        "beta": 1.4,
        "q1": 0.4,
        "kvav": 0.6,
        "av": 0.3,
        "bv": 1.5,
        "kp": 18.0,
    }
    parameters_q2 = {
        "bias": 0.12,
        "beta": 1.4,
        "q1": 0.4,
        "kvav": 0.6,
        "av": 0.3,
        "bv": 1.5,
        "kp": 18.0,
        "q2": 0.2,
    }

    priors_q = {
        "bias": [0, 1],
        "beta": [0, 5.0],
        "q1": [0, 5],
        "kvav": [0.1, 5.0],
        "av": [0, 2],
        "bv": [0, 5],
        "kp": [1, 50],
    }
    priors_q2 = {
        "bias": [0, 1],
        "beta": [0, 5.0],
        "q1": [0, 5],
        "kvav": [0.1, 5.0],
        "av": [0, 2],
        "bv": [0, 5],
        "kp": [1, 50],
        "q2": [0, 5],
    }

    return parameters_q, priors_q, parameters_q2, priors_q2


def get_flag_out(ind_sim, kmax_3d, kmax_1d):
    flag = (
        "fit_sim_label_"
        + str(ind_sim)
        + "_kmax3d_"
        + str(kmax_3d)
        + "_kmax1d_"
        + str(kmax_1d)
    )
    return flag


def get_input_data(data, kmax_fit):
    data_dict = {}
    data_dict["z"] = np.atleast_1d(data["z"])
    data_dict["kmu_modes"] = get_p3d_modes(kmax_fit)

    data_dict["k3d_Mpc"] = data["k3d_Mpc"]
    data_dict["mu3d"] = data["mu3d"]
    data_dict["p3d_Mpc"] = data["p3d_Mpc"]

    n_modes = np.zeros_like(data_dict["k3d_Mpc"])
    for ii in range(data_dict["k3d_Mpc"].shape[0]):
        for jj in range(data_dict["k3d_Mpc"].shape[1]):
            key = f"{ii}_{jj}_k"
            if key in data_dict["kmu_modes"]:
                n_modes[ii, jj] = data_dict["kmu_modes"][key].shape[0]

    data_dict["std_p3d"] = 1 / n_modes

    data_dict["k1d_Mpc"] = data["k_Mpc"]
    data_dict["p1d_Mpc"] = data["p1d_Mpc"]
    data_dict["std_p1d"] = np.pi / data["k_Mpc"] / np.mean(n_modes.sum(axis=0))

    model = data["model"]

    return data_dict, model


def main():
    path_program = "/home/jchaves/Proyectos/projects/lya/ForestFlow/"
    folder_lya_data = path_program + "/data/best_arinyo/"
    folder_save = (
        "/home/jchaves/Proyectos/projects/lya/data/forestflow/fits_modes/"
    )

    Archive3D = GadgetArchive3D(
        base_folder=path_program[:-1],
        folder_data=folder_lya_data,
        force_recompute_plin=False,
        average="both",
    )
    print(len(Archive3D.training_data))

    # fit options
    kmax_3d = 3
    kmax_1d = 3
    fit_type = "both"
    all_q = [True]

    # loop sim_labels
    for sim_label in Archive3D.list_sim:
        print(sim_label)
        print()
        print()
        if sim_label in Archive3D.list_sim_cube:
            list_sim_use = []
            for isim in Archive3D.training_data:
                if isim["sim_label"] == sim_label:
                    list_sim_use.append(isim)
        else:
            list_sim_use = Archive3D.get_testing_data(sim_label)

        res_params = np.zeros((len(list_sim_use), 8, len(all_q)))
        res_chi2 = np.zeros((len(list_sim_use), len(all_q)))
        ind_snap = np.zeros((len(list_sim_use)))
        val_scaling = np.zeros((len(list_sim_use)))

        # loop snapshots/scaligs
        for isim, sim_use in enumerate(list_sim_use):
            # w/o and w/ q2
            ind_snap[isim] = sim_use["ind_snap"]
            val_scaling[isim] = sim_use["val_scaling"]
            print()
            print(val_scaling[isim], sim_use["z"])
            for iq2, use_q2 in enumerate(all_q):
                _ = get_default_params()
                parameters_q, priors_q, parameters_q2, priors_q2 = _
                if use_q2:
                    parameters = parameters_q2
                    priors = priors_q2
                else:
                    parameters = parameters_q
                    priors = priors_q

                # set initial conditions for the fit
                for ii, par in enumerate(parameters):
                    parameters[par] = np.abs(sim_use["Arinyo"][par])
                parameters["beta"] += 0.2

                # get input data
                data_dict, model = get_input_data(sim_use, kmax_3d)

                # set fitting model
                fit = FitPk(
                    data_dict,
                    model,
                    fit_type=fit_type,
                    k3d_max=kmax_3d,
                    k1d_max=kmax_1d,
                    priors=priors,
                    all_kmu=True,
                )

                chia = fit.get_chi2(parameters)
                best_fit_params = parameters.copy()
                print("Initial chi2", chia)

                results, _best_fit_params = fit.maximize_likelihood(
                    best_fit_params
                )
                chi2 = fit.get_chi2(_best_fit_params)
                if chi2 < chia:
                    chia = chi2
                    best_fit_params = _best_fit_params.copy()
                # the output is chia and best_fit_params
                print("Final chi2", chia)

                val = np.array(list(best_fit_params.values()))
                res_params[isim, : val.shape[0], iq2] = val
                res_chi2[isim, iq2] = chia

        # save results to file
        # folder and name of output file
        out_file = get_flag_out(sim_label, kmax_3d, kmax_1d)
        np.savez(
            folder_save + out_file,
            chi2=res_chi2,
            best_params=res_params,
            ind_snap=ind_snap,
            val_scaling=val_scaling,
            # readme="sim, param, w/o w/ q2",
        )


if __name__ == "__main__":
    main()
