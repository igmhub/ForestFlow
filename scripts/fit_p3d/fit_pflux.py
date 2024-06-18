import numpy as np
import sys, os

from forestflow.model_p3d_arinyo import ArinyoModel
from forestflow.fit_p3d import FitPk
from forestflow.archive import GadgetArchive3D
from forestflow.rebin_p3d import get_p3d_modes
from forestflow.utils import params_numpy2dict


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


def get_input_data(z_use, data, kmax_fit):
    # these numbers come from notebook Fit_Arinyo
    p1d_res = [0.9887013, -2.94078465]
    p3d_res = [1.11635295, -3.32955821]
    mu_res = [1.1985935, -0.11867367, 0.00485981]
    alpha1d = 500
    alpha3d = 0.25
    k0_p1d = 2
    k0_p3d = 3

    data_dict = {}

    data_dict["z"] = z_use
    data_dict["kmu_modes"] = get_p3d_modes(kmax_fit)

    data_dict["k3d_Mpc"] = data["k3d_Mpc"]
    data_dict["mu3d"] = data["mu3d"]
    data_dict["k1d_Mpc"] = data["k_Mpc"]

    n_modes = np.zeros_like(data_dict["k3d_Mpc"])
    for ii in range(data_dict["k3d_Mpc"].shape[0]):
        for jj in range(data_dict["k3d_Mpc"].shape[1]):
            key = f"{ii}_{jj}_k"
            if key in data_dict["kmu_modes"]:
                n_modes[ii, jj] = data_dict["kmu_modes"][key].shape[0]

    data_dict["p3d_Mpc"] = np.zeros(
        (data_dict["k3d_Mpc"].shape[0], data_dict["k3d_Mpc"].shape[1])
    )
    data_dict["std_p3d"] = np.zeros_like(data_dict["p3d_Mpc"])
    data_dict["p1d_Mpc"] = np.zeros((data_dict["k1d_Mpc"].shape[0]))
    data_dict["std_p1d"] = np.zeros_like(data_dict["p1d_Mpc"])

    data_dict["p3d_Mpc"] = data["p3d_Mpc"]
    data_dict["p1d_Mpc"] = data["p1d_Mpc"]
    model = data["model"]
    norm3d = (
        alpha3d
        * n_modes
        / np.exp((1 + data_dict["z"]) * p3d_res[0] + p3d_res[1])
    )
    norm3d /= np.exp(
        data_dict["mu3d"] ** 2 * mu_res[0]
        + data_dict["mu3d"] * mu_res[1]
        + mu_res[2]
    )
    data_dict["std_p3d"] = 1 / norm3d
    norm1d = (
        (1 + data_dict["k1d_Mpc"] / k0_p1d) ** 2
        * alpha1d
        / np.exp((1 + data_dict["z"]) * p1d_res[0] + p1d_res[1])
    )
    data_dict["std_p1d"] = 1 / norm1d

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
    kmax_3d = 5
    kmax_1d = 4
    fit_type = "both"

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
            list_sim_use = Archive3D.get_testing_data(
                sim_label, kmax_3d=3, kmax_1d=3
            )

        if sim_label == "mpg_central":
            pass
        else:
            continue

        res_params = []
        res_chi2 = np.zeros((len(list_sim_use)))
        ind_snap = np.zeros((len(list_sim_use)))
        val_scaling = np.zeros((len(list_sim_use)))

        # loop snapshots/scaligs
        for isim, sim_use in enumerate(list_sim_use):
            ind_snap[isim] = sim_use["ind_snap"]
            val_scaling[isim] = sim_use["val_scaling"]
            print()
            print(sim_label, val_scaling[isim], sim_use["z"])
            print()

            parameters = sim_use["Arinyo_min"]
            parameters["q2"] = np.abs(parameters["q2"])
            print(parameters)

            params_minimizer = np.array(list(parameters.values()))
            names = np.array(list(parameters.keys())).reshape(-1)
            data_dict, model = get_input_data(sim_use["z"], sim_use, kmax_3d)

            # set fitting model
            fit = FitPk(
                data_dict,
                model,
                names=names,
                fit_type=fit_type,
                k3d_max=kmax_3d,
                k1d_max=kmax_1d,
                maxiter=400,
            )

            chia = fit.get_chi2(params_minimizer)
            print("Initial chi2", chia)

            results, best_fit_params = fit.maximize_likelihood(params_minimizer)
            params_minimizer = np.array(list(best_fit_params.values()))
            chi2 = fit.get_chi2(params_minimizer)
            print("Final chi2", chi2)
            print("and best_params", best_fit_params)

            val = np.array(list(best_fit_params.values()))
            res_params.append(best_fit_params)
            res_chi2[isim] = chia

        # save results to file
        # folder and name of output file
        out_file = get_flag_out(sim_label, kmax_3d, kmax_1d)
        np.savez(
            folder_save + out_file,
            chi2=res_chi2,
            best_params=res_params,
            ind_snap=ind_snap,
            val_scaling=val_scaling,
        )


if __name__ == "__main__":
    main()
