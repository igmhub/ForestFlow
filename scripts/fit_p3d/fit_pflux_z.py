import numpy as np
import forestflow
import sys, os

from forestflow.model_p3d_arinyo import ArinyoModel
from forestflow.fit_p3dz import FitPkz
from forestflow.archive import GadgetArchive3D
from forestflow.rebin_p3d import p3d_allkmu, get_p3d_modes, p3d_rebin_mu


def get_default_paramsz(folder, sim_label, z, val_scaling=1):
    names = ["bias", "beta", "q1", "kvav", "av", "bv", "kp"]

    priors = {
        "bias": [1e-3, 3],
        "beta": [5e-3, 7],
        "q1": [1e-2, 8],
        "kvav": [1e-3, 5],
        "av": [1e-3, 2],
        "bv": [1e-1, 5],
        "kp": [3, 30],
    }

    file = f"fit_sim_label_{sim_label}_kmax3d_3_kmax1d_3.npz"
    dat = np.load(folder + file)
    in_parameters = {}
    _ = dat["val_scaling"] == val_scaling
    # param_ind = dat["best_params"][_, :, 0]
    # order = np.array([2, 2, 1, 1, 2, 0, 1, 0])
    param_ind = dat["best_params"][_, :-1, 0]
    order = np.array([2, 2, 1, 1, 2, 0, 1])
    for ii in range(param_ind.shape[1]):
        _ = param_ind[:, ii] < priors[names[ii]][0]
        param_ind[_, ii] = priors[names[ii]][0]

        _ = param_ind[:, ii] > priors[names[ii]][1]
        param_ind[_, ii] = priors[names[ii]][1]

        res = np.polyfit(z, np.log10(param_ind[:, ii]), deg=order[ii])
        p = 10 ** np.poly1d(res)(z)
        print(
            names[ii],
            priors[names[ii]][0],
            np.min(p),
            priors[names[ii]][1],
            np.max(p),
        )
        in_parameters[names[ii]] = res

    return in_parameters, param_ind, order, priors


def get_flag_out(ind_sim, val_scaling, kmax_3d, kmax_1d):
    flag = (
        "fit_sim_label_"
        + str(ind_sim)
        + "_val_scaling_"
        + str(np.round(val_scaling, 2))
        + "_kmax3d_"
        + str(kmax_3d)
        + "_kmax1d_"
        + str(kmax_1d)
    )
    return flag


def get_input_dataz(list_data, kmax_fit):
    data_dict = {}

    data_dict["z"] = np.array([d["z"] for d in list_data])
    data_dict["kmu_modes"] = get_p3d_modes(kmax_fit)
    nz = data_dict["z"].shape[0]

    data_dict["k3d_Mpc"] = list_data[0]["k3d_Mpc"]
    data_dict["mu3d"] = list_data[0]["mu3d"]
    data_dict["k1d_Mpc"] = list_data[0]["k_Mpc"]

    n_modes = np.zeros_like(data_dict["k3d_Mpc"])
    for ii in range(data_dict["k3d_Mpc"].shape[0]):
        for jj in range(data_dict["k3d_Mpc"].shape[1]):
            key = f"{ii}_{jj}_k"
            if key in data_dict["kmu_modes"]:
                n_modes[ii, jj] = data_dict["kmu_modes"][key].shape[0]

    data_dict["p3d_Mpc"] = np.zeros((nz, *data_dict["k3d_Mpc"].shape))
    data_dict["std_p3d"] = np.zeros_like(data_dict["p3d_Mpc"])
    data_dict["p1d_Mpc"] = np.zeros((nz, data_dict["k1d_Mpc"].shape[0]))
    data_dict["std_p1d"] = np.zeros_like(data_dict["p1d_Mpc"])
    model = []
    for ii in range(nz):
        data_dict["p3d_Mpc"][ii] = list_data[ii]["p3d_Mpc"]
        data_dict["p1d_Mpc"][ii] = list_data[ii]["p1d_Mpc"]
        normz = data_dict["z"][ii] ** 3
        data_dict["std_p3d"][ii] = (
            normz / n_modes * data_dict["k3d_Mpc"] ** 0.25
        )
        data_dict["std_p1d"][ii] = (
            normz * np.pi / data_dict["k1d_Mpc"] / np.mean(n_modes.sum(axis=0))
        )
        model.append(list_data[ii]["model"])

    return data_dict, model


def main():
    args = sys.argv[1:]

    path_program = forestflow.__path__[0][:-10]
    print(path_program)
    folder_lya_data = path_program + "/data/best_arinyo/"
    folder_save = path_program + "/data/best_arinyo/minimizer_z/"
    folder_minimizer = path_program + "data/best_arinyo/minimizer/"

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
    all_val_scaling = np.array([0.9, 0.95, 1.0, 1.05, 1.1])
    maxiter = 5000

    # loop sim_labels
    for sim_label in Archive3D.list_sim:
        if (args[0] != "") and (sim_label == args[0]):
            pass
        else:
            continue

        print(sim_label)
        print()
        print()

        if sim_label in Archive3D.list_sim_cube:
            scalings_use = all_val_scaling.shape[0]
        else:
            scalings_use = 1

        for iscaling in range(scalings_use):
            if sim_label in Archive3D.list_sim_cube:
                val_scaling = all_val_scaling[iscaling]
                list_sim_use = []
                for isim in Archive3D.training_data:
                    if (isim["sim_label"] == sim_label) and (
                        np.round(isim["val_scaling"], 2) == val_scaling
                    ):
                        list_sim_use.append(isim)
            else:
                list_sim_use = Archive3D.get_testing_data(sim_label)
                val_scaling = 1.0

            data_dict, model = get_input_dataz(list_sim_use, kmax_3d)
            parameters, param_ind, order, priors = get_default_paramsz(
                folder_minimizer, sim_label, data_dict["z"]
            )

            out_file = folder_save + get_flag_out(
                sim_label, val_scaling, kmax_3d, kmax_1d
            )

            if os.path.isfile(out_file + ".npz"):
                file_data = np.load(out_file + ".npz", allow_pickle=True)
                chi2 = file_data["chi2"]
                if chi2 < 1:
                    continue
                else:
                    parameters = file_data["best_params"].item()

            params_minimizer = np.concatenate(list(parameters.values()))

            names = np.array(list(parameters.keys())).reshape(-1)

            # set fitting model
            fit = FitPkz(
                data_dict,
                model,
                names=names,
                priors=priors,
                fit_type=fit_type,
                k3d_max=kmax_3d,
                k1d_max=kmax_1d,
                order=order,
                verbose=False,
                maxiter=maxiter,
            )

            chia = fit.get_chi2(params_minimizer)
            print("Initial chi2", chia)

            results, best_fit_params = fit.maximize_likelihood(params_minimizer)
            params_minimizer = np.concatenate(list(best_fit_params.values()))

            chi2 = fit.get_chi2(params_minimizer)
            print("Final chi2", chi2)
            print("and best_params", best_fit_params)

            # save results to file
            # folder and name of output file

            np.savez(out_file, chi2=chi2, best_params=best_fit_params)
            print("Saved to", out_file)


if __name__ == "__main__":
    main()
