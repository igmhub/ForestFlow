import numpy as np

from lya_pk.archive import GadgetArchive3D


def save_data(
    folder_lya_data,
    sim_name,
    data,
    kmax_3d=5,
    noise_3d=0.075,
    kmax_1d=5,
    noise_1d=0.01,
):
    dict_conv = {}
    for ii, key in enumerate(data[0]["Arinyo"].keys()):
        dict_conv[key] = ii

    n_params = len(data[0]["Arinyo"].keys())
    n_sims = len(data)

    params_arinyo = np.zeros((n_sims, n_params, 3))
    flags = ["Arinyo", "Arinyo_25", "Arinyo_75"]

    for ii in range(n_sims):
        for jj in range(len(flags)):
            for a, b in data[ii][flags[jj]].items():
                params_arinyo[ii, dict_conv[a], jj] = b

    flag = (
        sim_name
        + "_both"
        + "_kmax3d"
        + str(kmax_3d)
        + "_noise3d"
        + str(noise_3d)
        + "_kmax1d"
        + str(kmax_1d)
        + "_noise1d"
        + str(noise_1d)
        + ".npy"
    )

    file = folder_lya_data + flag
    np.save(file, params_arinyo)


def main():
    folder_chains = (
        "/home/jchaves/Proyectos/projects/lya/data/pkfits/p3d_fits_new/"
    )
    folder_lya_data = path_program + "data/best_arinyo/"

    # load archive
    Archive3D = GadgetArchive3D(folder_chains=folder_chains)
    print(len(Archive3D.training_data))

    # save Arinyo params for hypercube
    sim_suite = "mpg_hypercube"
    save_data(folder_lya_data, sim_suite, Archive3D.training_data)

    # save Arinyo params for testing sims
    for sim_label in Archive3D.list_sim_test:
        testing_data = Archive3D.get_testing_data(sim_label)
        save_data(folder_lya_data, sim_label, testing_data)


if __name__ == "__main__":
    main()
