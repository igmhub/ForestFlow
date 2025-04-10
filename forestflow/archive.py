import numpy as np
import os

# import matplotlib.pyplot as plt
# from itertools import product

from lace.archive.gadget_archive import GadgetArchive
from lace.utils.exceptions import ExceptionList

# from lace.utils.misc import split_string

from forestflow.utils import (
    params_numpy2dict,
    params_numpy2dict_minimizer,
    params_numpy2dict_minimizerz,
)
from forestflow.model_p3d_arinyo import ArinyoModel


def get_camb_interp(file, data):
    from lace.cosmo import camb_cosmo
    from forestflow.camb_routines import get_matter_power_interpolator

    # check if Plin interporlator has been pre-computed for this simulation
    # if not, do it (to be fixed)
    if os.path.isfile(file) == False:
        cosmo = camb_cosmo.get_cosmology_from_dictionary(data["cosmo_params"])
        # get model
        zs = np.arange(2.0, 4.75, 0.25)
        camb_results = camb_cosmo.get_camb_results(
            cosmo, zs=zs, camb_kmax_Mpc=200
        )

        # get interpolator directly so we do not have to run camb each time
        pk_interp = get_matter_power_interpolator(
            camb_results,
            nonlinear=False,
            var1=8,
            var2=8,
            hubble_units=False,
            k_hunit=False,
            log_interp=True,
        )

        # save linear Plin interpolator
        np.save(file, pk_interp)
    else:
        pk_interp = np.load(file, allow_pickle=True).all()

    return pk_interp


def paramz_to_paramind(z, paramz):
    paramind = []
    for ii in range(len(z)):
        param = {}
        for key in paramz:
            param[key] = 10 ** np.poly1d(paramz[key])(z[ii])
        paramind.append(param)
    return paramind


class GadgetArchive3D(GadgetArchive):
    def __init__(
        self,
        folder_data=None,
        folder_chains=None,
        base_folder="/home/jchaves/Proyectos/projects/lya/lya_pk/",
        folder_interp="/data/plin_interp/",
        file_errors="/data/std_pnd_mpg.npz",
        file_plin="plin.npz",
        postproc="Cabayol23",
        average="both",
        kmax_3d=5,
        noise_3d=0.075,
        kmax_1d=5,
        noise_1d=0.01,
        kp_Mpc=None,
        force_recompute_linP_params=False,
        force_recompute_plin=False,
        verbose=False,
    ):
        """
        Archive

        Args:

            folder_data (str): Path to the folder containing the file with best-fitting parameters.
            folder_chains (str): Path to the folder containing Arinyo chains.
            kmax_3d (float): Maximum 3D wavenumber for the Arinyo model.
            noise_3d (float): Noise level for the Arinyo model in 3D.
            kmax_1d (float): Maximum 1D wavenumber for the Arinyo model.
            noise_1d (float): Noise level for the Arinyo model in 1D.

        """

        self.emu_params = [
            "Delta2_p",
            "n_p",
            "mF",
            "sigT_Mpc",
            "gamma",
            "kF_Mpc",
        ]

        self.average = average
        self.kmax_3d = kmax_3d
        self.noise_3d = noise_3d
        self.kmax_1d = kmax_1d
        self.noise_1d = noise_1d
        self.folder_data = folder_data
        self.folder_chains = folder_chains
        self.base_folder = base_folder
        self.file_plin = file_plin
        self.folder_interp = base_folder + folder_interp

        err_pnd = np.load(base_folder + file_errors)
        self.rel_err_p1d = err_pnd["std_p1d"]
        self.rel_err_p3d = err_pnd["std_p3d"]

        super().__init__(
            postproc=postproc,
            kp_Mpc=kp_Mpc,
            force_recompute_linP_params=force_recompute_linP_params,
            verbose=verbose,
        )

        self.training_data = self.get_training_data(
            self.emu_params, average=average
        )
        self.add_Arinyo_model(self.training_data)

        # mcmc chains, only computed for both
        if average == "both":
            self.add_Arinyo_fits(
                self.training_data,
                "mpg_hypercube",
            )
            self.add_Arinyo_minimizer_indiv(
                self.training_data, sim_label="mpg_hypercube"
            )
            self.add_Arinyo_minimizer_joint(
                self.training_data, sim_label="mpg_hypercube"
            )
        # # minimizer, computed for both, axes, and phases
        # self.add_Arinyo_minimizer(self.training_data)

        self.add_plin(
            self.training_data,
            "mpg_hypercube",
            compute_plin=force_recompute_plin,
        )

        self.pk_fid = self.training_data[0]["Plin"]
        self.pk_fid_p1d = self.training_data[0]["Plin_for_p1d"]

    def get_testing_data(
        self,
        sim_label,
        ind_rescaling=0,
        force_recompute_plin=False,
        kmax_3d=5,
        kmax_1d=4,
    ):
        testing_data = super().get_testing_data(
            sim_label, ind_rescaling=ind_rescaling
        )
        self.add_Arinyo_fits(testing_data, sim_label)
        self.add_Arinyo_minimizer_indiv(
            testing_data, sim_label, kmax_3d, kmax_1d
        )
        self.add_Arinyo_minimizer_joint(testing_data, sim_label)
        self.add_Arinyo_model(testing_data)
        self.add_plin(
            testing_data,
            sim_label,
            compute_plin=force_recompute_plin,
        )

        return testing_data

    def add_Arinyo_fits(self, archive, sim_label):
        """
        Adds Arinyo parameters to the archive for each entry.

        Modifies:
            training_data (list): Updated list of dictionaries with Arinyo parameters added.

        Returns:
            None
        """

        read = False
        if self.folder_data is not None:
            flag = (
                sim_label
                + "_both"
                + "_kmax3d"
                + str(self.kmax_3d)
                + "_noise3d"
                + str(self.noise_3d)
                + "_kmax1d"
                + str(self.kmax_1d)
                + "_noise1d"
                + str(self.noise_1d)
                + ".npy"
            )
            try:
                file_arinyo = np.load(self.folder_data + flag)
            except OSError:
                # check out Input_emu_v0.ipynb for creating this file
                print(
                    "No file with best-fitting parameters in "
                    + self.folder_data
                )

            else:
                read = True

        nelem = len(archive)
        if read:
            for ind_book in range(nelem):
                archive[ind_book]["Arinyo"] = params_numpy2dict(
                    file_arinyo[ind_book, :, 0]
                )
                archive[ind_book]["Arinyo_25"] = params_numpy2dict(
                    file_arinyo[ind_book, :, 1]
                )
                archive[ind_book]["Arinyo_75"] = params_numpy2dict(
                    file_arinyo[ind_book, :, 2]
                )
        else:
            print(
                "Reading best-fitting parameters from chains (it takes longer)"
            )

            for ind_book in range(nelem):
                _sim_label = archive[ind_book]["sim_label"]
                _scale_tau = archive[ind_book]["val_scaling"]
                _ind_z = archive[ind_book]["z"]

                tag = (
                    "fit_sim"
                    + _sim_label[4:]
                    + "_tau"
                    + str(np.round(_scale_tau, 2))
                    + "_z"
                    + str(_ind_z)
                    + "_kmax3d"
                    + str(self.kmax_3d)
                    + "_noise3d"
                    + str(self.noise_3d)
                    + "_kmax1d"
                    + str(self.kmax_1d)
                    + "_noise1d"
                    + str(self.noise_1d)
                )
                # check folder is not None
                file_arinyo = np.load(self.folder_chains + tag + ".npz")
                archive[ind_book]["Arinyo"] = params_numpy2dict(
                    file_arinyo["best_params"]
                )
                _ = np.percentile(file_arinyo["chain"], 25, axis=0)
                archive[ind_book]["Arinyo_25"] = params_numpy2dict(_)
                _ = np.percentile(file_arinyo["chain"], 75, axis=0)
                archive[ind_book]["Arinyo_75"] = params_numpy2dict(_)

    def add_Arinyo_minimizer(self, archive):
        file = (
            self.base_folder + "/data/best_arinyo/mpg_" + self.average + ".npz"
        )
        best_params = np.load(file)["out_params"]
        nelem = len(archive)
        for ii in range(nelem):
            archive[ii]["Arinyo_minin"] = params_numpy2dict(best_params[ii])

    def add_Arinyo_minimizer_indiv(
        self,
        archive,
        sim_label=None,
        kmax_3d=5,
        kmax_1d=4,
    ):
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

        if sim_label == "mpg_hypercube":
            ii = 0
            for isim in range(30):
                ind_sim = archive[ii]["sim_label"]
                flag = get_flag_out(ind_sim, kmax_3d, kmax_1d)
                file = (
                    self.base_folder
                    + "/data/best_arinyo/minimizer/"
                    + flag
                    + ".npz"
                )
                data = np.load(file, allow_pickle=True)
                best_params = data["best_params"]
                ind_snap = data["ind_snap"]
                val_scaling = data["val_scaling"]

                nelem = len(best_params)
                for jj in range(nelem):
                    if ind_sim != archive[ii]["sim_label"]:
                        raise ValueError("sim_label does not match")

                    ind = np.argwhere(
                        (ind_snap == archive[ii]["ind_snap"])
                        & (val_scaling == archive[ii]["val_scaling"])
                    )[0, 0]

                    # archive[ii]["Arinyo_min"] = params_numpy2dict_minimizer(
                    #     best_params[ind, :, 0]
                    # )
                    archive[ii]["Arinyo_min"] = best_params[ind]
                    archive[ii]["Arinyo_min"]["q1"] = np.abs(
                        archive[ii]["Arinyo_min"]["q1"]
                    )
                    archive[ii]["Arinyo_min"]["q2"] = np.abs(
                        archive[ii]["Arinyo_min"]["q2"]
                    )
                    ii += 1
        else:
            flag = get_flag_out(sim_label, kmax_3d, kmax_1d)
            file = (
                self.base_folder
                + "/data/best_arinyo/minimizer/"
                + flag
                + ".npz"
            )
            data = np.load(file, allow_pickle=True)
            best_params = data["best_params"]
            ind_snap = data["ind_snap"]
            val_scaling = data["val_scaling"]

            nelem = len(best_params)
            for ii in range(nelem):
                ind = np.argwhere(
                    (ind_snap == archive[ii]["ind_snap"])
                    & (val_scaling == archive[ii]["val_scaling"])
                )[0, 0]
                archive[ii]["Arinyo_min"] = best_params[ind]
                archive[ii]["Arinyo_min"]["q1"] = np.abs(
                    archive[ii]["Arinyo_min"]["q1"]
                )
                archive[ii]["Arinyo_min"]["q2"] = np.abs(
                    archive[ii]["Arinyo_min"]["q2"]
                )

    def add_Arinyo_minimizer_joint(
        self,
        archive,
        sim_label=None,
        kmax_3d=3,
        kmax_1d=3,
    ):
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

        if sim_label == "mpg_hypercube":
            nsim = 30
            arr_val_scaling = [0.9, 0.95, 1.0, 1.05, 1.1]
        else:
            nsim = 1
            arr_val_scaling = [1.0]
        z = self.list_sim_redshifts.copy()

        id_sim_label = []
        id_val_scaling = []
        id_z = []
        arr_params = []

        for isim in range(nsim):
            if sim_label == "mpg_hypercube":
                ind_sim = "mpg_" + str(isim)
            else:
                ind_sim = sim_label
            for val_scaling in arr_val_scaling:
                flag = get_flag_out(ind_sim, val_scaling, kmax_3d, kmax_1d)
                file = (
                    self.base_folder
                    + "/data/best_arinyo/minimizer_z/"
                    + flag
                    + ".npz"
                )
                data = np.load(file, allow_pickle=True)
                best_params = paramz_to_paramind(z, data["best_params"].item())
                for iz in range(len(z)):
                    id_sim_label.append(ind_sim)
                    id_val_scaling.append(np.round(val_scaling, 2))
                    id_z.append(np.round(z[iz], 2))
                    arr_params.append(best_params[iz])

        id_sim_label = np.array(id_sim_label)
        id_val_scaling = np.array(id_val_scaling)
        id_z = np.array(id_z)

        for ii in range(len(archive)):
            _ = np.argwhere(
                (archive[ii]["sim_label"] == id_sim_label)
                & (np.round(archive[ii]["z"], 2) == id_z)
                & (np.round(archive[ii]["val_scaling"], 2) == id_val_scaling)
            )[0, 0]
            archive[ii]["Arinyo_minz"] = params_numpy2dict_minimizerz(
                arr_params[_]
            )

    def add_Arinyo_model(
        self,
        archive,
    ):
        nelem = len(archive)

        for ind_book in range(nelem):
            # load linear Plin interpolator
            file = (
                self.folder_interp
                + "Plin_interp_sim"
                + archive[ind_book]["sim_label"][4:]
                + ".npy"
            )
            pk_interp = get_camb_interp(file, archive[ind_book])

            # add Arinyo model to archive
            archive[ind_book]["model"] = ArinyoModel(camb_pk_interp=pk_interp)
            z = archive[ind_book]["z"]
            k3d = archive[ind_book]["k3d_Mpc"]
            plin = archive[ind_book]["model"].linP_Mpc(z, k3d)
            archive[ind_book]["Plin"] = plin

    def add_plin(
        self,
        archive,
        sim_label,
        compute_plin=False,
        k_perp_min=0.001,
        k_perp_max=100,
        n_k_perp=99,
    ):
        nelem = len(archive)

        try:  # check file_plin is not None
            file = np.load(
                self.base_folder + "/data/" + sim_label + "_" + self.file_plin
            )
        except OSError:
            print("No file with Plin in " + self.base_folder + "/data/")
            compute_plin = True

        if compute_plin:
            k3d = archive[0]["k3d_Mpc"]

            # for computing plin for p1d integration
            kpar = archive[0]["k_Mpc"].copy()
            ln_k_perp = np.linspace(
                np.log(k_perp_min), np.log(k_perp_max), n_k_perp
            )
            dlnk = ln_k_perp[1] - ln_k_perp[0]
            k_perp = np.exp(ln_k_perp)
            k = np.sqrt(kpar[np.newaxis, :] ** 2 + k_perp[:, np.newaxis] ** 2)
            k_for_p1d = k.swapaxes(0, 1)
        else:
            file = np.load(
                self.base_folder + "/data/" + sim_label + "_" + self.file_plin
            )
            file_ind_snap = file["ind_snap"]
            file_sim_label = file["sim_label"]
            file_val_scaling = file["val_scaling"]
            file_Plin = file["Plin"]
            file_Plin_for_p1d = file["Plin_for_p1d"]

        for ind_book in range(nelem):
            if compute_plin:
                z = archive[ind_book]["z"]
                plin = archive[ind_book]["model"].linP_Mpc(z, k3d)
                plin_for_p1d = archive[ind_book]["model"].linP_Mpc(z, k_for_p1d)
            else:
                _ind = np.argwhere(
                    (archive[ind_book]["ind_snap"] == file_ind_snap)
                    & (archive[ind_book]["sim_label"] == file_sim_label)
                    & (
                        np.round(archive[ind_book]["val_scaling"], 3)
                        == np.round(file_val_scaling, 3)
                    )
                )[0, 0]
                plin = file_Plin[_ind]
                plin_for_p1d = file_Plin_for_p1d[_ind]

            archive[ind_book]["Plin"] = plin
            archive[ind_book]["Plin_for_p1d"] = plin_for_p1d

        if compute_plin:
            _Plin = np.zeros(
                (
                    nelem,
                    *archive[0]["Plin"].shape,
                )
            )
            _Plin_for_p1d = np.zeros((nelem, 676, 99))
            _ind_snap = np.zeros((nelem), dtype=int)
            _sim_label = []
            _val_scaling = np.zeros((nelem))

            for ind_book in range(nelem):
                _ind_snap[ind_book] = archive[ind_book]["ind_snap"]
                _sim_label.append(archive[ind_book]["sim_label"])
                _val_scaling[ind_book] = archive[ind_book]["val_scaling"]

                _Plin[ind_book] = archive[ind_book]["Plin"]
                _Plin_for_p1d[ind_book] = archive[ind_book]["Plin_for_p1d"]

            np.savez(
                self.base_folder + "/data/" + sim_label + "_" + self.file_plin,
                Plin=_Plin,
                Plin_for_p1d=_Plin_for_p1d,
                ind_snap=_ind_snap,
                sim_label=np.array(_sim_label),
                val_scaling=_val_scaling,
            )
