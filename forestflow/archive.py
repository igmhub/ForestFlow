import numpy as np
import os
from lace.archive.gadget_archive import GadgetArchive
import forestflow
from forestflow.utils import params_numpy2dict_minimizerz


class GadgetArchive3D(GadgetArchive):
    def __init__(
        self,
        base_folder=None,
        file_errors=None,
        postproc="Cabayol23",
        kp_Mpc=None,
        average="both",
    ):
        """
        Archive class for 3D simulations

        It calls the Lace GadgetArchive class and adds the Arinyo parameters
        """

        if base_folder == None:
            self.base_folder = os.path.dirname(forestflow.__path__[0])
        else:
            self.base_folder = base_folder

        if file_errors == None:
            file_errors = os.path.join(
                self.base_folder, "data", "std_pnd_mpg.npz"
            )

        err_pnd = np.load(file_errors)
        self.rel_err_p1d = err_pnd["std_p1d"]
        self.rel_err_p3d = err_pnd["std_p3d"]

        self.emu_params = [
            "Delta2_p",
            "n_p",
            "mF",
            "sigT_Mpc",
            "gamma",
            "kF_Mpc",
        ]

        super().__init__(postproc=postproc, kp_Mpc=kp_Mpc)

        self.training_data = self.get_training_data(
            self.emu_params, average=average
        )

        # mcmc chains, only computed for both
        if average == "both":
            self.add_Arinyo_minimizer_indiv(
                self.training_data, sim_label="mpg_hypercube"
            )
            self.add_Arinyo_minimizer_joint(
                self.training_data, sim_label="mpg_hypercube"
            )

    def get_testing_data(
        self, sim_label, ind_rescaling=0, kmax_3d=5, kmax_1d=4
    ):
        testing_data = super().get_testing_data(
            sim_label, ind_rescaling=ind_rescaling
        )
        self.add_Arinyo_minimizer_indiv(
            testing_data, sim_label, kmax_3d, kmax_1d
        )
        self.add_Arinyo_minimizer_joint(testing_data, sim_label)

        return testing_data

    def add_Arinyo_minimizer_indiv(
        self, archive, sim_label=None, kmax_3d=5, kmax_1d=4
    ):
        """
        Arinyo fits considering each snapshot separately
        """

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
                file = os.path.join(
                    self.base_folder,
                    "data",
                    "best_arinyo",
                    "minimizer",
                    flag + ".npz",
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

                    archive[ii]["Arinyo_min"] = best_params[ind]
                    archive[ii]["Arinyo_min"]["bias"] = -np.abs(
                        archive[ii]["Arinyo_min"]["bias"]
                    )
                    # bias_eta = bias * beta / fz
                    archive[ii]["Arinyo_min"]["bias_eta"] = (
                        archive[ii]["Arinyo_min"]["bias"]
                        * archive[ii]["Arinyo_min"]["beta"]
                        / archive[ii]["f_p"]
                    )
                    archive[ii]["Arinyo_min"]["q1"] = np.abs(
                        archive[ii]["Arinyo_min"]["q1"]
                    )
                    archive[ii]["Arinyo_min"]["q2"] = np.abs(
                        archive[ii]["Arinyo_min"]["q2"]
                    )
                    ii += 1
        else:
            flag = get_flag_out(sim_label, kmax_3d, kmax_1d)
            file = os.path.join(
                self.base_folder,
                "data",
                "best_arinyo",
                "minimizer",
                flag + ".npz",
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
                archive[ii]["Arinyo_min"]["bias"] = -np.abs(
                    archive[ii]["Arinyo_min"]["bias"]
                )
                archive[ii]["Arinyo_min"]["q1"] = np.abs(
                    archive[ii]["Arinyo_min"]["q1"]
                )
                archive[ii]["Arinyo_min"]["q2"] = np.abs(
                    archive[ii]["Arinyo_min"]["q2"]
                )

    def add_Arinyo_minimizer_joint(
        self, archive, sim_label=None, kmax_3d=3, kmax_1d=3
    ):
        """
        Fits parameterizing the redshift dependence of the Arinyo params
        """

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

        def paramz_to_paramind(z, paramz):
            paramind = []
            for ii in range(len(z)):
                param = {}
                for key in paramz:
                    param[key] = 10 ** np.poly1d(paramz[key])(z[ii])
                paramind.append(param)
            return paramind

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
                file = os.path.join(
                    self.base_folder,
                    "data",
                    "best_arinyo",
                    "minimizer_z",
                    flag + ".npz",
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
