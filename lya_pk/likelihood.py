import numpy as np

# import matplotlib.pyplot as plt
# from itertools import product

from lace.archive.gadget_archive import GadgetArchive
from lace.utils.exceptions import ExceptionList

from lya_pk.fit_p3d import FitPk


class Likelihood(object):
    def __init__(
        self,
        data,
        rel_err_p3d,
        rel_err_p1d,
        kmax_3d=5,
        noise_3d=0.075,
        kmax_1d=5,
        noise_1d=0.01,
        fit_type="both",
        verbose=False,
    ):
        """
        Archive

        Args:

            kmax_3d (float): Maximum 3D wavenumber for the Arinyo model.
            noise_3d (float): Noise level for the Arinyo model in 3D.
            kmax_1d (float): Maximum 1D wavenumber for the Arinyo model.
            noise_1d (float): Noise level for the Arinyo model in 1D.

        """

        data_dict = {}

        data_dict["z"] = np.atleast_1d(data["z"])
        # no units for P1D and P3D
        data_dict["units"] = "N"

        # P3D
        data_dict["k3d"] = data["k3d_Mpc"]
        data_dict["mu3d"] = data["mu3d"]
        data_dict["p3d"] = (
            data["p3d_Mpc"] * data["k3d_Mpc"] ** 3 / 2 / np.pi**2
        )
        data_dict["std_p3d"] = rel_err_p3d * data_dict["p3d"]

        # P1D
        data_dict["k1d"] = data["k_Mpc"]
        data_dict["p1d"] = data["p1d_Mpc"] * data["k_Mpc"] / np.pi
        data_dict["std_p1d"] = rel_err_p1d * data_dict["p1d"]
        data_dict["Plin"] = data["Plin"]
        


        self.like = FitPk(
            data_dict,
            data["model"],
            fit_type=fit_type,
            k3d_max=kmax_3d,
            k1d_max=kmax_1d,
            noise_3d=noise_3d,
            noise_1d=noise_1d,
        )
