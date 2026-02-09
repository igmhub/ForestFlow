# torch modules
import torch
from torch.utils.data import DataLoader, dataset, TensorDataset
from torch import nn, optim

# FrEIA imports
import FrEIA.framework as Ff
import FrEIA.modules as Fm

# lace models
from lace.cosmo import camb_cosmo, fit_linP
import lace

# forestflow models
import forestflow
from forestflow.model_p3d_arinyo import ArinyoModel
from forestflow.archive import GadgetArchive3D, get_camb_interp
from forestflow.likelihood import Likelihood
from forestflow.utils import (
    get_covariance,
    params_numpy2dict,
    transform_arinyo_params,
)
from forestflow.rebin_p3d import p3d_allkmu

from warnings import warn
import numpy as np
import os
import time
import random
import matplotlib.pyplot as plt
from matplotlib import rcParams
import corner
import gc
import psutil
from functools import lru_cache


rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"


def init_xavier(m):
    """Initialization of the NN.
    This is quite important for a faster training
    """
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def print_memory_usage(step_description):
    process = psutil.Process()
    memory_info = process.memory_info()
    print(
        f"{step_description} - RSS: {memory_info.rss / (1024 ** 2):.2f} MB, VMS: {memory_info.vms / (1024 ** 2):.2f} MB"
    )
    if torch.cuda.is_available():
        print(
            f"GPU memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB"
        )
        print(
            f"GPU memory cached: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB"
        )


class P3DEmulator:
    """A class for training an emulator.

    Args:
        training_data (Type): Description of training data.
        paramList (Type): Description of emulator parameters.
        kmax_Mpc (float): The maximum k in Mpc^-1 to use for training. Default is 4.
        nLayers_inn (int): Number of layers in the inner network. Default is 8.
        nepochs (int): The number of epochs to train for. Default is 100.
        batch_size (int): Size of batches during training. Default is 100.
        lr (float): Learning rate for the optimizer. Default is 1e-3.
        weight_decay (float): L2 regularization term for the optimizer. Default is 1e-4.
        step_size (int): Step size for learning rate scheduler. Default is 75.
        adamw (bool): Whether to use the AdamW optimizer. Default is False.
        train (bool): Whether to train the emulator. Default is True.
        save_path (str): Path to save the trained model. Default is None.
        model_path (str): Path to a pretrained model. Default is None.
        Archive: Archive3D object
        chain_samp (int): Chain sampling size. Default is 100000.
        Nrealizations (int): Number of realizations. Default is 100.
    """

    def __init__(
        self,
        training_data=None,
        emu_input_names=None,
        train=False,
        save_path=None,
        model_path=None,
        kmax_Mpc=4,
        nLayers_inn=12,
        nepochs=300,
        batch_size=100,
        lr=0.001,
        weight_decay=0,
        step_size=200,
        adamw=True,
        use_chains=False,
        Nrealizations=3000,
        training_type="Arinyo_min",
    ):
        # Initialize class attributes with provided arguments
        self.Nrealizations = Nrealizations

        self.Arinyo_params = [
            "bias",
            "beta",
            "q1",
            "kvav",
            "av",
            "bv",
            "kp",
        ]
        if training_type == "Arinyo_min":
            self.Arinyo_params.append("q2")
        dim_inputSpace = len(self.Arinyo_params)

        self.cosmo_fields = [
            "H0",
            "omch2",
            "ombh2",
            "mnu",
            "omk",
            "As",
            "ns",
            "nrun",
            "w",
        ]

        if train:
            self._train_emu(
                training_data,
                emu_input_names,
                adamw=adamw,
                lr=lr,
                nepochs=nepochs,
                step_size=step_size,
                use_chains=use_chains,
                train_seed=32,
                weight_decay=weight_decay,
                nLayers_inn=nLayers_inn,
                batch_size=batch_size,
                dim_inputSpace=dim_inputSpace,
                training_type=training_type,
                save_path=save_path,
                kmax_Mpc=kmax_Mpc,
            )
        elif model_path is not None:
            self._load_emu(model_path=model_path)
        else:
            raise ValueError("Either train or model_path must be provided.")

    def _get_training_data(self, training_data, emu_input_names, training_type):
        """
        Retrieve and preprocess training data for the emulator.

        This function obtains the training data from the provided archive

        Returns:
            torch.Tensor: Preprocessed training data.
        """
        # Extract relevant parameters from the training data
        input_emu = np.zeros((len(training_data), len(emu_input_names)))
        output_emu = np.zeros((len(training_data), len(self.Arinyo_params)))
        for ii in range(len(training_data)):
            for jj, par in enumerate(emu_input_names):
                input_emu[ii, jj] = training_data[ii][par]
            for jj, par in enumerate(self.Arinyo_params):
                output_emu[ii, jj] = training_data[ii][training_type][par]

        # Calculate and store the maximum and minimum values for parameter scaling
        self.input_param_lims_max = input_emu.max(axis=0)
        self.input_param_lims_min = input_emu.min(axis=0)

        self.output_param_lims_max = output_emu.max(axis=0)
        self.output_param_lims_min = output_emu.min(axis=0)

        # Scale the training data based on the parameter limits
        input_emu = (input_emu - self.input_param_lims_min) / (
            self.input_param_lims_max - self.input_param_lims_min
        )

        # output_emu[:, 2] = np.exp(output_emu[:, 2])
        # output_emu[:, 4] = np.exp(output_emu[:, 4])
        # output_emu[:, 6] = np.log(output_emu[:, 6])
        # if "q2" in self.Arinyo_params:
        #     output_emu[:, 7] = np.exp(output_emu[:, 7])

        # some special transformations applied to the output data
        output_emu = (output_emu - self.output_param_lims_min) / (
            self.output_param_lims_max - self.output_param_lims_min
        )

        # Convert the scaled training data to a torch.Tensor object
        input_emu = torch.Tensor(input_emu)
        output_emu = torch.Tensor(output_emu)

        return input_emu, output_emu

    def _get_Plin(self):
        """
        Retrieve linear power spectrum (Plin) from the training data.

        This function extracts Plin from the training data and returns it as a numpy array.

        Returns:
            np.array: Linear power spectrum (Plin) from the training data.
        """
        # Extract Plin from the training data
        Plin = [d["Plin"] for d in self.training_data]

        # Convert the list of Plin to a numpy array
        Plin = np.array(Plin)

        return Plin

    def _check_emu_params(self, emu_params, info_power, kp_Mpc=0.7):
        # check if all emulator parameters are provided
        compute_linP = False
        for param in self.emu_input_names:
            # check cosmo params
            if param in ["Delta2_p", "n_p"]:
                if param not in emu_params.keys():
                    compute_linP = True
            # check IGM params
            else:
                if param not in emu_params.keys():
                    raise ValueError(f"{param} not provided in emu_params")

        # If Delta2p and np are not provided, compute them
        if compute_linP:
            msg_err = (
                "Since Delta2_p or n_p are not provided, we need info_power"
                " to include either cosmo or sim_label, not both or none."
            )
            if info_power is None:
                raise ValueError(msg_err)
            elif (("cosmo" in info_power) and ("sim_label" in info_power)) | (
                ("cosmo" not in info_power) and ("sim_label" not in info_power)
            ):
                raise ValueError(msg_err)
            elif "cosmo" in info_power:
                cosmo = info_power["cosmo"]
            elif "sim_label" in info_power:
                repo = os.path.dirname(lace.__path__[0]) + "/"
                fname = repo + ("data/sim_suites/Australia20/mpg_emu_cosmo.npy")
                data_cosmo = np.load(fname, allow_pickle=True).item()
                cosmo = data_cosmo[sim_label]["cosmo_params"]

            # Get Delta2p and np from cosmology
            sim_cosmo = camb_cosmo.get_cosmology(**cosmo)
            linP_zs = fit_linP.get_linP_Mpc_zs(
                sim_cosmo, [info_power["z"]], kp_Mpc
            )[0]

            # add these to emu_params
            emu_params["Delta2_p"] = linP_zs["Delta2_p"]
            emu_params["n_p"] = linP_zs["n_p"]

        return emu_params

    def _get_Arinyo_coeffs(
        self,
        emu_params,
        out_dict,
        return_all_realizations=False,
        Nrealizations=None,
        seed=0,
    ):
        # Predict Arinyo coefficients for the given test conditions
        _ = self.predict_Arinyos(
            emu_params,
            return_all_realizations=return_all_realizations,
            Nrealizations=Nrealizations,
            seed=seed,
        )

        out_dict["coeffs_Arinyo"] = {}
        if return_all_realizations:
            coeffs_all, coeffs_mean = _
            out_dict["coeffs_all_Arinyo"] = {}
        else:
            coeffs_mean = _

        for jj, par in enumerate(self.Arinyo_params):
            out_dict["coeffs_Arinyo"][par] = coeffs_mean[jj]
            if return_all_realizations:
                out_dict["coeffs_all_Arinyo"][par] = coeffs_all[:, jj]
            # old training method
            if (self.training_type == "Arinyo_minz") and (par == "q1"):
                out_dict["coeffs_Arinyo"]["q2"] = out_dict["coeffs_Arinyo"][
                    "q1"
                ]
                if return_all_realizations:
                    out_dict["coeffs_all_Arinyo"]["q2"] = out_dict[
                        "coeffs_all_Arinyo"
                    ]["q1"]

        return out_dict

    def _rescale_cosmo(self, target_params, cosmo, z, kp_Mpc=0.7, ks_Mpc=0.05):
        sim_cosmo = camb_cosmo.get_cosmology(**cosmo)
        linP_zs = fit_linP.get_linP_Mpc_zs(sim_cosmo, [z], kp_Mpc)[0]

        fid_Ap = linP_zs["Delta2_p"]
        ratio_Ap = target_params["Delta2_p"] / fid_Ap

        fid_np = linP_zs["n_p"]
        delta_np = target_params["n_p"] - fid_np

        # logarithm of ratio of pivot points
        ln_kp_ks = np.log(kp_Mpc / ks_Mpc)

        # compute scalings
        delta_ns = delta_np
        ln_ratio_As = np.log(ratio_Ap) - delta_np * ln_kp_ks

        rescaled_cosmo = cosmo.copy()
        rescaled_cosmo["As"] = np.exp(ln_ratio_As) * cosmo["As"]
        rescaled_cosmo["ns"] = delta_ns + cosmo["ns"]

        return rescaled_cosmo

    def _get_p3d(self, info_power, out_dict, model_Arinyo):
        try:
            k_Mpc = info_power["k3d_Mpc"]
            mu = info_power["mu"]
        except:
            msg = "info_power must contain 'k3d_Mpc', 'mu', and (if needed) 'kmu_modes'."
            raise ValueError(msg)

        if (len(k_Mpc.shape) == 2) & (len(mu.shape) == 2):
            if (k_Mpc.shape[0] != mu.shape[0]) or (
                k_Mpc.shape[1] != mu.shape[1]
            ):
                raise ValueError("k and mu must have the same shape.")
            else:
                nd1 = k_Mpc.shape[0]
                nd2 = k_Mpc.shape[1]
                k_Mpc = k_Mpc.reshape(-1)
                mu = mu.reshape(-1)
        elif (len(k_Mpc.shape) == 1) & (len(mu.shape) == 1):
            if k_Mpc.shape[0] != mu.shape[0]:
                raise ValueError("k and mu must have the same shape.")
            else:
                nd1 = k_Mpc.shape[0]
                nd2 = 0
        else:
            raise ValueError("k and mu must be 1D or 2D arrays.")

        out_dict["k_Mpc"] = k_Mpc
        out_dict["mu"] = mu

        if "kmu_modes" in info_power:
            p3d_arinyo, out_dict["Plin"] = p3d_allkmu(
                model_Arinyo,
                info_power["z"],
                out_dict["coeffs_Arinyo"],
                info_power["kmu_modes"],
                nk=nd1,
                nmu=nd2,
                compute_plin=True,
            )
            out_dict["p3d"] = p3d_arinyo.reshape(-1)
        else:
            out_dict["p3d"] = model_Arinyo.P3D_Mpc(
                info_power["z"], k_Mpc, mu, out_dict["coeffs_Arinyo"]
            )
            out_dict["Plin"] = model_Arinyo.linP_Mpc(info_power["z"], k_Mpc)

        if nd2 != 0:
            out_dict["p3d"] = out_dict["p3d"].reshape(nd1, nd2)
            out_dict["k_Mpc"] = out_dict["k_Mpc"].reshape(nd1, nd2)
            out_dict["mu"] = out_dict["mu"].reshape(nd1, nd2)

        return out_dict

    def _get_p3d_cov(self, info_power, out_dict, model_Arinyo):
        try:
            k_Mpc = info_power["k3d_Mpc"]
            mu = info_power["mu"]
        except:
            msg = "info_power must contain 'k3d_Mpc', 'mu', and (if needed) 'kmu_modes'."
            raise ValueError(msg)

        if (len(k_Mpc.shape) == 2) & (len(mu.shape) == 2):
            if (k_Mpc.shape[0] != mu.shape[0]) or (
                k_Mpc.shape[1] != mu.shape[1]
            ):
                raise ValueError("k and mu must have the same shape.")
            else:
                nd1 = k_Mpc.shape[0]
                nd2 = k_Mpc.shape[1]
                k_Mpc = k_Mpc.reshape(-1)
                mu = mu.reshape(-1)
        elif (len(k_Mpc.shape) == 1) & (len(mu.shape) == 1):
            if k_Mpc.shape[0] != mu.shape[0]:
                raise ValueError("k and mu must have the same shape.")
            else:
                nd1 = k_Mpc.shape[0]
                nd2 = 0
        else:
            raise ValueError("k and mu must be 1D or 2D arrays.")

        Nrea = out_dict["coeffs_all_Arinyo"]["q1"].shape[0]
        p3ds_pred = np.zeros((Nrea, len(k_Mpc)))
        for r in range(Nrea):
            input_pars = {}
            for par in self.Arinyo_params:
                input_pars[par] = out_dict["coeffs_all_Arinyo"][par][r]
            if "kmu_modes" in info_power:
                _ = p3d_allkmu(
                    model_Arinyo,
                    info_power["z"],
                    input_pars,
                    info_power["kmu_modes"],
                    nk=nd1,
                    nmu=nd2,
                    compute_plin=False,
                )
                p3ds_pred[r] = _.reshape(-1)
            else:
                p3ds_pred[r] = model_Arinyo.P3D_Mpc(
                    info_power["z"], k_Mpc, mu, input_pars
                )

        p3d_cov = get_covariance(p3ds_pred, out_dict["p3d"])
        diag = np.diag(p3d_cov)
        if nd2 != 0:
            diag = diag.reshape(nd1, nd2)
        out_dict["p3d_std"] = np.sqrt(diag)

        return out_dict

    def _get_p1d(self, info_power, out_dict, model_Arinyo):
        try:
            k1d_Mpc = info_power["k1d_Mpc"]
        except:
            msg = "info_power must contain 'k1d_Mpc'."
            raise ValueError(msg)

        out_dict["p1d"] = model_Arinyo.P1D_Mpc(
            info_power["z"], k1d_Mpc, parameters=out_dict["coeffs_Arinyo"]
        )
        out_dict["k1d_Mpc"] = k1d_Mpc

        return out_dict

    def _get_p1d_cov(self, info_power, out_dict, model_Arinyo):
        try:
            k1d_Mpc = info_power["k1d_Mpc"]
        except:
            msg = "info_power must contain 'k1d_Mpc'."
            raise ValueError(msg)

        Nrea = out_dict["coeffs_all_Arinyo"]["q1"].shape[0]
        p1ds_pred = np.zeros((Nrea, len(k1d_Mpc)))

        for r in range(Nrea):
            input_pars = {}
            for par in self.Arinyo_params:
                input_pars[par] = out_dict["coeffs_all_Arinyo"][par][r]
            p1ds_pred[r] = model_Arinyo.P1D_Mpc(
                info_power["z"], k1d_Mpc, parameters=input_pars
            )

        p1d_cov = get_covariance(p1ds_pred, out_dict["p1d"])
        out_dict["p1d_std"] = np.sqrt(np.diag(p1d_cov))

        return out_dict

    def evaluate(
        self,
        emu_params,
        info_power=None,
        Nrealizations=None,
        return_all_realizations=False,
        verbose=True,
        seed=0,
        kp_Mpc=0.7,
        return_bias_eta=False,
    ):
        """
        Predict the power spectrum using the emulator for a given simulation label and redshift.

        This function predicts the power spectrum using the emulator for a specified simulation label and redshift.
        It utilizes the Arinyo model to generate power spectrum predictions based on the emulator coefficients.

        Args:
            z (float): Redshift to evaluate the model
            emu_params (dict): Dictionary containing emulator parameters
            info_power (dict, optional): Dictionary containing information for computing power spectrum
            return_bias_eta (bool, optional): Return bias_delta. Default is False.
            Nrealizations (int, optional): Number of realizations to generate.

        Returns:
            Dict: Dictionary containing the predicted Arinyo parameters and (if needed) power spectrum
        """

        if Nrealizations is None:
            Nrealizations = self.Nrealizations

        if info_power is not None:
            if "return_cov" in info_power:
                return_all_realizations = True

        # output
        out_dict = {}

        # Check if all emulator parameters are provided
        emu_params = self._check_emu_params(emu_params, info_power)

        # Get Arinyo coefficients
        out_dict = self._get_Arinyo_coeffs(
            emu_params,
            out_dict,
            return_all_realizations=return_all_realizations,
            Nrealizations=Nrealizations,
            seed=seed,
        )

        # Redshift
        if "z" not in info_power:
            raise ValueError("z must be in info_power")

        # In order to compute P3D or P1D, we need to compute Plin first
        # for the target cosmology. Get the cosmology
        if (("cosmo" in info_power) and ("sim_label" in info_power)) | (
            ("cosmo" not in info_power) and ("sim_label" not in info_power)
        ):
            msg = (
                "info_power needs to include either cosmo or sim_label, not both or none."
                + "The cosmology is set by one of these."
            )
            raise ValueError(msg)
        else:
            if "cosmo" in info_power:
                cosmo = info_power["cosmo"]
            elif "sim_label" in info_power:
                repo = os.path.dirname(lace.__path__[0]) + "/"
                fname = repo + ("data/sim_suites/Australia20/mpg_emu_cosmo.npy")
                data_cosmo = np.load(fname, allow_pickle=True).item()
                cosmo = data_cosmo[info_power["sim_label"]]["cosmo_params"]

        if return_bias_eta:
            linP_zs = fit_linP.get_linP_Mpc_zs(
                camb_cosmo.get_cosmology(**cosmo), [info_power["z"]], kp_Mpc
            )[0]
            out_dict["coeffs_Arinyo"]["bias_eta"] = (
                out_dict["coeffs_Arinyo"]["bias"]
                * out_dict["coeffs_Arinyo"]["beta"]
                / linP_zs["f_p"]
            )
            out_dict["coeffs_Arinyo"]["f_p"] = linP_zs["f_p"]

        # Check if enough cosmology parameters are provided
        for key in self.cosmo_fields:
            if key not in cosmo.keys():
                raise ValueError("cosmo must contain:", self.cosmo_fields)

        # rescale cosmology to Delta2_p and n_p in orig_params if present
        if ("Delta2_p" in emu_params.keys()) | ("n_p" in emu_params.keys()):
            cosmo = self._rescale_cosmo(emu_params, cosmo, info_power["z"])

        if ("return_p3d" in info_power) | ("return_p1d" in info_power):
            pk_interp = get_camb_interp("a", {"cosmo_params": cosmo})
            model_Arinyo = ArinyoModel(camb_pk_interp=pk_interp)

        if "return_p3d" in info_power:
            out_dict = self._get_p3d(info_power, out_dict, model_Arinyo)
            if "return_cov" in info_power:
                out_dict = self._get_p3d_cov(info_power, out_dict, model_Arinyo)

        if "return_p1d" in info_power:
            out_dict = self._get_p1d(info_power, out_dict, model_Arinyo)
            if "return_cov" in info_power:
                out_dict = self._get_p1d_cov(info_power, out_dict, model_Arinyo)

        # Clear GPU memory if using PyTorch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return out_dict

    def _define_cINN_Arinyo(self, nLayers_inn, batch_size, dim_inputSpace):
        """
        Define a conditional invertible neural network (cINN) for Arinyo model.

        This function defines the architecture of a conditional invertible neural network (cINN) for the Arinyo model.
        It specifies the structure of the neural network, including the number of layers, dropout, and activation functions.

        Args:
            dim_inputSpace (int): Dimension of the input space. Default is 8.

        Returns:
            Ff.SequenceINN: Conditional invertible neural network for Arinyo model.
        """

        def subnet_fc(dims_in, dims_out):
            return nn.Sequential(
                nn.Linear(dims_in, 64),
                nn.ReLU(),
                nn.Dropout(0),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Dropout(0),
                nn.Linear(128, dims_out),
            )

        self.nLayers_inn = nLayers_inn
        self.batch_size = batch_size
        self.dim_inputSpace = dim_inputSpace

        # Initialize the cINN model
        emulator = Ff.SequenceINN(self.dim_inputSpace)

        # Append AllInOneBlocks to the cINN model based on the specified number of layers
        for l in range(self.nLayers_inn):
            emulator.append(
                Fm.AllInOneBlock,
                cond=[i for i in range(self.batch_size)],
                cond_shape=[6],
                subnet_constructor=subnet_fc,
            )

        return emulator

    def _load_emu(self, model_path):
        """
        Load a pre-trained Arinyo model emulator.
        """

        # load metadata
        metadata = np.load(
            model_path + "_metadata.npy", allow_pickle=True
        ).item()

        self.training_type = metadata["training_type"]
        self.input_param_lims_min = metadata["input_param_lims_min"]
        self.input_param_lims_max = metadata["input_param_lims_max"]
        self.output_param_lims_min = metadata["output_param_lims_min"]
        self.output_param_lims_max = metadata["output_param_lims_max"]
        self.emu_input_names = metadata["emu_input_names"]

        self.emulator = self._define_cINN_Arinyo(
            metadata["nLayers_inn"],
            metadata["batch_size"],
            metadata["dim_inputSpace"],
        )

        # Load a pre-trained model if model_path is provided
        warn("Loading a pre-trained emulator")
        self.emulator.load_state_dict(torch.load(model_path + ".pt"))

    def _train_emu(
        self,
        training_data,
        emu_input_names,
        adamw=True,
        lr=1e-3,
        nepochs=300,
        step_size=200,
        use_chains=False,
        chain_samp=100_000,
        weight_decay=0,
        dim_inputSpace=8,
        nLayers_inn=12,
        batch_size=100,
        training_type="Arinyo_min",
        save_path=None,
        kmax_Mpc=4.0,
        train_seed=32,
    ):
        """
        Train the Arinyo model emulator using conditional invertible neural network (cINN).

        This function trains the Arinyo model emulator by optimizing the cINN parameters.
        It supports loading a pre-trained model if a model_path is provided.

        Returns:
            None
        """

        random.seed(train_seed)
        np.random.seed(train_seed)
        torch.manual_seed(train_seed)
        torch.cuda.manual_seed_all(train_seed)

        # Get the training data and define the cINN model
        emu_input, emu_output = self._get_training_data(
            training_data, emu_input_names, training_type
        )
        self.emulator = self._define_cINN_Arinyo(
            nLayers_inn, batch_size, dim_inputSpace
        )

        # Extract k_Mpc and mu from the training data
        k_Mpc = training_data[0]["k3d_Mpc"]
        mu = training_data[0]["mu3d"]
        # Create a mask for k values within the specified kmax_Mpc range
        k_mask = (k_Mpc < kmax_Mpc) & (k_Mpc > 0)

        # store metadata
        metadata = {
            "nLayers_inn": nLayers_inn,
            "batch_size": batch_size,
            "dim_inputSpace": dim_inputSpace,
            "input_param_lims_min": self.input_param_lims_min,
            "input_param_lims_max": self.input_param_lims_max,
            "output_param_lims_min": self.output_param_lims_min,
            "output_param_lims_max": self.output_param_lims_max,
            "training_type": training_type,
            "kmax_Mpc": kmax_Mpc,
            "lr": lr,
            "nepochs": nepochs,
            "step_size": step_size,
            "use_chains": use_chains,
            "chain_samp": chain_samp,
            "weight_decay": weight_decay,
            "adamw": adamw,
            "emu_input_names": emu_input_names,
            "k_Mpc": k_Mpc,
            "mu": mu,
            "k_mask": k_mask,
            "k_Mpc_masked": k_Mpc[k_mask],
            "mu_masked": mu[k_mask],
            "train_seed": train_seed,
        }
        if save_path is not None:
            np.save(save_path + "_metadata.npy", metadata)

        # Initialize the cINN model with Xavier initialization
        self.emulator.apply(init_xavier)

        # Create a PyTorch dataset and loader for training
        trainig_dataset = TensorDataset(emu_input, emu_output)
        loader = DataLoader(
            trainig_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        # Choose the optimizer (Adam or AdamW)
        if adamw:
            optimizer = torch.optim.AdamW(
                self.emulator.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            optimizer = optim.Adam(
                self.emulator.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size
        )

        # Training loop
        self.loss_arr = []
        t0 = time.time()
        for i in range(nepochs):
            if i % 25 == 0:
                print(f"Epoch {i}/{nepochs}")
            _loss_arr = []
            _latent_space = []

            for cond, coeffs in loader:
                optimizer.zero_grad()

                # Sample from the chains if use_chains is True
                if use_chains:
                    idx = np.random.choice(
                        chain_samp, size=2_000, replace=False
                    )
                    coeffs = coeffs[:, idx, :].mean(axis=1)

                # Forward pass through the cINN
                z, log_jac_det = self.emulator(coeffs, cond)

                # Calculate the negative log-likelihood
                loss = 0.5 * torch.sum(z**2, 1) - log_jac_det
                loss = loss.mean()

                # Backpropagate and update the weights
                loss.backward()
                optimizer.step()

                _loss_arr.append(loss.item())
                _latent_space.append(z)

            scheduler.step()
            self.loss_arr.append(np.mean(_loss_arr))

            # Store latent space for the last epoch
            if i == (nepochs - 1):
                self._latent_space = _latent_space

        print(f"Emulator optimized in {time.time() - t0} seconds")

        # Save the model if save_path is provided
        if save_path is not None:
            torch.save(self.emulator.state_dict(), save_path + ".pt")

    def predict_Arinyos(
        self,
        emu_params,
        Nrealizations=None,
        return_all_realizations=False,
        seed=0,
    ):
        """
        Predict Arinyo coefficients using the trained emulator.

        Args:
            emu_params (list of dict): List of dictionaries containing the
                cosmo + IGM input parameters.
            Nrealizations (int): Number of realizations to generate. Default is None.

            return_all_realizations (bool): Whether to return all realizations
                or just the mean. Default is False.

        Returns:
            Dictionary with mean Arinyo coefficient predictions.
            If return_all_realizations is True, returns a tuple with all realizations and the mean.
        """

        if isinstance(emu_params, dict):
            emu_params = [emu_params]

        if len(emu_params) > 250:
            print(
                "WARNING: More than 500 instances of emu_params will take too much memory. "
                "Please use a smaller number of emu_params at a time. "
                "Returning None"
            )
            return

        # Use default number of realizations set when loading the emulator
        # if not specified
        if Nrealizations is None:
            Nrealizations = self.Nrealizations

        # Set the seed
        g = torch.Generator().manual_seed(seed)

        # Number of combinations of input parameters
        neval = len(emu_params)
        ninpt_pars = len(emu_params[0])

        condition = np.zeros((neval * Nrealizations, ninpt_pars))
        for jj in range(neval):
            # Input to emulator
            input_emu = []
            for par in self.emu_input_names:
                input_emu.append(emu_params[jj][par])
            input_emu = np.array(input_emu)
            # normalize the input data and arrange it along the first axis
            condition[jj * Nrealizations : (jj + 1) * Nrealizations, :] = (
                input_emu - self.input_param_lims_min
            ) / (self.input_param_lims_max - self.input_param_lims_min)
        condition = torch.Tensor(condition)

        # cINN stuff
        aran = np.arange(neval * Nrealizations)
        self.emulator.conditions = []
        for ii in range(self.nLayers_inn):
            self.emulator.conditions.append(aran)

        # Generate predictions
        with torch.no_grad():
            z_test = torch.randn(
                neval * Nrealizations, self.dim_inputSpace, generator=g
            )
            Arinyo_preds, _ = self.emulator(z_test, condition, rev=True)

            # Transform the predictions back to original space
            Arinyo_preds = (
                Arinyo_preds
                * (self.output_param_lims_max - self.output_param_lims_min)
                + self.output_param_lims_min
            )

            # Arinyo_preds[:, 2] = torch.log(Arinyo_preds[:, 2])
            # Arinyo_preds[:, 4] = torch.log(Arinyo_preds[:, 4])
            # Arinyo_preds[:, 6] = torch.exp(Arinyo_preds[:, 6])
            # if "q2" in self.Arinyo_params:
            #     Arinyo_preds[:, 7] = torch.log(Arinyo_preds[:, 7])

        all_realizations = np.array(
            Arinyo_preds.reshape(neval, Nrealizations, self.dim_inputSpace)
        )

        # Calculate the median of the predictions
        Arinyo_mean = np.mean(all_realizations, axis=1)

        if Arinyo_mean.shape[0] == 1:
            Arinyo_mean = Arinyo_mean[0]
            all_realizations = all_realizations[0]

        if return_all_realizations == True:
            return all_realizations, Arinyo_mean
        else:
            return Arinyo_mean
