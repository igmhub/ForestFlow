# torch modules
import torch
from torch.utils.data import DataLoader, dataset, TensorDataset
from torch import nn, optim

# FrEIA imports
import FrEIA.framework as Ff
import FrEIA.modules as Fm

# lace models
from lace.cosmo import camb_cosmo, fit_linP

# forestflow models
import forestflow
from forestflow.model_p3d_arinyo import ArinyoModel
from forestflow.archive import GadgetArchive3D, get_camb_interp
from forestflow.likelihood import Likelihood
from forestflow.utils import (
    get_covariance,
    sort_dict,
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

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"


def init_xavier(m):
    """Initialization of the NN.
    This is quite important for a faster training
    """
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


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
        gamma (float): Learning rate decay factor. Default is 0.1.
        amsgrad (bool): Whether to use AMSGrad variant of Adam optimizer. Default is False.
        step_size (int): Step size for learning rate scheduler. Default is 75.
        adamw (bool): Whether to use the AdamW optimizer. Default is False.
        Nsim (int): Number of simulations to use for training. Default is 30.
        train (bool): Whether to train the emulator. Default is True.
        drop_sim (float): Simulation to drop during training. Default is None.
        drop_z (float): Drop all snapshots at redshift z from training. Default is None.
        pick_z (float): Pick only snapshots at redshift z. Default is None.
        save_path (str): Path to save the trained model. Default is None.
        model_path (str): Path to a pretrained model. Default is None.
        drop_rescalings (bool): Whether to drop the optical-depth rescalings. Default is False.
        Archive: Archive3D object
        chain_samp (int): Chain sampling size. Default is 100000.
        Nrealizations (int): Number of realizations. Default is 100.
    """

    def __init__(
        self,
        training_data,
        paramList,
        kmax_Mpc=4,
        nLayers_inn=8,
        nepochs=100,
        batch_size=100,
        lr=1e-3,
        weight_decay=1e-4,
        gamma=0.1,
        amsgrad=False,
        step_size=75,
        adamw=False,
        Nsim=30,
        train=True,
        drop_sim=None,
        drop_z=None,
        pick_z=None,
        save_path=None,
        model_path=None,
        drop_rescalings=False,
        Archive=None,
        use_chains=False,
        chain_samp=100_000,
        Nrealizations=100,
        training_type="Arinyo_min_q1",
    ):
        # Initialize class attributes with provided arguments
        self.training_data = training_data
        self.emuparams = paramList
        self.kmax_Mpc = kmax_Mpc
        self.nepochs = nepochs
        self.step_size = step_size
        self.drop_sim = drop_sim
        self.drop_z = drop_z
        self.pick_z = pick_z
        self.adamw = adamw
        self.drop_rescalings = drop_rescalings
        self.nLayers_inn = nLayers_inn
        self.train = train
        self.Nrealizations = Nrealizations
        self.use_chains = use_chains
        self.training_type = training_type

        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.archive = Archive
        self.chain_samp = chain_samp

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
        self.dim_inputSpace = len(self.Arinyo_params)

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
        self.folder_interp = (
            os.path.dirname(forestflow.__path__[0]) + "/data/plin_interp/"
        )
        self.model_path = model_path
        self.save_path = save_path

        # Set random seeds for reproducibility
        torch.manual_seed(32)
        np.random.seed(32)
        random.seed(32)

        # Extract k_Mpc and mu from the training data
        k_Mpc = self.archive.training_data[0]["k3d_Mpc"]
        mu = self.archive.training_data[0]["mu3d"]

        # Create a mask for k values within the specified kmax_Mpc range
        k_mask = (k_Mpc < self.kmax_Mpc) & (k_Mpc > 0)

        self.k_mask = k_mask

        self.k_Mpc_masked = k_Mpc[self.k_mask]
        self.mu_masked = mu[self.k_mask]

        self.pk_fid = self.archive.pk_fid
        self.pk_fid_p1d = self.archive.pk_fid_p1d

        self._train_Arinyo()

    def _get_training_data(self):
        """
        Retrieve and preprocess training data for the emulator.

        This function obtains the training data from the provided archive based on self.emuparams.
        It sorts the training data according to self.emuparams and scales the data based on self.paramLims.
        The scaled training data is returned as a torch.Tensor object.

        Returns:
            torch.Tensor: Preprocessed training data.
        """
        # Extract relevant parameters from the training data
        training_data = [
            {
                key: value
                for key, value in self.training_data[i].items()
                if key in self.emuparams
            }
            for i in range(len(self.training_data))
        ]

        # Sort the training data based on self.emuparams
        training_data = sort_dict(training_data, self.emuparams)

        # Convert the sorted training data to a list of lists
        training_data = [
            list(training_data[i].values())
            for i in range(len(self.training_data))
        ]

        # Convert the training data to a numpy array
        training_data = np.array(training_data)

        # Calculate and store the maximum and minimum values for parameter scaling
        self.param_lims_max = training_data.max(0)
        self.param_lims_min = training_data.min(0)

        # Scale the training data based on the parameter limits
        training_data = (training_data - self.param_lims_max) / (
            self.param_lims_max - self.param_lims_min
        )

        # Convert the scaled training data to a torch.Tensor object
        training_data = torch.Tensor(training_data)

        return training_data

    def _get_test_condition(self, emu_params):
        """
        Extract and sort test condition from given emu_params.

        This function takes a list of dictionaries or dictionary (emu_params) containing emulator parameters
        and extracts the relevant parameters specified by self.archive.emu_params. It then sorts
        the extracted parameters based on the emulator parameters and returns them as a numpy array.

        Args:
            test_data (list): List of dictionaries containing emulator parameters for test data.

        Returns:
            np.array: Test conditions sorted based on emulator parameters.
        """
        # Extract emulator parameters from the test data
        condition = [
            {
                key: value
                for key, value in emu_params.items()
                if key in self.archive.emu_params
            }
            for i in range(len(emu_params))
        ]

        # Sort the conditions based on emulator parameters
        condition = sort_dict(condition, self.archive.emu_params)

        # Convert the sorted conditions to a list of values
        condition = [list(condition[i].values()) for i in range(len(condition))]

        # Convert the list of conditions to a numpy array
        condition = np.array(condition)

        return condition

    def _get_Arinyo_params(self):
        """
        Extract and preprocess Arinyo parameters from the training data.

        This function retrieves Arinyo parameters from the training data and applies necessary transformations.
        It sorts the parameters and returns them as a torch.Tensor object.

        Returns:
            torch.Tensor: Preprocessed Arinyo parameters.
        """
        # Extract relevant Arinyo parameters from the training data
        training_label = [
            {
                key: value
                for key, value in self.training_data[i][
                    self.training_type
                ].items()
                if key in self.Arinyo_params
            }
            for i in range(len(self.training_data))
        ]

        # Sort the Arinyo parameters based on self.Arinyo_params
        training_label = sort_dict(training_label, self.Arinyo_params)

        # Convert the sorted Arinyo parameters to a list of lists
        training_label = [
            list(training_label[i].values())
            for i in range(len(self.training_data))
        ]

        # Convert the Arinyo parameters to a numpy array
        training_label = np.array(training_label)

        # Apply specific transformations to certain parameters
        training_label[:, 2] = np.exp(training_label[:, 2])
        training_label[:, 4] = np.exp(training_label[:, 4])
        training_label[:, 6] = np.log(training_label[:, 6])
        if "q2" in self.Arinyo_params:
            training_label[:, 7] = np.exp(training_label[:, 7])

        # Convert the preprocessed Arinyo parameters to a torch.Tensor object
        training_label = torch.Tensor(training_label)

        return training_label

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

    def evaluate(
        self,
        emu_params,
        info_power=None,
        natural_params=False,
        Nrealizations=None,
        verbose=True,
    ):
        """
        Predict the power spectrum using the emulator for a given simulation label and redshift.

        This function predicts the power spectrum using the emulator for a specified simulation label and redshift.
        It utilizes the Arinyo model to generate power spectrum predictions based on the emulator coefficients.

        Args:
            z (float): Redshift to evaluate the model
            emu_params (dict): Dictionary containing emulator parameters
            info_power (dict, optional): Dictionary containing information for computing power spectrum
            natural_params (bool, optional): Whether to return bias_delta and bias_eta instead of bias and beta. Default is False.
            Nrealizations (int, optional): Number of realizations to generate.

        Returns:
            Dict: Dictionary containing the predicted Arinyo parameters and (if needed) power spectrum
        """

        return_p1d = False
        return_p3d = False
        if Nrealizations is None:
            Nrealizations = self.Nrealizations

        # output
        out_dict = {}

        # check p3d info
        if info_power is not None:
            # check cosmology
            # cosmo (dict, optional): Dictionary containing cosmology
            # sim_label (str, optional): Label of simulation from which we extract the cosmology
            if (("cosmo" in info_power) and ("sim_label" in info_power)) | (
                ("cosmo" not in info_power) and ("sim_label" not in info_power)
            ):
                msg = (
                    "Either cosmo or sim_label must be in info_power"
                    + "The cosmology is set by one of these."
                )
                raise ValueError(msg)

            if "cosmo" in info_power:
                cosmo = info_power["cosmo"]
            else:
                cosmo = None

            if "sim_label" in info_power:
                sim_label = info_power["sim_label"]
            else:
                sim_label = None

            if "return_cov" in info_power:
                return_cov = info_power["return_cov"]
            else:
                return_cov = False

            # Redshift
            if "z" in info_power:
                z = info_power["z"]
            else:
                msg = "z must be in info_power"
                raise ValueError(msg)

            # scale at which amplitude and slope of Plin is computed, in Mpc
            if "kp_Mpc" in info_power:
                kp_Mpc = info_power["kp_Mpc"]
            else:
                kp_Mpc = 0.7

            if "return_p3d" in info_power:
                return_p3d = info_power["return_p3d"]

            if "return_p1d" in info_power:
                return_p1d = info_power["return_p1d"]

            if return_p3d:
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

            if return_p1d:
                try:
                    k1d_Mpc = info_power["k1d_Mpc"]
                except:
                    msg = "info_power must contain 'k1d_Mpc'."
                    raise ValueError(msg)

            # set cosmology
            if sim_label is not None:
                # Load the pre-interpolated matter power spectrum
                # Extract simulation index from the given simulation label
                underscore_index = sim_label.find("_")
                s = sim_label[underscore_index + 1 :]
                flag = f"Plin_interp_sim{s}.npy"

                # Load pre-interpolated matter power spectrum for the specified simulation
                file_plin_inter = self.folder_interp + flag
                pk_interp = np.load(file_plin_inter, allow_pickle=True).all()
            else:
                # Compute the matter power spectrum using CAMB
                for key in self.cosmo_fields:
                    if key not in cosmo.keys():
                        raise ValueError(
                            "cosmo must contain:", self.cosmo_fields
                        )
                pk_interp = get_camb_interp("a", {"cosmo_params": cosmo})
                # Adjusting linP values to this cosmo
                sim_cosmo = camb_cosmo.get_cosmology(**cosmo)
                linP_zs = fit_linP.get_linP_Mpc_zs(sim_cosmo, [z], kp_Mpc)[0]
                out_dict["linP_zs"] = linP_zs
                emu_params["Delta2_p"] = linP_zs["Delta2_p"]
                emu_params["n_p"] = linP_zs["n_p"]
                emu_params["alpha_p"] = linP_zs["alpha_p"]
                if natural_params:
                    emu_params["f_p"] = linP_zs["f_p"]

        # Predict Arinyo coefficients for the given test conditions
        coeffs_all, coeffs_mean = self.predict_Arinyos(
            emu_params,
            return_all_realizations=True,
            Nrealizations=Nrealizations,
        )

        arinyo_pred = params_numpy2dict(coeffs_mean)
        if self.training_type == "Arinyo_minz":
            arinyo_pred["q2"] = arinyo_pred["q1"]

        coeff_dict = []
        for ii in range(coeffs_all.shape[0]):
            _par = params_numpy2dict(coeffs_all[ii])
            if self.training_type == "Arinyo_minz":
                _par["q2"] = _par["q1"]
            coeff_dict.append(_par)

        if natural_params:
            arinyo_pred = transform_arinyo_params(
                arinyo_pred, emu_params["f_p"]
            )

            coeffs_all_natural = np.zeros_like(coeffs_all)
            for ii in range(coeffs_all.shape[0]):
                _par = transform_arinyo_params(
                    params_numpy2dict(coeffs_all[ii]), emu_params["f_p"]
                )
                coeffs_all_natural[ii] = np.array(list(_par.values()))
            std_coeffs = np.std(coeffs_all_natural, axis=0)
        else:
            std_coeffs = np.std(coeffs_all, axis=0)

        arinyo_pred_std = {}
        for ii, key in enumerate(arinyo_pred.keys()):
            if (self.training_type == "Arinyo_minz") and (key == "q2"):
                arinyo_pred_std["q2"] = arinyo_pred_std["q1"]
            else:
                arinyo_pred_std[key] = std_coeffs[ii]

        out_dict["coeffs_Arinyo"] = arinyo_pred
        out_dict["coeffs_Arinyo_std"] = arinyo_pred_std

        if return_p3d | return_p1d:
            # Initialize Arinyo model with the loaded matter power spectrum
            model_Arinyo = ArinyoModel(camb_pk_interp=pk_interp)

            if return_p3d:
                if "kmu_modes" in info_power:
                    _, out_dict["Plin"] = p3d_allkmu(
                        model_Arinyo,
                        info_power["z"],
                        coeff_dict[0],
                        info_power["kmu_modes"],
                        nk=nd1,
                        nmu=nd2,
                        compute_plin=True,
                    )
                else:
                    out_dict["Plin"] = model_Arinyo.linP_Mpc(
                        info_power["z"], k_Mpc
                    )

            # Predict power spectrum using Arinyo model with predicted coefficients
            # Predict multiple realizations and calculate the covariance matrix
            if return_p3d:
                p3ds_pred = np.zeros(shape=(Nrealizations, len(k_Mpc)))
            if return_p1d:
                p1ds_pred = np.zeros(shape=(Nrealizations, len(k1d_Mpc)))
            for r in range(len(coeff_dict)):
                if return_p3d:
                    if "kmu_modes" in info_power:
                        _ = p3d_allkmu(
                            model_Arinyo,
                            info_power["z"],
                            coeff_dict[r],
                            info_power["kmu_modes"],
                            nk=nd1,
                            nmu=nd2,
                            compute_plin=False,
                        )
                        p3ds_pred[r] = _.reshape(-1)
                    else:
                        p3ds_pred[r] = model_Arinyo.P3D_Mpc(
                            info_power["z"], k_Mpc, mu, coeff_dict[r]
                        )
                if return_p1d:
                    p1ds_pred[r] = model_Arinyo.P1D_Mpc(
                        info_power["z"], k1d_Mpc, parameters=coeff_dict[r]
                    )

            if return_p3d:
                # Set the median power spectrum and its covariance matrix
                p3d_arinyo = np.nanmedian(p3ds_pred, 0)

                if return_cov:
                    p3d_cov = get_covariance(p3ds_pred, p3d_arinyo)
                    diag = np.diag(p3d_cov)
                    if nd2 != 0:
                        diag = diag.reshape(nd1, nd2)
                    out_dict["p3d_std"] = np.sqrt(diag)

                if nd2 != 0:
                    p3d_arinyo = p3d_arinyo.reshape(nd1, nd2)
                    k_Mpc = k_Mpc.reshape(nd1, nd2)
                    mu = mu.reshape(nd1, nd2)
                out_dict["p3d"] = p3d_arinyo
                out_dict["k_Mpc"] = k_Mpc
                out_dict["mu"] = mu

            if return_p1d:
                p1d_arinyo = np.nanmedian(p1ds_pred, 0)
                out_dict["p1d"] = p1d_arinyo
                out_dict["k1d_Mpc"] = k1d_Mpc
                if return_cov:
                    p1d_cov = get_covariance(p1ds_pred, p1d_arinyo)
                    out_dict["p1d_std"] = np.sqrt(np.diag(p1d_cov))

        return out_dict

    def _define_cINN_Arinyo(self):
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

    def _train_Arinyo(self):
        """
        Train the Arinyo model emulator using conditional invertible neural network (cINN).

        This function trains the Arinyo model emulator by optimizing the cINN parameters.
        It supports loading a pre-trained model if a model_path is provided.

        Returns:
            None
        """
        # Get the training data and define the cINN model
        training_data = self._get_training_data()
        self.emulator = self._define_cINN_Arinyo()

        # Load a pre-trained model if model_path is provided
        if self.model_path != None:
            warn("Loading a pre-trained emulator")
            self.emulator.load_state_dict(torch.load(self.model_path))
            return

        # Initialize the cINN model with Xavier initialization
        self.emulator.apply(init_xavier)

        # Get Arinyo coefficients for training
        Arinyo_coeffs = self._get_Arinyo_params()

        # Create a PyTorch dataset and loader for training
        trainig_dataset = TensorDataset(training_data, Arinyo_coeffs)
        loader = DataLoader(
            trainig_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

        # Choose the optimizer (Adam or AdamW)
        if self.adamw:
            optimizer = torch.optim.AdamW(
                self.emulator.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = optim.Adam(
                self.emulator.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size
        )

        # Training loop
        self.loss_arr = []
        t0 = time.time()
        for i in range(self.nepochs):
            _loss_arr = []
            _latent_space = []

            for cond, coeffs in loader:
                optimizer.zero_grad()

                # Sample from the chains if use_chains is True
                if self.use_chains == True:
                    idx = np.random.choice(
                        self.chain_samp, size=2_000, replace=False
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
            if i == (self.nepochs - 1):
                self._latent_space = _latent_space

        print(f"Emulator optimized in {time.time() - t0} seconds")

        # Save the model if save_path is provided
        if self.save_path != None:
            torch.save(self.emulator.state_dict(), self.save_path)

    def predict_Arinyos(
        self,
        emu_params,
        plot=False,
        true_coeffs=None,
        Nrealizations=None,
        return_all_realizations=False,
    ):
        """
        Predict Arinyo coefficients using the trained emulator.

        Args:
            input_emu (list): List of cosmo+astro input parameters.
            plot (bool): Whether to generate a corner plot. Default is False.
            true_coeffs (list): True Arinyo coefficients for plotting comparison. Default is None.
            return_all_realizations (bool): Whether to return all realizations or just the mean. Default is False.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: Arinyo coefficient predictions. If return_all_realizations is True, returns a tuple with all realizations and the mean.
        """

        if Nrealizations is None:
            Nrealizations = self.Nrealizations

        self.emulator = self.emulator.eval()

        # Extract and sort emulator parameters from the test data
        input_emu = self._get_test_condition(emu_params)

        # Normalize the input data
        test_data = np.array(input_emu)
        test_data = (test_data - self.param_lims_max) / (
            self.param_lims_max - self.param_lims_min
        )
        test_data = torch.Tensor(test_data)

        # Number of iterations for batch processing
        Niter = int(Nrealizations / self.batch_size)

        # Initialize array for Arinyo predictions
        Arinyo_preds = np.zeros(
            shape=(Niter, self.batch_size, self.dim_inputSpace)
        )
        condition = torch.tile(test_data, (self.batch_size, 1))

        # Generate predictions
        for ii in range(Niter):
            z_test = torch.randn(self.batch_size, self.dim_inputSpace)
            Arinyo_pred, _ = self.emulator(z_test, condition, rev=True)

            # Transform the predictions back to original space
            Arinyo_pred[:, 2] = torch.log(Arinyo_pred[:, 2])
            Arinyo_pred[:, 4] = torch.log(Arinyo_pred[:, 4])
            Arinyo_pred[:, 6] = torch.exp(Arinyo_pred[:, 6])
            if "q2" in self.Arinyo_params:
                Arinyo_pred[:, 7] = torch.log(Arinyo_pred[:, 7])

            Arinyo_preds[ii, :] = Arinyo_pred.detach().cpu().numpy()

        Arinyo_preds = Arinyo_preds.reshape(
            Niter * int(self.batch_size), self.dim_inputSpace
        )

        # Generate corner plot if plot is True
        if plot == True:
            if true_coeffs is None:
                corner_plot = corner.corner(
                    Arinyo_preds,
                    labels=[
                        r"$b$",
                        r"$\beta$",
                        "$q_1$",
                        "$k_{vav}$",
                        "$a_v$",
                        "$b_v$",
                        "$k_p$",
                        "$q_2$",
                    ],
                    truth_color="crimson",
                )
            else:
                corner_plot = corner.corner(
                    Arinyo_preds,
                    labels=[
                        r"$b$",
                        r"$\beta$",
                        "$q_1$",
                        "$k_{vav}$",
                        "$a_v$",
                        "$b_v$",
                        "$k_p$",
                        "$q_2$",
                    ],
                    truths=true_coeffs,
                    truth_color="crimson",
                )

            # Increase the label font size for this plot
            for ax in corner_plot.get_axes():
                ax.xaxis.label.set_fontsize(16)
                ax.yaxis.label.set_fontsize(16)
                ax.xaxis.set_tick_params(labelsize=12)
                ax.yaxis.set_tick_params(labelsize=12)
            plt.show()

        # Calculate the mean of the predictions
        Arinyo_mean = np.median(Arinyo_preds, 0)

        if return_all_realizations == True:
            return Arinyo_preds, Arinyo_mean
        else:
            return Arinyo_mean

    def get_p1d_sim(self, dict_sim):
        like = Likelihood(
            dict_sim[0], self.archive.rel_err_p3d, self.archive.rel_err_p1d
        )
        k1d_mask = like.like.ind_fit1d.copy()
        p1d_sim = like.like.data["p1d"][k1d_mask]
        p1d_k = dict_sim[0]["k_Mpc"][k1d_mask]

        return p1d_sim, p1d_k
