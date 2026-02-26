import numpy as np
import os
import time
import random
from warnings import warn

# torch modules
import torch
from torch.utils.data import DataLoader, TensorDataset

# FrEIA imports
import FrEIA.framework as Ff
import FrEIA.modules as Fm


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
        if train and ((save_path is None) | (training_data is None)):
            raise ValueError(
                "If train is true, save_path and training_data must be provided."
            )
        if train and (model_path is not None):
            raise ValueError(
                "If train is true, model_path must not be provided. Use save_path instead."
            )

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

        for ipar in [0]:
            input_emu[:, ipar] = np.log(input_emu[:, ipar])

        # Scale the training data based on the parameter limits
        input_emu = (input_emu - self.input_param_lims_min) / (
            self.input_param_lims_max - self.input_param_lims_min
        )

        for ipar in [0, 2, 3, 7]:
            output_emu[:, ipar] = np.log(output_emu[:, ipar])

        # some special transformations applied to the output data
        output_emu = (output_emu - self.output_param_lims_min) / (
            self.output_param_lims_max - self.output_param_lims_min
        )

        # Convert the scaled training data to a torch.Tensor object
        input_emu = torch.Tensor(input_emu)
        output_emu = torch.Tensor(output_emu)

        return input_emu, output_emu

    # def _rescale_cosmo(self, target_params, cosmo, z, kp_Mpc=0.7, ks_Mpc=0.05):
    #     sim_cosmo = camb_cosmo.get_cosmology(**cosmo)
    #     linP_zs = fit_linP.get_linP_Mpc_zs(sim_cosmo, [z], kp_Mpc)[0]

    #     fid_Ap = linP_zs["Delta2_p"]
    #     ratio_Ap = target_params["Delta2_p"] / fid_Ap

    #     fid_np = linP_zs["n_p"]
    #     delta_np = target_params["n_p"] - fid_np

    #     # logarithm of ratio of pivot points
    #     ln_kp_ks = np.log(kp_Mpc / ks_Mpc)

    #     # compute scalings
    #     delta_ns = delta_np
    #     ln_ratio_As = np.log(ratio_Ap) - delta_np * ln_kp_ks

    #     rescaled_cosmo = cosmo.copy()
    #     rescaled_cosmo["As"] = np.exp(ln_ratio_As) * cosmo["As"]
    #     rescaled_cosmo["ns"] = delta_ns + cosmo["ns"]

    #     return rescaled_cosmo

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
            return torch.nn.Sequential(
                torch.nn.Linear(dims_in, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0),
                torch.nn.Linear(64, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0),
                torch.nn.Linear(128, dims_out),
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
            "lr": lr,
            "nepochs": nepochs,
            "step_size": step_size,
            "use_chains": use_chains,
            "chain_samp": chain_samp,
            "weight_decay": weight_decay,
            "adamw": adamw,
            "emu_input_names": emu_input_names,
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
            optimizer = torch.optim.Adam(
                self.emulator.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=0.7
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
        return_dict=True,
    ):
        """
        Predict Arinyo coefficients using the trained emulator.

        Args:
            emu_params (list of dict): List of dictionaries containing the
                cosmo + IGM input parameters.
            Nrealizations (int): Number of realizations to generate. Default is None.
            return_all_realizations (bool): Whether to return all realizations
                or just the mean. Default is False.
            seed (int): Seed for the random number generator. Default is 0.
            return_dict (bool): Whether to return the mean Arinyo coefficients
                as a dictionary or as a numpy array. Default is True.

        Returns:
            dict or numpy.ndarray: Depending on the value of `return_dict`,
                this function returns either a dictionary with the mean Arinyo
                coefficient predictions or a numpy array with all realizations
                and the mean.
        """

        # Check if emu_params is a single dictionary and convert it to a list
        if isinstance(emu_params, dict):
            emu_params = [emu_params]

        # Warn the user if the number of emu_params is too large
        if len(emu_params) > 250:
            print(
                "WARNING: More than 500 instances of emu_params will take too much memory. "
                "Please use a smaller number of emu_params at a time. "
                "Returning None"
            )
            return

        # Use the default number of realizations if not specified
        if Nrealizations is None:
            Nrealizations = self.Nrealizations

        # Set the random seed
        g = torch.Generator().manual_seed(seed)

        # Calculate the number of combinations of input parameters and the number of input parameters
        neval = len(emu_params)
        ninpt_pars = len(emu_params[0])

        # Normalize the input data and arrange it along the first axis
        condition = np.zeros((neval * Nrealizations, ninpt_pars))
        for jj in range(neval):
            input_emu = []
            for par in self.emu_input_names:
                input_emu.append(emu_params[jj][par])
            input_emu = np.array(input_emu)
            for ipar in [0]:
                if input_emu.ndim == 1:
                    input_emu[ipar] = np.log(input_emu[ipar])
                else:
                    input_emu[:, ipar] = np.log(input_emu[:, ipar])
            condition[jj * Nrealizations : (jj + 1) * Nrealizations, :] = (
                input_emu - self.input_param_lims_min
            ) / (self.input_param_lims_max - self.input_param_lims_min)
        condition = torch.Tensor(condition)

        # Prepare the conditions for the cINN
        aran = np.arange(neval * Nrealizations)
        self.emulator.conditions = []
        for ii in range(self.nLayers_inn):
            self.emulator.conditions.append(aran)

        # Generate the Arinyo coefficient predictions
        with torch.no_grad():
            z_test = torch.randn(
                neval * Nrealizations, self.dim_inputSpace, generator=g
            )
            Arinyo_preds, _ = self.emulator(z_test, condition, rev=True)

            # Transform the predictions back to the original space
            Arinyo_preds = (
                Arinyo_preds
                * (self.output_param_lims_max - self.output_param_lims_min)
                + self.output_param_lims_min
            )
            for ipar in [0, 2, 3, 7]:
                Arinyo_preds[:, ipar] = torch.exp(Arinyo_preds[:, ipar])

        # Reshape the predictions and calculate the mean
        all_realizations = np.array(
            Arinyo_preds.reshape(neval, Nrealizations, self.dim_inputSpace)
        )
        Arinyo_mean = np.mean(all_realizations, axis=1)

        # Format the output as a dictionary or a numpy array
        if return_dict:
            _Arinyo_mean = []
            for ii in range(Arinyo_mean.shape[0]):
                _dict_int = {}
                for jj, par in enumerate(self.Arinyo_params):
                    _dict_int[par] = Arinyo_mean[ii, jj]
                _Arinyo_mean.append(_dict_int)
            Arinyo_mean = _Arinyo_mean
            if len(Arinyo_mean) == 1:
                Arinyo_mean = Arinyo_mean[0]
        else:
            if Arinyo_mean.shape[0] == 1:
                Arinyo_mean = Arinyo_mean[0]
                all_realizations = all_realizations[0]

        # Return the results
        if return_all_realizations == True:
            return all_realizations, Arinyo_mean
        else:
            return Arinyo_mean
