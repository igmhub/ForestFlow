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

from forestflow.set_training import Transf_data


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
        train=False,
        drop_sim=None,
        save_path=None,
        model_path=None,
        transf_file=None,
        nLayers_inn=5,
        dims_int=16,
        nepochs=1000,
        batch_size=16,
        lr=5e-4,
        step_size=200,  # not used
        gamma=0.7,  # not used
        weight_decay=1e-4,
        use_val_set=False,
        adamw=True,
        use_chains=False,
        Nrealizations=3000,
    ):
        if train and ((save_path is None) | (training_data is None)):
            raise ValueError(
                "If train is true, save_path and training_data must be provided."
            )
        if train and (model_path is not None):
            raise ValueError(
                "If train is true, model_path must not be provided. Use save_path instead."
            )

        if train:

            self.input_labels = list(training_data["input_par"].keys())
            self.output_labels = list(training_data["output_par"].keys())

            self._train_emu(
                training_data,
                adamw=adamw,
                lr=lr,
                nepochs=nepochs,
                step_size=step_size,
                use_chains=use_chains,
                train_seed=32,
                gamma=gamma,
                dims_int=dims_int,
                drop_sim=drop_sim,
                weight_decay=weight_decay,
                nLayers_inn=nLayers_inn,
                batch_size=batch_size,
                dim_inputSpace=len(self.output_labels),
                save_path=save_path,
                use_val_set=use_val_set,
            )
        elif model_path is not None:
            self.Nrealizations = Nrealizations
            self.transf_data = Transf_data(preload_file=transf_file)
            self._load_emu(model_path=model_path)
        else:
            raise ValueError("Either train or model_path must be provided.")

    def _define_cINN_Arinyo(self, nLayers_inn, batch_size, dim_inputSpace, dims_int=16):
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
                torch.nn.Linear(dims_in, dims_int),
                torch.nn.ReLU(),
                torch.nn.Dropout(0),
                torch.nn.Linear(dims_int, dims_int * 2),
                torch.nn.ReLU(),
                torch.nn.Dropout(0),
                torch.nn.Linear(dims_int * 2, dims_out),
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
        metadata = np.load(model_path + "_metadata.npy", allow_pickle=True).item()

        self.input_labels = metadata["input_labels"]
        self.output_labels = metadata["output_labels"]

        self.emulator = self._define_cINN_Arinyo(
            metadata["nLayers_inn"],
            metadata["batch_size"],
            metadata["dim_inputSpace"],
            dims_int=metadata["dims_int"],
        )

        # Load a pre-trained model if model_path is provided
        warn("Loading a pre-trained emulator")
        self.emulator.load_state_dict(torch.load(model_path + ".pt"))

    def _train_emu(
        self,
        training_data,
        adamw=True,
        lr=5e-4,
        nepochs=1000,
        step_size=200,
        use_chains=False,
        chain_samp=100_000,
        weight_decay=1e-4,
        dim_inputSpace=8,
        nLayers_inn=5,
        dims_int=16,
        batch_size=16,
        save_path=None,
        train_seed=32,
        gamma=0.7,
        drop_sim=None,
        use_val_set=False,
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

        for label in ["input_par", "output_par"]:

            # move training data to arrays
            key = list(training_data[label].keys())[0]
            nelem = training_data[label][key].shape[0]
            npar = len(training_data[label])
            arr_data = np.zeros((nelem, npar))
            for ii, par in enumerate(training_data[label]):
                arr_data[:, ii] = training_data[label][par]

            # native type for numpy is float64, for torch it is float32
            if label == "input_par":
                emu_input = torch.tensor(arr_data, dtype=torch.float32)
            else:
                emu_output = torch.tensor(arr_data, dtype=torch.float32)

        self.emulator = self._define_cINN_Arinyo(
            nLayers_inn, batch_size, dim_inputSpace, dims_int=dims_int
        )

        # store metadata
        metadata = {
            "input_labels": self.input_labels,
            "output_labels": self.output_labels,
            "nLayers_inn": nLayers_inn,
            "batch_size": batch_size,
            "dim_inputSpace": dim_inputSpace,
            "dims_int": dims_int,
            # "input_param_lims_min": self.input_param_lims_min,
            # "input_param_lims_max": self.input_param_lims_max,
            # "output_param_lims_min": self.output_param_lims_min,
            # "output_param_lims_max": self.output_param_lims_max,
            # "training_type": training_type,
            "lr": lr,
            "nepochs": nepochs,
            # "step_size": step_size,
            "use_chains": use_chains,
            "chain_samp": chain_samp,
            "weight_decay": weight_decay,
            "adamw": adamw,
            # "emu_input_names": emu_input_names,
            "train_seed": train_seed,
        }
        if save_path is not None:
            np.save(save_path + "_metadata.npy", metadata)

        # Initialize the cINN model with Xavier initialization
        self.emulator.apply(init_xavier)

        # Create a PyTorch dataset and loader for training
        trainig_dataset = TensorDataset(emu_input, emu_output)

        if use_val_set:
            n_val = int(0.2 * len(trainig_dataset))
            n_train = len(trainig_dataset) - n_val

            train_dataset, val_dataset = torch.utils.data.random_split(
                trainig_dataset, [n_train, n_val]
            )

            loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=True,
            )
        else:
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

        # early stopping
        best_val = np.inf
        patience = 50
        counter = 0

        # Learning rate scheduler
        # scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer, step_size=step_size, gamma=gamma
        # )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=25,
            threshold=5e-5,
            verbose=True,
        )

        # Training loop
        self.loss_arr = []
        self.val_loss_arr = []
        t0 = time.time()
        for i in range(nepochs):
            _loss_arr = []
            _latent_space = []

            for cond, coeffs in loader:
                optimizer.zero_grad()

                # Sample from the chains if use_chains is True
                # if use_chains:
                #     idx = np.random.choice(chain_samp, size=2_000, replace=False)
                #     coeffs = coeffs[:, idx, :].mean(axis=1)

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

            train_loss = np.mean(_loss_arr)
            self.loss_arr.append(train_loss)

            if use_val_set:
                val_loss = compute_val_loss(self.emulator, val_loader)
                self.val_loss_arr.append(val_loss)

                scheduler.step(val_loss)

                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    counter = 0
                    self._latent_space = _latent_space
                else:
                    counter += 1

                if counter > patience:
                    print(f"Early stopping {np.round(best_val, 2)}")
                    break
            else:
                scheduler.step(train_loss)

            # Store latent space for the last epoch
            if i == (nepochs - 1):
                self._latent_space = _latent_space

            if i % 25 == 0:
                if use_val_set:
                    string2 = f", val loss {np.round(self.val_loss_arr[-1],2)}, best {np.round(best_val,2)}"
                else:
                    string2 = ""
                print(
                    f"Epoch {i}/{nepochs}, train loss {np.round(self.loss_arr[-1],2)}"
                    + string2
                )

        print(f"Emulator optimized in {time.time() - t0} seconds")

        # Save the model if save_path is provided
        if save_path is not None:
            torch.save(self.emulator.state_dict(), save_path + ".pt")

    def evaluate(
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
            dict_input = self.transf_data.transf_stand(
                emu_params[jj], type_stand="input", direct=True
            )
            arr_input = np.zeros((ninpt_pars))
            for ii, par in enumerate(self.input_labels):
                arr_input[ii] = dict_input[par]
            condition[jj * Nrealizations : (jj + 1) * Nrealizations, :] = arr_input
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

            out_emu, _ = self.emulator(z_test, condition, rev=True)

        # Reshape the predictions and calculate the mean
        all_realizations = np.array(
            out_emu.reshape(neval, Nrealizations, self.dim_inputSpace)
        )
        arr_tswn_output = np.mean(all_realizations, axis=1)

        dict_tswn_output = {}
        for ii, par in enumerate(self.output_labels):
            dict_tswn_output[par] = arr_tswn_output[:, ii]

        # Transform the predictions back to the original space
        output = self.transf_data.transf_stand(
            dict_tswn_output, type_stand="output", direct=False
        )
        # output = self.transf_data.transf_stand_white_norm(
        #     dict_tswn_output, type_stand="output", direct=False
        # )

        for par in output:
            if len(output[par].shape) == 1:
                output[par] = output[par][0]

        return output


def compute_val_loss(model, loader):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for cond, coeffs in loader:

            z, log_jac_det = model(coeffs, cond)

            loss = 0.5 * torch.sum(z**2, 1) - log_jac_det
            loss = loss.mean()

            total_loss += loss.item()
            n_batches += 1

    model.train()
    return total_loss / n_batches
