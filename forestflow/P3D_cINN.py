"""
Conditional Invertible Neural Network (cINN) Emulator for the Arinyo P3D Model.

This module provides a flexible emulator for the Arinyo P3D model using conditional
invertible neural networks. It supports training new emulators from simulation data
or loading pre-trained models for rapid predictions.

The emulator generates Monte Carlo realizations of model parameters by sampling
the latent space and returns mean predictions.
"""

import numpy as np
import os
import time
import random
from warnings import warn
from typing import Optional, Dict, List, Union, Tuple, Any

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import FrEIA.framework as Ff
import FrEIA.modules as Fm

import forestflow
from forestflow.set_training import Transf_data


def init_xavier(m: torch.nn.Module) -> None:
    """
    Initialize neural network weights using Xavier uniform initialization.

    This function applies Xavier initialization to Linear layers, setting weights
    from a uniform distribution with variance based on the number of input units.
    Biases are initialized to a small constant value.

    Parameters
    ----------
    m : torch.nn.Module
        The neural network module to initialize. Only Linear layers are modified.

    Returns
    -------
    None
        The module is modified in-place.

    Examples
    --------
    >>> model = torch.nn.Sequential(torch.nn.Linear(10, 5))
    >>> model.apply(init_xavier)
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class P3DEmulator:
    """
    Conditional invertible neural network (cINN) emulator for the Arinyo P3D model.

    This class provides a flexible interface for training and using a cINN-based
    emulator for the Arinyo P3D model. It supports both training from simulation
    data and loading pre-trained models.

    Attributes
    ----------
    input_labels : List[str]
        Names of input parameters used by the emulator.
    output_labels : List[str]
        Names of output parameters predicted by the emulator.
    emulator : Ff.SequenceINN
        The underlying cINN model.
    transf_data : Transf_data
        Data transformation object for normalization/de-normalization.
    Nrealizations : int
        Default number of latent space realizations for evaluation.
    loss_arr : List[float]
        Training loss history.
    val_loss_arr : List[float]
        Validation loss history (if validation was used during training).
    nLayers_inn : int
        Number of invertible layers in the cINN.
    batch_size : int
        Training batch size.
    dim_inputSpace : int
        Dimension of the input/output space.
    """

    def __init__(
        self,
        key: str = "forest_mpg",
        training_data: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        train: bool = False,
        save_path: Optional[str] = None,
        model_path: Optional[str] = None,
        transf_file: Optional[str] = None,
        nLayers_inn: int = 6,
        dims_int: int = 12,
        nepochs: int = 1000,
        batch_size: int = 8,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        use_val_set: bool = False,
        adamw: bool = True,
        Nrealizations: int = 3000,
    ) -> None:
        """
        Initialize the P3D emulator.

        Parameters
        ----------
        key : str, default="forest_mpg"
            Name identifier for pre-trained emulator models.
        training_data : dict, optional
            Dictionary containing training data with 'input_par' and 'output_par' keys.
            Required when `train=True`.
        train : bool, default=False
            Whether to train a new emulator. If False, loads a pre-trained model.
        save_path : str, optional
            Path prefix for saving trained model and metadata. Required when `train=True`.
        model_path : str, optional
            Path prefix for loading a pre-trained emulator.
        transf_file : str, optional
            File path containing normalization transformations.
        nLayers_inn : int, default=6
            Number of invertible blocks in the cINN.
        dims_int : int, default=12
            Width of hidden layers in each subnet.
        nepochs : int, default=1000
            Number of training epochs.
        batch_size : int, default=8
            Training batch size.
        lr : float, default=1e-3
            Learning rate for the optimizer.
        weight_decay : float, default=1e-4
            Weight decay (L2 regularization) for the optimizer.
        use_val_set : bool, default=False
            Whether to reserve 20% of training data for validation.
        adamw : bool, default=True
            If True use AdamW optimizer, otherwise use Adam.
        Nrealizations : int, default=3000
            Default number of latent space realizations for evaluation.

        Raises
        ------
        ValueError
            If training is requested without required parameters, or if neither
            training nor a model path is provided.
        """

        if train == True:
            key = None
        # Load default emulator configuration if using a pre-defined key
        if key is not None:
            model_path = os.path.join(
                os.path.dirname(forestflow.__path__[0]),
                "data",
                "emulator_models",
                key,
            )
            transf_file = os.path.join(
                os.path.dirname(forestflow.__path__[0]),
                "data",
                "emulator_models",
                key + "_transf.npy",
            )

        # Validate training arguments
        if train and ((save_path is None) or (training_data is None)):
            raise ValueError(
                "When train=True, both save_path and training_data must be provided."
            )
        if train and (model_path is not None):
            raise ValueError(
                "When train=True, model_path must be None. Use save_path instead."
            )

        if train:
            self.input_labels = list(training_data["input_par"].keys())
            self.output_labels = list(training_data["output_par"].keys())
            self._train_emulator(
                training_data,
                adamw=adamw,
                lr=lr,
                nepochs=nepochs,
                train_seed=32,
                dims_int=dims_int,
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
            self._load_emulator(model_path=model_path)
        else:
            raise ValueError(
                "Either train=True with required parameters, or model_path must be provided."
            )

    def _define_cINN_Arinyo(
        self, nLayers_inn: int, batch_size: int, dim_inputSpace: int, dims_int: int = 16
    ) -> Ff.SequenceINN:
        """
        Define the architecture of the conditional invertible neural network.

        This method constructs a cINN with the specified number of invertible blocks,
        each containing a fully-connected subnet with ReLU activations.

        Parameters
        ----------
        nLayers_inn : int
            Number of invertible AllInOneBlocks.
        batch_size : int
            Batch size used for training/evaluation.
        dim_inputSpace : int
            Dimension of the input/output space.
        dims_int : int, default=16
            Width of hidden layers in the subnets.

        Returns
        -------
        Ff.SequenceINN
            The constructed cINN model ready for training or inference.

        Notes
        -----
        The subnet architecture uses two hidden layers with ReLU activations.
        Dropout is currently disabled (rate=0) but can be enabled if needed.
        """

        def subnet_fc(dims_in: int, dims_out: int) -> torch.nn.Sequential:
            """
            Create a fully-connected subnet with two hidden layers.

            Parameters
            ----------
            dims_in : int
                Input dimension of the subnet.
            dims_out : int
                Output dimension of the subnet.

            Returns
            -------
            torch.nn.Sequential
                The subnet module.
            """
            return torch.nn.Sequential(
                torch.nn.Linear(dims_in, dims_int),
                torch.nn.ReLU(),
                torch.nn.Dropout(0),  # Dropout disabled, keep for potential future use
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

        # Append AllInOneBlocks with conditioning
        for _ in range(self.nLayers_inn):
            emulator.append(
                Fm.AllInOneBlock,
                cond=[i for i in range(self.batch_size)],
                cond_shape=[6],
                subnet_constructor=subnet_fc,
            )

        return emulator

    def _load_emulator(self, model_path: str) -> None:
        """
        Load a pre-trained emulator model from disk.

        Parameters
        ----------
        model_path : str
            Path prefix for the saved model and metadata files.
            Expects `model_path.pt` for weights and `model_path_metadata.npy`
            for metadata.

        Notes
        -----
        The metadata file must contain 'input_labels', 'output_labels',
        'nLayers_inn', 'batch_size', 'dim_inputSpace', and 'dims_int' keys.
        """
        # Load metadata
        metadata = np.load(model_path + "_metadata.npy", allow_pickle=True).item()

        self.input_labels = metadata["input_labels"]
        self.output_labels = metadata["output_labels"]

        # Reconstruct the cINN architecture
        self.emulator = self._define_cINN_Arinyo(
            metadata["nLayers_inn"],
            metadata["batch_size"],
            metadata["dim_inputSpace"],
            dims_int=metadata["dims_int"],
        )

        # Load pre-trained weights
        warn("Loading a pre-trained emulator")
        self.emulator.load_state_dict(torch.load(model_path + ".pt"))

    def _train_emulator(
        self,
        training_data: Dict[str, Dict[str, np.ndarray]],
        adamw: bool = True,
        lr: float = 5e-4,
        nepochs: int = 1000,
        weight_decay: float = 1e-4,
        dim_inputSpace: int = 8,
        nLayers_inn: int = 5,
        dims_int: int = 16,
        batch_size: int = 16,
        save_path: Optional[str] = None,
        train_seed: int = 32,
        use_val_set: bool = False,
    ) -> None:
        """
        Train the cINN emulator on the provided dataset.

        This method handles data preparation, model initialization, training loop,
        validation, early stopping, and model saving.

        Parameters
        ----------
        training_data : dict
            Dictionary with 'input_par' and 'output_par' keys, each containing
            parameter arrays.
        adamw : bool, default=True
            If True use AdamW optimizer, otherwise use Adam.
        lr : float, default=5e-4
            Learning rate for the optimizer.
        nepochs : int, default=1000
            Maximum number of training epochs.
        weight_decay : float, default=1e-4
            Weight decay regularization strength.
        dim_inputSpace : int, default=8
            Dimension of the input/output space.
        nLayers_inn : int, default=5
            Number of invertible layers.
        dims_int : int, default=16
            Width of hidden layers in subnets.
        batch_size : int, default=16
            Training batch size.
        save_path : str, optional
            Path prefix for saving the trained model and metadata.
        train_seed : int, default=32
            Random seed for reproducibility.
        use_val_set : bool, default=False
            Whether to use a validation set (20% split).

        Notes
        -----
        The training uses negative log-likelihood loss and implements early stopping
        with a patience of 50 epochs when validation is used.
        """
        # Set random seeds for reproducibility
        random.seed(train_seed)
        np.random.seed(train_seed)
        torch.manual_seed(train_seed)
        torch.cuda.manual_seed_all(train_seed)

        # Convert training data to PyTorch tensors
        emu_input, emu_output = self._prepare_training_data(training_data)

        # Define the cINN architecture
        self.emulator = self._define_cINN_Arinyo(
            nLayers_inn, batch_size, dim_inputSpace, dims_int=dims_int
        )

        # Store metadata
        metadata = {
            "input_labels": self.input_labels,
            "output_labels": self.output_labels,
            "nLayers_inn": nLayers_inn,
            "batch_size": batch_size,
            "dim_inputSpace": dim_inputSpace,
            "dims_int": dims_int,
            "lr": lr,
            "nepochs": nepochs,
            "weight_decay": weight_decay,
            "adamw": adamw,
            "train_seed": train_seed,
        }
        if save_path is not None:
            np.save(save_path + "_metadata.npy", metadata)

        # Apply Xavier initialization
        self.emulator.apply(init_xavier)

        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(
            emu_input, emu_output, batch_size, use_val_set, train_seed
        )

        # Setup optimizer
        optimizer = self._setup_optimizer(adamw, lr, weight_decay)

        # Setup learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=25, threshold=5e-5
        )

        # Training loop
        self.loss_arr = []
        self.val_loss_arr = []
        best_val = np.inf
        patience = 50
        counter = 0

        t0 = time.time()
        for epoch in range(nepochs):
            train_loss = self._train_epoch(optimizer, train_loader)
            self.loss_arr.append(train_loss)

            # Validation and early stopping
            if use_val_set and val_loader is not None:
                val_loss = self._compute_validation_loss(val_loader)
                self.val_loss_arr.append(val_loss)
                scheduler.step(val_loss)

                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    counter = 0
                else:
                    counter += 1

                if counter > patience:
                    print(
                        f"Early stopping at epoch {epoch}, best val loss: {np.round(best_val, 2)}"
                    )
                    break
            else:
                scheduler.step(train_loss)

            # Periodic logging
            if epoch % 25 == 0:
                self._log_training_progress(epoch, nepochs, use_val_set)

        print(f"Emulator optimized in {time.time() - t0:.2f} seconds")

        # Save the trained model
        if save_path is not None:
            torch.save(self.emulator.state_dict(), save_path + ".pt")

    def _prepare_training_data(
        self, training_data: Dict[str, Dict[str, np.ndarray]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert training data from dictionary format to PyTorch tensors.

        Parameters
        ----------
        training_data : dict
            Dictionary with 'input_par' and 'output_par' keys.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Input tensor and output tensor for training.

        Raises
        ------
        ValueError
            If the required keys are missing from training_data.
        """
        emu_input = None
        emu_output = None

        for label in ["input_par", "output_par"]:
            if label not in training_data:
                raise ValueError(f"Missing '{label}' key in training_data")

            param_dict = training_data[label]
            key = list(param_dict.keys())[0]
            nelem = param_dict[key].shape[0]
            npar = len(param_dict)
            arr_data = np.zeros((nelem, npar))

            for ii, par in enumerate(param_dict):
                arr_data[:, ii] = param_dict[par]

            tensor = torch.tensor(arr_data, dtype=torch.float32)
            if label == "input_par":
                emu_input = tensor
            else:
                emu_output = tensor

        return emu_input, emu_output

    def _create_data_loaders(
        self,
        emu_input: torch.Tensor,
        emu_output: torch.Tensor,
        batch_size: int,
        use_val_set: bool,
        seed: int,
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Create training and optional validation data loaders.

        Parameters
        ----------
        emu_input : torch.Tensor
            Input parameter tensor.
        emu_output : torch.Tensor
            Output parameter tensor.
        batch_size : int
            Batch size for data loaders.
        use_val_set : bool
            Whether to create a validation set.
        seed : int
            Random seed for splitting.

        Returns
        -------
        Tuple[DataLoader, Optional[DataLoader]]
            Training data loader and optional validation data loader.
        """
        dataset = TensorDataset(emu_input, emu_output)

        if use_val_set:
            n_val = int(0.2 * len(dataset))
            n_train = len(dataset) - n_val

            train_dataset, val_dataset = random_split(
                dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
            )

            train_loader = DataLoader(
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
            train_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
            )
            val_loader = None

        return train_loader, val_loader

    def _setup_optimizer(
        self, adamw: bool, lr: float, weight_decay: float
    ) -> torch.optim.Optimizer:
        """
        Setup the optimizer for training.

        Parameters
        ----------
        adamw : bool
            If True use AdamW, otherwise use Adam.
        lr : float
            Learning rate.
        weight_decay : float
            Weight decay factor.

        Returns
        -------
        torch.optim.Optimizer
            Configured optimizer.
        """
        if adamw:
            return torch.optim.AdamW(
                self.emulator.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            return torch.optim.Adam(
                self.emulator.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )

    def _train_epoch(
        self, optimizer: torch.optim.Optimizer, loader: DataLoader
    ) -> float:
        """
        Perform one training epoch.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer for updating weights.
        loader : DataLoader
            Training data loader.

        Returns
        -------
        float
            Average loss for this epoch.
        """
        epoch_losses = []

        for cond, coeffs in loader:
            optimizer.zero_grad()

            # Forward pass through the cINN
            z, log_jac_det = self.emulator(coeffs, cond)

            # Calculate negative log-likelihood loss
            loss = 0.5 * torch.sum(z**2, 1) - log_jac_det
            loss = loss.mean()

            # Backpropagation
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        return np.mean(epoch_losses)

    def _compute_validation_loss(self, loader: DataLoader) -> float:
        """
        Compute validation loss.

        Parameters
        ----------
        loader : DataLoader
            Validation data loader.

        Returns
        -------
        float
            Average validation loss.
        """
        self.emulator.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for cond, coeffs in loader:
                z, log_jac_det = self.emulator(coeffs, cond)

                loss = 0.5 * torch.sum(z**2, 1) - log_jac_det
                loss = loss.mean()

                total_loss += loss.item()
                n_batches += 1

        self.emulator.train()
        return total_loss / n_batches

    def _log_training_progress(
        self, epoch: int, nepochs: int, use_val_set: bool
    ) -> None:
        """
        Log training progress information.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        nepochs : int
            Total number of epochs.
        use_val_set : bool
            Whether validation set is being used.
        """
        progress_str = (
            f"Epoch {epoch}/{nepochs}, train loss {np.round(self.loss_arr[-1], 2)}"
        )

        if use_val_set and len(self.val_loss_arr) > 0:
            progress_str += f", val loss {np.round(self.val_loss_arr[-1], 2)}"

            if len(self.val_loss_arr) > 1:
                progress_str += f", best {np.round(np.min(self.val_loss_arr), 2)}"

        print(progress_str)

    def evaluate(
        self,
        emu_params: Union[Dict[str, float], List[Dict[str, float]]],
        Nrealizations: Optional[int] = None,
        seed: int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        Predict Arinyo coefficients using the trained emulator.

        This method generates Monte Carlo realizations from the latent space and
        returns the mean predictions for the given input parameters.

        Parameters
        ----------
        emu_params : dict or list of dict
            Input parameter dictionaries containing cosmo + IGM parameters.
            If a single dict is provided, it's automatically converted to a list.
        Nrealizations : int, optional
            Number of latent space realizations to generate.
            If None, uses the default value from initialization.
        seed : int, default=0
            Random seed for reproducibility.

        Returns
        -------
        dict
            Dictionary containing the predicted output parameters for each input.

        Raises
        ------
        ValueError
            If the emulator hasn't been trained or loaded properly.
        """
        # Check if emulator is initialized
        if not hasattr(self, "emulator"):
            raise ValueError(
                "Emulator not initialized. Please train or load a model first."
            )

        # Convert single input to list
        if isinstance(emu_params, dict):
            emu_params = [emu_params]

        # Warn if too many inputs (memory warning)
        if len(emu_params) > 250:
            warn(
                "More than 250 instances of emu_params may consume significant memory. "
                "Consider processing in smaller batches."
            )

        # Set default number of realizations
        if Nrealizations is None:
            Nrealizations = self.Nrealizations

        # Setup random generator
        generator = torch.Generator().manual_seed(seed)

        # Prepare conditioned inputs
        neval = len(emu_params)
        condition = self._prepare_condition_tensor(emu_params, neval, Nrealizations)

        # Generate predictions
        all_realizations = self._generate_predictions(
            condition, neval, Nrealizations, generator
        )

        # Process and transform predictions
        return self._process_predictions(all_realizations, neval)

    def _prepare_condition_tensor(
        self, emu_params: List[Dict[str, float]], neval: int, Nrealizations: int
    ) -> torch.Tensor:
        """
        Prepare the condition tensor for the cINN.

        Parameters
        ----------
        emu_params : list of dict
            Input parameter dictionaries.
        neval : int
            Number of evaluation points.
        Nrealizations : int
            Number of realizations per point.

        Returns
        -------
        torch.Tensor
            Condition tensor for the cINN.
        """
        ninpt_pars = len(emu_params[0])
        condition = np.zeros((neval * Nrealizations, ninpt_pars))

        for jj in range(neval):
            # Normalize input parameters
            dict_input = self.transf_data.transf_stand(
                emu_params[jj], type_stand="input", direct=True
            )

            # Create array of normalized parameters
            arr_input = np.array([dict_input[par] for par in self.input_labels])
            condition[jj * Nrealizations : (jj + 1) * Nrealizations, :] = arr_input

        return torch.tensor(condition, dtype=torch.float32)

    def _generate_predictions(
        self,
        condition: torch.Tensor,
        neval: int,
        Nrealizations: int,
        generator: torch.Generator,
    ) -> np.ndarray:
        """
        Generate predictions from the cINN model.

        Parameters
        ----------
        condition : torch.Tensor
            Condition tensor for the cINN.
        neval : int
            Number of evaluation points.
        Nrealizations : int
            Number of realizations per point.
        generator : torch.Generator
            Random generator for reproducibility.

        Returns
        -------
        np.ndarray
            Array of all realizations with shape (neval, Nrealizations, dim_inputSpace).
        """
        # Setup conditions for the cINN
        aran = np.arange(neval * Nrealizations)
        self.emulator.conditions = [aran] * self.nLayers_inn

        # Generate predictions
        with torch.no_grad():
            z_test = torch.randn(
                neval * Nrealizations, self.dim_inputSpace, generator=generator
            )
            out_emu, _ = self.emulator(z_test, condition, rev=True)

        return np.array(out_emu.reshape(neval, Nrealizations, self.dim_inputSpace))

    def _process_predictions(
        self, all_realizations: np.ndarray, neval: int
    ) -> Dict[str, np.ndarray]:
        """
        Process and transform predictions back to the original space.

        Parameters
        ----------
        all_realizations : np.ndarray
            Array of all realizations.
        neval : int
            Number of evaluation points.

        Returns
        -------
        dict
            Dictionary of processed predictions.
        """
        # Calculate mean across realizations
        arr_tswn_output = np.mean(all_realizations, axis=1)

        # Convert to dictionary format
        dict_tswn_output = {
            par: arr_tswn_output[:, ii] for ii, par in enumerate(self.output_labels)
        }

        # Transform back to original space
        output = self.transf_data.transf_stand(
            dict_tswn_output, type_stand="output", direct=False
        )

        # Ensure 1D arrays are converted to scalars for single values
        for par in output:
            if output[par].ndim == 1 and output[par].shape[0] == 1:
                output[par] = output[par][0]

        return output


def compute_val_loss(model: Ff.SequenceINN, loader: DataLoader) -> float:
    """
    Compute validation loss for a cINN model.

    This function evaluates the model on a validation dataset and returns the
    average negative log-likelihood loss.

    Parameters
    ----------
    model : Ff.SequenceINN
        The cINN model to evaluate.
    loader : DataLoader
        Validation data loader providing conditioned inputs and outputs.

    Returns
    -------
    float
        Average validation loss.

    Notes
    -----
    The model is temporarily set to evaluation mode during computation and
    restored to training mode after completion.
    """
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
    return total_loss / n_batches if n_batches > 0 else 0.0
