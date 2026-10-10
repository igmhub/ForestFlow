"""
Conditional Invertible Neural Network (cINN) Emulator for the Arinyo P3D Model.

This module provides a flexible emulator for the Arinyo P3D model using conditional
invertible neural networks. It supports training new emulators from simulation data
or loading pre-trained models for rapid predictions.

The emulator generates Monte Carlo realizations of model parameters by sampling
the latent space and returns mean predictions.
"""

import copy
import os
import random
import time
from typing import Optional, Dict, List, Union, Tuple, Any, Mapping, Sequence
from warnings import warn

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

import forestflow
from forestflow.emulator.bundle import load_manifest, write_manifest
from forestflow.emulator.training import Transf_data

from .network import _training_data_fingerprint, init_xavier

# Scientific domain metadata belongs to an emulator bundle, not to an inference
# client. Bundled metadata predates this field, so these documented defaults
# provide a migration path until the bundles are republished.
_LEGACY_MODEL_DOMAINS = {
    "forest_mpg": {
        "kp_iMpc": 0.7,
        "kmax_3d_iMpc": 4.0,
        "kmax_1d_iMpc": 4.0,
        "zmax": 4.1,
        "list_sim_cube": [f"mpg_{index}" for index in range(30)],
    },
    "forest_mpg_lowk": {
        "kp_iMpc": 0.7,
        "kmax_3d_iMpc": 4.0,
        "kmax_1d_iMpc": 4.0,
        "zmax": 4.1,
        "list_sim_cube": [f"mpg_{index}" for index in range(30)],
    },
}

def _normalise_model_domain(domain: Mapping[str, Any] | None) -> dict[str, Any]:
    """
    Validate and copy emulator-domain metadata.
    """
    if domain is None:
        return {}
    result = dict(domain)
    for name in (
        "kp_iMpc",
        "kmax_3d_iMpc",
        "kmax_1d_iMpc",
        "zmax",
    ):
        if name in result:
            result[name] = float(result[name])
            if result[name] <= 0:
                raise ValueError(f"model_domain[{name!r}] must be positive")
    if "list_sim_cube" in result:
        result["list_sim_cube"] = [str(label) for label in result["list_sim_cube"]]
        if not result["list_sim_cube"]:
            raise ValueError("model_domain['list_sim_cube'] must not be empty")
    return result


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
    best_epoch : int or None
        Zero-based epoch of the saved validation checkpoint.
    best_validation_loss : float or None
        Minimum validation loss, or None when training had no validation split.
    model_domain : dict
        Scientific domain metadata stored with the emulator bundle.
    """

    def __init__(
        self,
        key: str = "forest_mpg_fix",
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
        Nrealizations: int = 2048,
        training_provenance: Optional[Mapping[str, Any]] = None,
        model_domain: Optional[Mapping[str, Any]] = None,
        compile_model: bool = False,
    ) -> None:
        """
        Initialize the P3D emulator.

        Parameters
        ----------
        key : str, default="forest_mpg_fix"
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
            File path containing normalization transformations. When training,
            this existing file is recorded and checksummed in the saved model
            manifest. If omitted, ``save_path + '_transf.npy'`` is included
            automatically when it already exists.
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
        Nrealizations : int, default=2048
            Default number of latent space realizations for evaluation.
        training_provenance : mapping, optional
            Archive, selection, and preprocessing details to preserve in the
            saved model manifest. The standardized training-data fingerprint and
            all optimizer settings are recorded automatically.
        model_domain : mapping, optional
            Pivot and maximum wavenumber, maximum redshift, and training
            simulation labels. It is written into metadata for newly trained
            models; legacy bundled models use documented ForestFlow defaults.
        compile_model : bool, default=False
            Compile the neural network with ``torch.compile`` for repeated
            inference. This has a noticeable one-time cost, but can reduce
            evaluation time in long sampling runs.

        Raises
        ------
        ValueError
            If training is requested without required parameters, or if neither
            training nor a model path is provided.
        """

        if train:
            key = None
        self.model_key = key
        if model_domain is None and key in _LEGACY_MODEL_DOMAINS:
            warn(
                f"Emulator bundle {key!r} has no model_domain metadata and "
                "relies on deprecated built-in defaults. Regenerate the "
                "bundle to embed model_domain before the next major release.",
                FutureWarning,
                stacklevel=2,
            )
        self.model_domain = _normalise_model_domain(
            model_domain or _LEGACY_MODEL_DOMAINS.get(key)
        )
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
                transf_file=transf_file,
                training_provenance=training_provenance,
                model_domain=self.model_domain,
            )
        elif model_path is not None:
            self.Nrealizations = Nrealizations
            self.manifest = load_manifest(model_path, transf_file)
            self.transf_data = Transf_data(preload_file=transf_file)
            self._load_emulator(model_path=model_path)
        else:
            raise ValueError(
                "Either train=True with required parameters, or model_path must be provided."
            )

        self.emulator.eval()
        self._latent_cache_key = None
        self._latent_cache = None
        self._compiled = False
        if compile_model:
            self.compile()

    def _domain_value(self, name: str) -> Any:
        try:
            return self.model_domain[name]
        except KeyError as error:
            raise AttributeError(
                f"This emulator bundle does not define {name!r} domain metadata"
            ) from error

    @property
    def kp_iMpc(self) -> float:
        """
        Linear-power pivot wavenumber in inverse Mpc.
        """
        return self._domain_value("kp_iMpc")

    @property
    def kmax_3d_iMpc(self) -> float:
        """
        Maximum P3D fitting cut used to calibrate this emulator.
        """
        return self._domain_value("kmax_3d_iMpc")

    @property
    def kmax_1d_iMpc(self) -> float:
        """
        Maximum P1D fitting cut used to calibrate this emulator.
        """
        return self._domain_value("kmax_1d_iMpc")

    @property
    def zmax(self) -> float:
        """
        Maximum supported redshift.
        """
        return self._domain_value("zmax")

    @property
    def list_sim_cube(self) -> list[str]:
        """
        Simulation labels that define the emulator training domain.
        """
        return self._domain_value("list_sim_cube")

    def compile(self, mode: str = "reduce-overhead") -> None:
        """
        Compile the network for faster repeated inference.

        The first evaluation of each new input shape triggers PyTorch
        compilation and is therefore slower. Subsequent evaluations, such as
        likelihood calls with a fixed number of redshifts, reuse the graph.

        Parameters
        ----------
        mode : str, default="reduce-overhead"
            Compilation mode forwarded to :func:`torch.compile`.
        """
        if not hasattr(torch, "compile"):
            raise RuntimeError("Compiled inference requires PyTorch 2.0 or newer")
        if self._compiled:
            return
        self.emulator = torch.compile(self.emulator, mode=mode)
        self._compiled = True

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
        self.best_epoch = metadata.get("best_epoch")
        self.best_validation_loss = metadata.get("best_validation_loss")
        self.model_domain = _normalise_model_domain(
            metadata.get("model_domain", self.model_domain)
        )

        # Reconstruct the cINN architecture
        self.emulator = self._define_cINN_Arinyo(
            metadata["nLayers_inn"],
            metadata["batch_size"],
            metadata["dim_inputSpace"],
            dims_int=metadata["dims_int"],
        )

        # Load pre-trained weights
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
        transf_file: Optional[str] = None,
        training_provenance: Optional[Mapping[str, Any]] = None,
        model_domain: Optional[Mapping[str, Any]] = None,
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
            "training_data_fingerprint": _training_data_fingerprint(training_data),
            "model_domain": _normalise_model_domain(model_domain),
        }
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
        best_epoch = None
        best_state_dict = None
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

                significant_improvement = val_loss < best_val - 1e-6
                if val_loss < best_val:
                    best_val = val_loss
                    best_epoch = epoch
                    best_state_dict = copy.deepcopy(self.emulator.state_dict())

                if significant_improvement:
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

        # Validation selects the weights that are persisted and left active.
        # Without validation, retain the final epoch as before.
        if use_val_set:
            if best_state_dict is None:
                raise RuntimeError("Validation did not produce a finite checkpoint")
            self.emulator.load_state_dict(best_state_dict)
            self.best_epoch = best_epoch
            self.best_validation_loss = float(best_val)
        else:
            self.best_epoch = len(self.loss_arr) - 1
            self.best_validation_loss = None

        metadata["best_epoch"] = self.best_epoch
        metadata["best_validation_loss"] = self.best_validation_loss

        # Write metadata and weights together after selecting the checkpoint.
        if save_path is not None:
            np.save(save_path + "_metadata.npy", metadata)
            torch.save(self.emulator.state_dict(), save_path + ".pt")
            if transf_file is None:
                candidate = save_path + "_transf.npy"
                transf_file = candidate if os.path.isfile(candidate) else None
            provenance = dict(training_provenance or {})
            provenance.update(
                {
                    "train_seed": train_seed,
                    "use_validation_set": use_val_set,
                    "training_data_fingerprint": metadata["training_data_fingerprint"],
                    "transformation_file": (
                        None if transf_file is None else str(transf_file)
                    ),
                }
            )
            self.manifest = write_manifest(save_path, transf_file, provenance)

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
        latent_indices: Optional[Sequence[int]] = None,
        latent_group_ids: Optional[Sequence[int]] = None,
        *,
        sampler: str = "sobol_antithetic",
        statistic: str = "mean",
        aggregation_space: str = "transformed",
        draw_policy: str = "nested",
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
        latent_indices : sequence of int, optional
            Latent block assigned to each input. This allows a larger batch to
            reproduce the random samples used by independent redshift batches.
        latent_group_ids : sequence of int, optional
            Stable group identities, one per input. These make nested draws
            independent of evaluation chunking. Cannot be combined with
            ``latent_indices``.
        sampler : {"gaussian", "antithetic", "sobol", "sobol_antithetic"}, default="sobol_antithetic"
            Latent sampler. Antithetic sampling uses paired ``z`` and ``-z``.
        statistic : {"mean", "median"}, default="mean"
            Statistic used to reduce latent predictions.
        aggregation_space : {"transformed", "physical"}, default="transformed"
            Reduce before or after inverse output transformation. The default
            reproduces the historical prediction.
        draw_policy : {"legacy", "nested"}, default="nested"
            Nested draws retain the same per-group prefix as N increases.

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
        if not isinstance(Nrealizations, (int, np.integer)) or Nrealizations < 1:
            raise ValueError("Nrealizations must be a positive integer")
        if sampler not in {"gaussian", "antithetic", "sobol", "sobol_antithetic"}:
            raise ValueError("sampler must be 'gaussian', 'antithetic', 'sobol', or 'sobol_antithetic'")
        if statistic not in {"mean", "median"}:
            raise ValueError("statistic must be 'mean' or 'median'")
        if aggregation_space not in {"transformed", "physical"}:
            raise ValueError("aggregation_space must be 'transformed' or 'physical'")
        if draw_policy not in {"legacy", "nested"}:
            raise ValueError("draw_policy must be 'legacy' or 'nested'")
        if sampler in {"antithetic", "sobol_antithetic"} and Nrealizations % 2:
            raise ValueError("antithetic sampling requires an even Nrealizations")
        if latent_indices is not None and latent_group_ids is not None:
            raise ValueError("latent_indices and latent_group_ids are mutually exclusive")

        # Prepare conditioned inputs
        neval = len(emu_params)
        condition = self._prepare_condition_tensor(emu_params, neval, Nrealizations)

        # Generate predictions
        generator = torch.Generator(device=condition.device).manual_seed(seed)
        prediction = self._generate_predictions(
            condition,
            neval,
            Nrealizations,
            generator,
            seed=seed,
            latent_indices=latent_indices,
            latent_group_ids=latent_group_ids,
            sampler=sampler,
            draw_policy=draw_policy,
        )

        return self._aggregate_predictions(
            prediction, neval, statistic, aggregation_space
        )

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
        normalized = []
        for params in emu_params:
            # Normalize input parameters
            dict_input = self.transf_data.transf_stand(
                params, type_stand="input", direct=True
            )
            normalized.append([dict_input[par] for par in self.input_labels])

        device = next(self.emulator.parameters()).device
        condition = torch.as_tensor(normalized, dtype=torch.float32, device=device)
        return torch.repeat_interleave(condition, Nrealizations, dim=0)

    def _generate_predictions(
        self,
        condition: torch.Tensor,
        neval: int,
        Nrealizations: int,
        generator: torch.Generator,
        seed: Optional[int] = None,
        latent_indices: Optional[Sequence[int]] = None,
        latent_group_ids: Optional[Sequence[int]] = None,
        sampler: str = "gaussian",
        draw_policy: str = "legacy",
    ) -> torch.Tensor:
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
        seed : int, optional
            Seed associated with ``generator``. When supplied, deterministic
            latent samples are cached for subsequent calls with the same shape.
        latent_indices : sequence of int, optional
            Latent block to use for each evaluation point.

        Returns
        -------
        torch.Tensor
            Per-realization transformed predictions with shape
            ``(neval, Nrealizations, dim_inputSpace)``.
        """
        # Setup conditions for the cINN
        aran = np.arange(neval * Nrealizations)
        self.emulator.conditions = [aran] * self.nLayers_inn

        n_samples = neval * Nrealizations
        device = condition.device
        if latent_group_ids is not None:
            latent_group_ids = np.asarray(latent_group_ids, dtype=np.int64)
            if latent_group_ids.shape != (neval,):
                raise ValueError("latent_group_ids must contain one integer per input")
            latent_indices_key = ("stable", tuple(latent_group_ids.tolist()))
            n_latent_groups = neval
        elif latent_indices is None:
            latent_indices_key = None
            n_latent_groups = neval
        else:
            latent_indices = np.asarray(latent_indices, dtype=int)
            if latent_indices.shape != (neval,) or np.any(latent_indices < 0):
                raise ValueError(
                    "latent_indices must contain one non-negative index per input"
                )
            latent_indices_key = tuple(latent_indices.tolist())
            n_latent_groups = int(np.max(latent_indices)) + 1
        cache_key = (
            n_samples,
            self.dim_inputSpace,
            seed,
            device.type,
            device.index,
            latent_indices_key,
            sampler,
            draw_policy,
        )
        if seed is not None and self._latent_cache_key == cache_key:
            z_test = self._latent_cache
        else:
            latent = self._draw_latents(
                n_latent_groups,
                Nrealizations,
                device,
                seed,
                generator,
                sampler,
                draw_policy,
                latent_indices if latent_indices is not None else None,
                latent_group_ids,
            )
            if latent_group_ids is not None or latent_indices is None:
                z_test = latent
            else:
                z_test = latent.reshape(
                    n_latent_groups, Nrealizations, self.dim_inputSpace
                )[latent_indices].reshape(n_samples, self.dim_inputSpace)
            if seed is not None:
                # Keep only the most recently used tensor to bound memory use.
                self._latent_cache_key = cache_key
                self._latent_cache = z_test

        # Reduce on the model device before crossing the PyTorch/NumPy boundary.
        # Only the mean is part of the public emulator prediction, so this
        # avoids materialising every realization in NumPy. ``no_grad`` is kept
        # rather than ``inference_mode`` because torch.compile guards FrEIA
        # tensors against an inference-mode dispatch-key change.
        with torch.no_grad():
            out_emu, _ = self.emulator(z_test, condition, rev=True)

        return out_emu.reshape(neval, Nrealizations, self.dim_inputSpace)

    def _draw_latents(
        self,
        n_groups,
        n_realizations,
        device,
        seed,
        generator,
        sampler,
        draw_policy,
        latent_indices,
        latent_group_ids=None,
    ):
        """Draw latent vectors with reproducible group and prefix semantics."""

        if draw_policy == "legacy" and sampler == "gaussian":
            return torch.randn(
                n_groups * n_realizations,
                self.dim_inputSpace,
                generator=generator,
                device=device,
            )

        draws = []
        for group in range(n_groups):
            group_identity = group if latent_group_ids is None else int(latent_group_ids[group])
            # Keep hashed stable identities within torch's accepted seed range.
            group_seed = (int(seed) + 1_000_003 * group_identity) % (2**63 - 1)
            if sampler in {"sobol", "sobol_antithetic"}:
                engine = torch.quasirandom.SobolEngine(
                    self.dim_inputSpace, scramble=True, seed=group_seed
                )
                n_base = n_realizations // 2 if sampler == "sobol_antithetic" else n_realizations
                uniform = engine.draw(n_base).to(device=device)
                eps = torch.finfo(uniform.dtype).eps
                latent = torch.erfinv(uniform.clamp(eps, 1.0 - eps) * 2.0 - 1.0)
                latent = latent * np.sqrt(2.0)
                if sampler == "sobol_antithetic":
                    latent = torch.stack((latent, -latent), dim=1).reshape(
                        n_realizations, self.dim_inputSpace
                    )
            else:
                group_generator = torch.Generator(device=device).manual_seed(group_seed)
                if sampler == "antithetic":
                    # Generate one vector at a time: torch's vectorized normal
                    # kernels need not retain a prefix across requested shapes.
                    half = torch.stack(
                        [
                            torch.randn(
                                self.dim_inputSpace,
                                generator=group_generator,
                                device=device,
                            )
                            for _ in range(n_realizations // 2)
                        ]
                    )
                    # Interleaving makes a smaller even-N sequence a prefix
                    # of a larger one while preserving each z, -z pair.
                    latent = torch.stack((half, -half), dim=1).reshape(
                        n_realizations, self.dim_inputSpace
                    )
                else:
                    latent = torch.stack(
                        [
                            torch.randn(
                                self.dim_inputSpace,
                                generator=group_generator,
                                device=device,
                            )
                            for _ in range(n_realizations)
                        ]
                    )
            draws.append(latent)

        # The caller applies ``latent_indices`` after this group-major layout,
        # matching the historical Gaussian implementation.
        return torch.stack(draws).reshape(
            n_groups * n_realizations, self.dim_inputSpace
        )

    def _aggregate_predictions(self, prediction, neval, statistic, aggregation_space):
        """Reduce latent predictions in transformed or physical output space."""

        if aggregation_space == "transformed":
            if statistic == "mean":
                reduced = prediction.mean(dim=1)
            else:
                # quantile uses the midpoint of the two central values for an
                # even realization count, matching numpy.median below.
                reduced = torch.quantile(prediction, 0.5, dim=1)
            return self._process_predictions(reduced.cpu().numpy(), neval)

        samples = {
            name: prediction[:, :, index].cpu().numpy()
            for index, name in enumerate(self.output_labels)
        }
        physical = self.transf_data.transf_stand(
            samples, type_stand="output", direct=False
        )
        result = {}
        for name, values in physical.items():
            values = np.asarray(values)
            result[name] = (
                values.mean(axis=1)
                if statistic == "mean"
                else np.median(values, axis=1)
            )
            if neval == 1:
                result[name] = result[name][0]
        return result

    def _process_predictions(
        self, mean_prediction: np.ndarray, neval: int
    ) -> Dict[str, np.ndarray]:
        """
        Process and transform predictions back to the original space.

        Parameters
        ----------
        mean_prediction : np.ndarray
            Mean prediction in transformed output space.
        neval : int
            Number of evaluation points.

        Returns
        -------
        dict
            Dictionary of processed predictions.
        """
        # Convert to dictionary format
        dict_tswn_output = {
            par: mean_prediction[:, ii] for ii, par in enumerate(self.output_labels)
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
