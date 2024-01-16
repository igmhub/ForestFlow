#torch modules
import torch
from torch.utils.data import DataLoader, dataset, TensorDataset
from torch import nn, optim

# FrEIA imports
import FrEIA.framework as Ff
import FrEIA.modules as Fm

#lace modeles
from ForestFlow.model_p3d_arinyo import ArinyoModel
from lace.cosmo.camb_cosmo import get_cosmology
from ForestFlow.archive import GadgetArchive3D
from ForestFlow.likelihood import Likelihood


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
        use_chains (bool): Whether to use the MCMC chains instead of the bestfitting Arinyo parameters. Default is False.
        folder_chains (str): Path to the folder containing chains. Default is '/data/desi/scratch/jchavesm/p3d_fits_new/'.
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
        use_chains=False,
        folder_chains='/data/desi/scratch/jchavesm/p3d_fits_new/',
        Archive=None,
        chain_samp=100_000,
        Nrealizations=100
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
        self.nLayers_inn= nLayers_inn
        self.train=train
        self.Nrealizations=Nrealizations
        
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_chains=use_chains
        self.folder_chains=folder_chains
        self.archive=Archive
        self.chain_samp=chain_samp
        self.Arinyo_params=['bias', 'beta', 'q1', 'kvav', 'av',  'bv', 'kp','q2']
        self.folder_interp="../data/plin_interp/"
        self.model_path=model_path
        self.save_path=save_path
        
        if self.use_chains==True:
            if self.archive==None:
                raise ValueError('Use_chains==True requires loadinig an archive')
                
            print('Warning: using the chains takes longer '
                  'Loading the chains is around 3 minutes. '
                  'Be pacient!')
            
        # Set random seeds for reproducibility
        torch.manual_seed(32)
        np.random.seed(32)
        random.seed(32)
        
        # Extract k_Mpc and mu from the training data
        k_Mpc = self.archive.training_data[0]["k3d_Mpc"]
        mu = self.archive.training_data[0]["mu3d"]

        # Create a mask for k values within the specified kmax_Mpc range
        k_mask = (k_Mpc < self.kmax_Mpc) & (k_Mpc > 0)
        
        self.k_mask=k_mask
        
        self.k_Mpc_masked=k_Mpc[self.k_mask]
        self.mu_masked= mu[self.k_mask]
        
        self._train_Arinyo()
             

    def _params_numpy2dict(self, array, key_strings=["bias", "beta", "q1", "kvav", "av", "bv", "kp", "q2"]):
        """
        Convert a numpy array of parameters to a dictionary.

        Args:
            array (numpy.ndarray): Array of parameters.
            key_strings (list): List of strings for dictionary keys. Default is ["bias", "beta", "q1", "kvav", "av", "bv", "kp", "q2"].

        Returns:
            dict: Dictionary with key-value pairs corresponding to parameters.
        """
        # Create a dictionary with key strings and array elements
        array_dict = {}
        for key, value in zip(key_strings, array):
            array_dict[key] = value

        return array_dict


    def _sort_dict(self, dct, keys):
        """
        Sort a list of dictionaries based on specified keys.

        Args:
            dct (list): List of dictionaries to be sorted.
            keys (list): List of keys to sort the dictionaries by.

        Returns:
            list: The sorted list of dictionaries.
        """
        for d in dct:
            sorted_d = {
                k: d[k] for k in keys
            }  # create a new dictionary with only the specified keys
            d.clear()  # remove all items from the original dictionary
            d.update(
                sorted_d
            )  # update the original dictionary with the sorted dictionary
        return dct

    
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
            {key: value for key, value in self.training_data[i].items() if key in self.emuparams}
            for i in range(len(self.training_data))
        ]

        # Sort the training data based on self.emuparams
        training_data = self._sort_dict(training_data, self.emuparams)

        # Convert the sorted training data to a list of lists
        training_data = [list(training_data[i].values()) for i in range(len(self.training_data))]

        # Convert the training data to a numpy array
        training_data = np.array(training_data)

        # Calculate and store the maximum and minimum values for parameter scaling
        self.param_lims_max = training_data.max(0)
        self.param_lims_min = training_data.min(0)

        # Scale the training data based on the parameter limits
        training_data = (training_data - self.param_lims_max) / (self.param_lims_max - self.param_lims_min)

        # Convert the scaled training data to a torch.Tensor object
        training_data = torch.Tensor(training_data)

        return training_data

    def _get_test_condition(self, test_data):
        """
        Extract and sort test condition from given test data.

        This function takes a list of dictionaries (test_data) containing emulator parameters
        and extracts the relevant parameters specified by self.archive.emu_params. It then sorts
        the extracted parameters based on the emulator parameters and returns them as a numpy array.

        Args:
            test_data (list): List of dictionaries containing emulator parameters for test data.

        Returns:
            np.array: Test conditions sorted based on emulator parameters.
        """
        # Extract emulator parameters from the test data
        condition = [
            {key: value for key, value in test_data[i].items() if key in self.archive.emu_params}
            for i in range(len(test_data))
        ]

        # Sort the conditions based on emulator parameters
        condition = self._sort_dict(condition, self.archive.emu_params)

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
            {key: value for key, value in self.training_data[i]["Arinyo"].items() if key in self.Arinyo_params}
            for i in range(len(self.training_data))
        ]

        # Sort the Arinyo parameters based on self.Arinyo_params
        training_label = self._sort_dict(training_label, self.Arinyo_params)

        # Convert the sorted Arinyo parameters to a list of lists
        training_label = [list(training_label[i].values()) for i in range(len(self.training_data))]

        # Convert the Arinyo parameters to a numpy array
        training_label = np.array(training_label)

        # Apply specific transformations to certain parameters
        training_label[:, 2] = np.exp(training_label[:, 2])
        training_label[:, 4] = np.exp(training_label[:, 4])
        training_label[:, 6] = np.log(training_label[:, 6])
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
        Plin = [d['Plin'] for d in self.training_data]

        # Convert the list of Plin to a numpy array
        Plin = np.array(Plin)

        return Plin


    def predict_P3D_Mpc(self, sim_label, z, test_sim, return_cov=True):
        """
        Predict the power spectrum using the emulator for a given simulation label and redshift.

        This function predicts the power spectrum using the emulator for a specified simulation label and redshift.
        It utilizes the Arinyo model to generate power spectrum predictions based on the emulator coefficients.

        Args:
            sim_label (str): The simulation label.
            z (float): The redshift.
            test_sim (list): List of dictionaries containing emulator parameters for test data.
            return_cov (bool, optional): Whether to return the covariance matrix. Default is True.

        Returns:
            np.array: Predicted power spectrum.
            np.array or None: Covariance matrix if return_cov is True, otherwise None.
        """
        # Extract simulation index from the given simulation label
        underscore_index = sim_label.find('_')
        s = sim_label[underscore_index + 1:]
        flag = f'Plin_interp_sim{s}.npy'

        # Load pre-interpolated matter power spectrum for the specified simulation
        file_plin_inter = self.folder_interp + flag
        pk_interp = np.load(file_plin_inter, allow_pickle=True).all()

        # Initialize Arinyo model with the loaded matter power spectrum
        model_Arinyo = ArinyoModel(camb_pk_interp=pk_interp)

        # Extract and sort emulator parameters from the test data
        testing_condition = self._get_test_condition(test_sim)

        # Predict Arinyo coefficients for the given test conditions
        coeffs_all, coeffs_mean = self.predict_Arinyos(
            testing_condition, return_all_realizations=True
        )

        # Predict power spectrum using Arinyo model with predicted coefficients
        if return_cov:
            # If return_cov is True, predict multiple realizations and calculate the covariance matrix
            p3ds_pred = np.zeros(shape=(self.Nrealizations, len(self.k_Mpc_masked)))
            for r in range(self.Nrealizations):
                arinyo_params = self._params_numpy2dict(coeffs_all[r])
                p3d_pred = model_Arinyo.P3D_Mpc(z, self.k_Mpc_masked, self.mu_masked, arinyo_params)
                p3ds_pred[r] = p3d_pred

            # Return the median power spectrum and its covariance matrix
            p3d_arinyo = np.nanmedian(p3ds_pred, 0)
            p3d_cov = np.cov(p3ds_pred.T)
            return p3d_arinyo, p3d_cov

        else:
            # If return_cov is False, predict the power spectrum using the mean coefficients
            NF_arinyo = self._params_numpy2dict(coeffs_mean)
            p3d_arinyo = model_Arinyo.P3D_Mpc(z, self.k_Mpc_masked, self.mu_masked, NF_arinyo)
            return p3d_arinyo
 
                
    def predict_P1D_Mpc(self, sim_label, z, test_sim, return_cov=True):
        """
        Predict the one-dimensional matter power spectrum using the emulator for a given simulation label and redshift.

        This function predicts the one-dimensional matter power spectrum using the emulator for a specified simulation
        label and redshift. It utilizes the Arinyo model and likelihood calculations to generate predictions based on
        the emulator coefficients.

        Args:
            sim_label (str): The simulation label.
            z (float): The redshift.
            test_sim (list): List of dictionaries containing emulator parameters for test data.
            return_cov (bool, optional): Whether to return the covariance matrix. Default is True.

        Returns:
            np.array: Predicted one-dimensional power spectrum.
            np.array or None: Covariance matrix if return_cov is True, otherwise None.
        """
        # Extract simulation index from the given simulation label
        underscore_index = sim_label.find('_')
        s = sim_label[underscore_index + 1:]
        flag = f'Plin_interp_sim{s}.npy'

        # Load pre-interpolated matter power spectrum for the specified simulation
        file_plin_inter = self.folder_interp + flag
        pk_interp = np.load(file_plin_inter, allow_pickle=True).all()

        # Initialize likelihood with test data and relative errors
        like = Likelihood(test_sim[0], self.archive.rel_err_p3d, self.archive.rel_err_p1d)

        # Create a mask for the 1D power spectrum fit
        k1d_mask = like.like.ind_fit1d.copy()

        # Extract and sort emulator parameters from the test data
        testing_condition = self._get_test_condition(test_sim)

        # Predict Arinyo coefficients for the given test conditions
        coeffs_all, coeffs_mean = self.predict_Arinyos(
            testing_condition, return_all_realizations=True
        )

        # Predict one-dimensional power spectrum using Arinyo model with predicted coefficients
        if return_cov:
            # If return_cov is True, predict multiple realizations and calculate the covariance matrix
            p1ds_pred = np.zeros(shape=(self.Nrealizations, 53))
            for r in range(self.Nrealizations):
                arinyo_params = self._params_numpy2dict(coeffs_all[r])
                p1d_pred = like.like.get_model_1d(parameters=arinyo_params)
                p1d_pred = p1d_pred[k1d_mask]
                p1ds_pred[r] = p1d_pred

            # Return the median one-dimensional power spectrum and its covariance matrix
            p1d_pred = np.nanmedian(p1ds_pred, 0)
            p1d_cov = np.cov(p1ds_pred.T)
            return p1d_pred, p1d_cov

        else:
            # If return_cov is False, predict the one-dimensional power spectrum using the mean coefficients
            NF_arinyo = self._params_numpy2dict(coeffs_mean)
            p1d_pred = like.like.get_model_1d(parameters=NF_arinyo)
            p1d_pred = p1d_pred[k1d_mask]
            return p1d_pred



    def _load_Arinyo_chains(self):
        """
        Load Arinyo model chains from stored files.

        This function loads Arinyo model chains corresponding to different simulations from saved files. 
        It extracts relevant information such as simulation label, scaling factor, redshift, and other parameters 
        to construct the file tag for each simulation. The loaded chains are then processed and returned.

        Returns:
            np.array: Array containing Arinyo model chains for all simulations.
        """
        print('Loading Arinyo chains')

        # Initialize array to store Arinyo model chains
        chains = np.zeros(shape=(len(self.training_data), self.chain_samp, 8))

        # Loop over simulations in the training data
        for ind_book in range(0, len(self.training_data)):
            sim_label = self.training_data[ind_book]["sim_label"]
            scale_tau = self.training_data[ind_book]["val_scaling"]
            ind_z = self.training_data[ind_book]["z"]

            # Construct file tag based on simulation parameters
            tag = (
                "fit_sim"
                + sim_label[4:]
                + "_tau"
                + str(np.round(scale_tau, 2))
                + "_z"
                + str(ind_z)
                + "_kmax3d"
                + str(self.archive.kmax_3d)
                + "_noise3d"
                + str(self.archive.noise_3d)
                + "_kmax1d"
                + str(self.archive.kmax_1d)
                + "_noise1d"
                + str(self.archive.noise_1d)
            )

            # Load Arinyo model chain from file
            file_arinyo = np.load(self.folder_chains + tag + ".npz")
            chain = file_arinyo["chain"].copy()

            # Ensure non-positive values for the first parameter
            chain[:, 0] = -np.abs(chain[:, 0])

            # Randomly sample from the loaded chain
            idx = np.random.randint(len(chain), size=(self.chain_samp))
            chain_sampled = chain[idx]
            chains[ind_book] = chain_sampled

        # Apply logarithmic transformations to specified parameters
        chains[:, :, 2] = np.log(chains[:, :, 2])
        chains[:, :, 6] = np.log(chains[:, :, 6])
        chains[:, :, 7] = np.log(chains[:, :, 7])

        print('Chains loaded')
        return chains

    def _get_Arinyo_chains(self):
        """
        Load and convert Arinyo model chains into torch.Tensor.

        This function utilizes the _load_Arinyo_chains function to load Arinyo model chains, 
        and then converts them into torch.Tensor format before returning.

        Returns:
            torch.Tensor: Tensor containing Arinyo model chains.
        """
        chains = self._load_Arinyo_chains()
        chains = torch.Tensor(chains)
        return chains


    def _define_cINN_Arinyo(self, dim_inputSpace=8):
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
            return nn.Sequential(nn.Linear(dims_in, 64), nn.ReLU(),
                                 nn.Dropout(0),
                                 nn.Linear(64, 128), nn.ReLU(),
                                 nn.Dropout(0),
                                 nn.Linear(128,  dims_out))

        # Initialize the cINN model
        emulator  = Ff.SequenceINN(dim_inputSpace)

        # Append AllInOneBlocks to the cINN model based on the specified number of layers
        for l in range(self.nLayers_inn):
            emulator.append(Fm.AllInOneBlock, 
                        cond=[i for i in range(self.batch_size)], 
                        cond_shape=[6], 
                        subnet_constructor=subnet_fc
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
            print('WARNING: loading a pre-trained emulator')
            self.emulator.load_state_dict(torch.load(self.model_path))
            return

        # Initialize the cINN model with Xavier initialization
        self.emulator.apply(init_xavier)

        # Get Arinyo coefficients for training
        if self.use_chains == False:
            Arinyo_coeffs = self._get_Arinyo_params()
        else:
            Arinyo_coeffs = self._get_Arinyo_chains()

        # Create a PyTorch dataset and loader for training
        trainig_dataset = TensorDataset(training_data, Arinyo_coeffs)
        loader = DataLoader(trainig_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        # Choose the optimizer (Adam or AdamW)
        if self.adamw:
            optimizer = torch.optim.AdamW(self.emulator.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = optim.Adam(self.emulator.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size)

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
                    idx = np.random.choice(self.chain_samp, size=2_000, replace=False)
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

    def predict_Arinyos(self, input_emu, plot=False, true_coeffs=None, return_all_realizations=False):
        """
        Predict Arinyo coefficients using the trained emulator.

        Args:
            input_emu (list): List of Arinyo input parameters.
            plot (bool): Whether to generate a corner plot. Default is False.
            true_coeffs (list): True Arinyo coefficients for plotting comparison. Default is None.
            return_all_realizations (bool): Whether to return all realizations or just the mean. Default is False.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: Arinyo coefficient predictions. If return_all_realizations is True, returns a tuple with all realizations and the mean.
        """
        self.emulator = self.emulator.eval()

        # Normalize the input data
        test_data = np.array(input_emu)
        test_data = (test_data - self.param_lims_max) / (self.param_lims_max - self.param_lims_min)
        test_data = torch.Tensor(test_data)

        # Number of iterations for batch processing
        Niter = int(self.Nrealizations / self.batch_size)

        # Initialize array for Arinyo predictions
        Arinyo_preds = np.zeros(shape=(Niter, self.batch_size, 8))
        condition = torch.tile(test_data, (self.batch_size, 1))

        # Generate predictions
        for ii in range(Niter):
            z_test = torch.randn(self.batch_size, 8)
            Arinyo_pred, _ = self.emulator(z_test, condition, rev=True)

            # Transform the predictions back to original space
            Arinyo_pred[:, 2] = torch.log(Arinyo_pred[:, 2])
            Arinyo_pred[:, 4] = torch.log(Arinyo_pred[:, 4])
            Arinyo_pred[:, 6] = torch.exp(Arinyo_pred[:, 6])
            Arinyo_pred[:, 7] = torch.log(Arinyo_pred[:, 7])

            Arinyo_preds[ii, :] = Arinyo_pred.detach().cpu().numpy()

        Arinyo_preds = Arinyo_preds.reshape(Niter * int(self.batch_size), 8)

        # Generate corner plot if plot is True
        if plot == True:
            if true_coeffs is None:
                corner_plot = corner.corner(
                    Arinyo_preds,
                    labels=[r'$b$', r'$\beta$', '$q_1$', '$k_{vav}$', '$a_v$', '$b_v$', '$k_p$', '$q_2$'],
                    truth_color='crimson'
                )
            else:
                corner_plot = corner.corner(
                    Arinyo_preds,
                    labels=[r'$b$', r'$\beta$', '$q_1$', '$k_{vav}$', '$a_v$', '$b_v$', '$k_p$', '$q_2$'],
                    truths=true_coeffs,
                    truth_color='crimson'
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

            

        
