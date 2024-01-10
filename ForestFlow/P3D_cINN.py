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
        emuparams (dict): A dictionary of emulator parameters.
        kmax_Mpc (float): The maximum k in Mpc^-1 to use for training. Default is 3.5.
        zmax (float): The maximum redshift to use for training. Default is 4.5.
        Nsim (int): The number of simulations to use for training. Default is 30.
        nepochs (int): The number of epochs to train for. Default is 200.
        postprocessing (str): The post-processing method to use. Default is '3A'.
        model_path (str): The path to a pretrained model. Default is None.
        drop_sim (float): The simulation to drop during training. Default is None.
        drop_z (float): Drop all snapshpts at redshift z from the training. Default is None.
        pick_z (float): Pick only snapshpts at redshift z. Default is None.
        drop_rescalings (bool): Wheather to drop the optical-depth rescalings or not. Default False.
        train (bool): Wheather to train the emulator or not. Default True. If False, a model path must is required.
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
        input_space='Arinyo',
        grad_clip_threshold=1e25
    ):
        
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
        
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.use_chains=use_chains
        self.folder_chains=folder_chains
        self.archive=Archive
        self.chain_samp=chain_samp
        self.Arinyo_params=['bias', 'beta', 'q1', 'kvav', 'av',  'bv', 'kp','q2']
        self.input_space=input_space
        self.grad_clip_threshold = grad_clip_threshold
        self.folder_interp="/data/plin_interp/"
        self.model_path=model_path
        self.save_path=save_path
        
        if self.use_chains==True:
            if self.archive==None:
                raise ValueError('Use_chains==True requires loadinig an archive')
                
            print('Warning: using the chains takes longer '
                  'Loading the chains is around 3 minutes. '
                  'Be pacient!')

        torch.manual_seed(32)
        np.random.seed(32)
        random.seed(32)
        
        k_Mpc = self.archive.training_data[0]["k3d_Mpc"]
        mu = self.archive.training_data[0]["mu3d"]

        k_mask = (k_Mpc < self.kmax_Mpc) & (k_Mpc > 0)
        
        self.k_mask=k_mask
        
        self.k_Mpc_masked=k_Mpc[self.k_mask]
        self.mu_masked= mu[self.k_mask]
        
            
        
        if self.input_space=='Arinyo':
            self._train_Arinyo()
        
        if self.input_space=='P3D':
            self._train_P3D()        
        

    def params_numpy2dict(self, array, key_strings=["bias", "beta", "q1", "kvav", "av", "bv", "kp", "q2"]):

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
        Given an archive and key_av, it obtains the training data based on self.emuparams
        Sorts the training data according to self.emuparams and scales the data based on self.paramLims
        Finally, it returns the training data as a torch.Tensor object.
        """
        training_data = [
            {
                key: value
                for key, value in self.training_data[i].items()
                if key in self.emuparams
            }
            for i in range(len(self.training_data))
        ]
        training_data = self._sort_dict(training_data, self.emuparams)
        training_data = [
            list(training_data[i].values())
            for i in range(len(self.training_data))
        ]

        
        training_data = np.array(training_data)
        self.param_lims_max = training_data.max(0)
        self.param_lims_min = training_data.min(0)
        
        training_data = (training_data - self.param_lims_max) / (self.param_lims_max-self.param_lims_min)
        
        training_data = torch.Tensor(training_data)

        return training_data
    
    
    def _get_Arinyo_params(self):

        training_label = [
                {
                    key: value
                    for key, value in self.training_data[i]["Arinyo"].items()
                    if key in self.Arinyo_params
                }
                for i in range(len(self.training_data))
            ]
         
        training_label = self._sort_dict(training_label, self.Arinyo_params)
        
        training_label = [
            list(training_label[i].values())
            for i in range(len(self.training_data))
        ]
        
        training_label = np.array(training_label)
        
        
        #training_label[:,2] = np.log(training_label[:,2])
        #training_label[:,6] = np.log(training_label[:,6])
        #training_label[:,7] = np.log(training_label[:,7])
        training_label[:,2] = np.exp(training_label[:,2])
        training_label[:,4] = np.exp(training_label[:,4])
        training_label[:,6] = np.log(training_label[:,6])
        training_label[:,7] = np.exp(training_label[:,7])        
        
        training_label = torch.Tensor(training_label)
        
        return training_label
    
    def _get_Plin(self):
        Plin = [ d['Plin'] for d in self.training_data]
        Plin = np.array(Plin)
        return Plin
    
    def _get_p3d_training(self):
        p3d = [ d['p3d_Mpc'] for d in self.training_data]
        p3d = np.array(p3d)
        
        #z = [ d['z'] for d in self.training_data]
        #z = np.array(z)
        
        #Plin = self._get_Plin()
                
        p3d = [
            p3d[i][self.k_mask]
            for i in range(len(p3d))
                ]

        p3d = torch.Tensor(np.array(p3d))
        
        
        p3d_normed =  (p3d - p3d.mean(0)[None,:]) / p3d.std(0)[None,:]
        
        self.mean_training=p3d.mean(0)
        self.std_training=p3d.std(0)

    
        return p3d, p3d_normed 
    
    """def get_P3D_Mpc(self, sim_label, model):
        
        flag = f'Plin_interp_sim{s}.npy'

        file_plin_inter = folder_interp + flag
        pk_interp = np.load(file_plin_inter, allow_pickle=True).all()

        model_Arinyo = model_p3d_arinyo.ArinyoModel(camb_pk_interp=pk_interp)
        

        
        coeffs_all, coeffs_mean = self.get_coeff(model,
                                        Nrealizations=1000,
                                        return_all_realizations=True
                                                )"""
                
    
    def Arinyo_to_P3d(self, linP, params, epsilon=0.001, log_params=True):
        
        
        if log_params==True:
            #params[:,2] = np.exp(params[:,2])
            #params[:,6] = np.exp(params[:,6])
            #params[:,7] = np.exp(params[:,7])
            params[:,2] = np.log(params[:,2])
            params[:,4] = np.log(params[:,4])
            params[:,6] = np.exp(params[:,6])
            params[:,7] = np.log(params[:,7])        
        
        k_Mpc = self.training_data[0]["k3d_Mpc"]
        mu = self.training_data[0]["mu3d"]



        k = k_Mpc[self.k_mask]
        mu = mu[self.k_mask]
        
        self.k_Mpc_masked=k
        self.mu_masked=mu

        
        linP = [
            linP[i][self.k_mask]
            for i in range(len(linP))
                ]

        linP = torch.Tensor(np.array(linP))
        
        # avoid the code to crash (log)
        mu[mu == 0] += epsilon
        mu = torch.Tensor(mu)
        k = torch.Tensor(k)

        bias, beta, q1, kvav, av, bv, kp, q2 = (
            params[:,0],
            params[:,1],
            params[:,2],
            params[:,3],
            params[:,4],
            params[:,5],
            params[:,6],
            params[:,7],
        )

        ## CALCULATE P3D

        # model large-scales biasing for delta_flux(k)
        linear_rsd = 1 + beta[:,None] * mu[None,:]**2
        lowk_bias = bias[:,None]  * linear_rsd

        # model small-scales correction (D_NL in Arinyo-i-Prats 2015)
        delta2 = (1 / (2 * torch.pi**2)) * k[None,:]**3 * linP
        nonlin = delta2 * (q1[:,None]  + q2[:,None]  * delta2)
        nonvel = ((k[None,:]**av[:,None] ) / kvav[:,None]) * mu[None,:]**bv[:,None] 
        nonpress = (k[None,:] / kp[:,None] ) ** 2

        D_NL = np.exp(nonlin * (1 - nonvel) - nonpress)

        p3d = linP * lowk_bias**2 * D_NL
        
        p3d_norm = p3d / linP

        return p3d, p3d_norm
    
    def _load_Arinyo_chains(self):
        print('Loading Arinyo chains')
        chains = np.zeros(shape=(len(self.training_data), self.chain_samp, 8))
        for ind_book in range(0, len(self.training_data)):
            sim_label = self.training_data[ind_book]["sim_label"]
            scale_tau = self.training_data[ind_book]["val_scaling"]
            ind_z = self.training_data[ind_book]["z"]

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
            # check folder is not None
            file_arinyo = np.load(self.folder_chains + tag + ".npz")
            #print(file_arinyo["best_params"])
            chain = file_arinyo["chain"].copy()
            chain[:, 0] = -np.abs(chain[:, 0])
            idx = np.random.randint(len(chain), size=(self.chain_samp))
            chain_sampled = chain[idx]
            chains[ind_book] = chain_sampled
        
        chains[:,:,2] = np.log(chains[:,:,2])
        chains[:,:,6] = np.log(chains[:,:,6])
        chains[:,:,7] = np.log(chains[:,:,7])
        
            
        print('Chains loaded')
        return chains
    
    def _get_Arinyo_chains(self):
        chains = self._load_Arinyo_chains()
        chains = torch.Tensor(chains)
        return chains
        

    def _define_cINN_Arinyo(self, dim_inputSpace=8):
        def subnet_fc(dims_in, dims_out):
            return nn.Sequential(nn.Linear(dims_in, 64), nn.ReLU(),
                                 nn.Dropout(0),
                                 nn.Linear(64, 128), nn.ReLU(),
                                 nn.Dropout(0),
                                 nn.Linear(128,  dims_out))
        emulator  = Ff.SequenceINN(dim_inputSpace)
        for l in range(self.nLayers_inn):
            emulator.append(Fm.AllInOneBlock, 
                        cond = [i for i in range(self.batch_size)], 
                        cond_shape=[6], 
                        subnet_constructor=subnet_fc
                       )
            
        return emulator
    

    
    
    def _define_cINN_P3D(self, dim_inputSpace=8):
        def subnet_fc(dims_in, dims_out):
            return nn.Sequential(nn.Linear(dims_in, 100), nn.ReLU(),
                                 nn.Dropout(0),
                                 nn.Linear(100, 50), nn.ReLU(),
                                 nn.Dropout(0),
                                 nn.Linear(50, 100), nn.ReLU(),
                                 nn.Dropout(0),
                                 nn.Linear(100,  dims_out))
        emulator  = Ff.SequenceINN(dim_inputSpace)
        for l in range(self.nLayers_inn):
            emulator.append(Fm.AllInOneBlock, 
                        cond = [i for i in range(self.batch_size)], 
                        cond_shape=[6], 
                        subnet_constructor=subnet_fc
                       )
            
        return emulator
    
    
        
    def _train_P3D(self):
        
        self.emulator = self._define_cINN_P3D(dim_inputSpace=148)
        self.emulator.apply(init_xavier)
        
        
        training_data = self._get_training_data()
        Arinyo = self._get_Arinyo_params()
        linP = self._get_Plin()
        
        
        #p3d_label, p3d_label_normed = self.Arinyo_to_P3d(linP,Arinyo, epsilon=0.001)
        
        p3d_label, p3d_label_normed = self._get_p3d_training()
        
        self.p3d_label=p3d_label_normed
        
        trainig_dataset = TensorDataset(
                training_data, p3d_label_normed
            )
        
        loader = DataLoader(
            trainig_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            drop_last=True
        )        
        
        if self.adamw:
            optimizer = torch.optim.AdamW(
            self.emulator.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay)
        else:
            optimizer = optim.Adam(
                self.emulator.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size)

        self.loss_arr = []
        self.zlist= []

        t0 = time.time()
        for i in range(self.nepochs):
            _loss_arr=[]
            for cond, lab in loader:
                
                optimizer.zero_grad()
                
                z, log_jac_det = self.emulator(lab,cond)
                # calculate the negative log-likelihood of the model with a standard normal prior
                loss = 0.5*torch.sum(z**2, 1) - log_jac_det
                loss = loss.mean() 
                # backpropagate and update the weights
                loss.backward()
                optimizer.step()
            
                _loss_arr.append(loss.item())
                if i==(self.nepochs-1):
                    self.zlist.append(z.detach().cpu().numpy())
            scheduler.step()
            #plt.hist(z.detach().cpu().numpy(), bins=50, range = (-10,10))
            #plt.show()
            #print(np.mean(_loss_arr))
            self.loss_arr.append(np.mean(_loss_arr))
        print(f"Emualtor optimized in {time.time() - t0} seconds")
        
        
    
    def _train_Arinyo(self):
        

        training_data = self._get_training_data()
        self.emulator = self._define_cINN_Arinyo()
        
        if self.model_path!=None:
            print('WARNING: loading a pre-trained emulator')
            self.emulator.load_state_dict(torch.load(self.model_path))
            return
            
        
        self.emulator.apply(init_xavier)
        if self.use_chains==False:
            Arinyo_coeffs = self._get_Arinyo_params()
        else:
            Arinyo_coeffs = self._get_Arinyo_chains()
                    
        trainig_dataset = TensorDataset(
                training_data, Arinyo_coeffs
            )
        
        loader = DataLoader(
            trainig_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            drop_last=True
        )        
        
        if self.adamw:
            optimizer = torch.optim.AdamW(
            self.emulator.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay)
        else:
            optimizer = optim.Adam(
                self.emulator.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size)

        self.loss_arr = []
        t0 = time.time()
        for i in range(self.nepochs):
            _loss_arr=[]
            _latent_space=[]
            #if i%100==0:
                #print(i)
    
            for cond, coeffs in loader:
                optimizer.zero_grad()
                if self.use_chains==True:
                    idx = np.random.choice(self.chain_samp, 
                                           size=2_000, 
                                           replace=False)

                    coeffs = coeffs[:, idx, :].mean(axis=1)

                z, log_jac_det = self.emulator(coeffs,cond)
                # calculate the negative log-likelihood of the model with a standard normal prior
                loss = 0.5*torch.sum(z**2, 1) - log_jac_det
                loss = loss.mean() 
                # backpropagate and update the weights
                loss.backward()
                #nn.utils.clip_grad_norm_(self.emulator.parameters(), self.grad_clip_threshold)
                optimizer.step()
            
                _loss_arr.append(loss.item())
                _latent_space.append(z)
            scheduler.step()
            #plt.hist(z.detach().cpu().numpy(), bins=50, range = (-10,10))
            #plt.show()
            #print(np.mean(_loss_arr))
            self.loss_arr.append(np.mean(_loss_arr))
            if i== (self.nepochs -1):
                self._latent_space=_latent_space
                
        print(f"Emualtor optimized in {time.time() - t0} seconds")
        if self.save_path!=None:
            torch.save(self.emulator.state_dict(), self.save_path)
        
        
    def get_coeff(self, input_emu, Nrealizations=1000, plot=False, true_coeffs=None, return_all_realizations=False):
        self.emulator = self.emulator.eval()
        test_data = np.array(input_emu)
        
        test_data = (test_data - self.param_lims_max) / (self.param_lims_max-self.param_lims_min)
        
        test_data = torch.Tensor(test_data)
        Niter = int(Nrealizations/self.batch_size)
        
        Arinyo_preds = np.zeros(shape = (Niter,self.batch_size,8))
        condition = torch.tile(test_data, (self.batch_size,1))
        
        
        for ii in range(Niter):
            z_test = torch.randn(self.batch_size, 8)
            Arinyo_pred, _ = self.emulator(z_test,condition, rev=True)
            
            #Arinyo_pred[:,2] = torch.clamp(torch.exp(Arinyo_pred[:,2]), 1e-10,10)
            #Arinyo_pred[:,6] = torch.exp(Arinyo_pred[:,6])
            #Arinyo_pred[:,7] = torch.clamp(torch.exp(Arinyo_pred[:,7]), 1e-10,10)
            
            Arinyo_pred[:,2] = torch.log(Arinyo_pred[:,2])
            Arinyo_pred[:,4] = torch.log(Arinyo_pred[:,4])
            Arinyo_pred[:,6] = torch.exp(Arinyo_pred[:,6])
            Arinyo_pred[:,7] = torch.log(Arinyo_pred[:,7])
            
            
            Arinyo_preds[ii,:] = Arinyo_pred.detach().cpu().numpy()
        Arinyo_preds = Arinyo_preds.reshape(Niter*int(self.batch_size),8)  
        
        if plot==True:
            if true_coeffs is None:
                corner_plot = corner.corner(Arinyo_preds, 
                              labels=[r'$b$', r'$\beta$', '$q_1$', '$k_{vav}$','$a_v$','$b_v$','$k_p$','$q_2$'], 
                              truth_color='crimson')
            else:
                corner_plot = corner.corner(Arinyo_preds, 
                              labels=[r'$b$', r'$\beta$', '$q_1$', '$k_{vav}$','$a_v$','$b_v$','$k_p$','$q_2$'], 
                              truths=true_coeffs,
                              truth_color='crimson')


            # Increase the label font size for this plot
            for ax in corner_plot.get_axes():
                ax.xaxis.label.set_fontsize(16)
                ax.yaxis.label.set_fontsize(16)
                ax.xaxis.set_tick_params(labelsize=12)  
                ax.yaxis.set_tick_params(labelsize=12)
            plt.show()
            
        Arinyo_mean = np.median(Arinyo_preds,0)
        
        if return_all_realizations == True:
            return Arinyo_preds,Arinyo_mean
        else:
            return Arinyo_mean
        
            
    def predict_P3D(self, input_emu, Nrealizations=1000, return_all_realizations=False):
        self.emulator = self.emulator.eval()
        test_data = np.array(input_emu)
        test_data = torch.Tensor(test_data)
        Niter = int(Nrealizations/self.batch_size)
        
        P3D_preds = np.zeros(shape = (Niter,self.batch_size,148))
        condition = torch.tile(test_data, (self.batch_size,1))
        
        
        for ii in range(Niter):
            z_test = torch.randn(self.batch_size, 148)
            pred_P3D, _ = self.emulator(z_test,condition, rev=True)
            
            
            P3D_preds[ii,:] = pred_P3D.detach().cpu().numpy()
        P3D_preds = P3D_preds.reshape(Niter*int(self.batch_size),148)  
        
            
        P3D_mean = np.mean(P3D_preds,0)
        
        if return_all_realizations == True:
            return P3D_preds,P3D_mean
        else:
            return P3D_mean
        
