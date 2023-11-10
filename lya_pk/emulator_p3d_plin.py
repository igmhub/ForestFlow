import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import sys
import sklearn

# LaCE modules
from lace.emulator import gp_emulator
from lace.emulator import poly_p1d
from lace.emulator import utils

from ForestFlow.emulator_p3d_architecture import P3D_emuv0
from ForestFlow.input_emu import data_for_emu_v1, params_numpy2dict


import torch
from torch.utils.data import DataLoader, dataset, TensorDataset
from torch import nn, optim
from torch.optim import lr_scheduler

from lace.emulator import nn_architecture

import copy


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
        rerr_p3d=None,
        target_space="p3d",
        kmax_Mpc=4,
        ndeg=5,
        nhidden=5,
        nepochs=100,
        batch_size=100,
        lr=1e-3,
        weight_decay=1e-4,
        gamma=0.1,
        amsgrad=False,
        step_size=75,
        init_xavier=True,
        adamw=False,
        Nsim=30,
        train=True,
        drop_sim=None,
        drop_z=None,
        pick_z=None,
        save_path=None,
        drop_rescalings=False,
        model_path=None,
    ):
        self.training_data = training_data
        self.rerr_p3d = rerr_p3d
        self.emuparams = paramList
        self.kmax_Mpc = kmax_Mpc
        self.nepochs = nepochs
        self.step_size = step_size
        self.model_path = model_path
        self.drop_sim = drop_sim
        self.drop_z = drop_z
        self.pick_z = pick_z
        self.init_xavier = init_xavier
        self.adamw = adamw
        self.drop_rescalings = drop_rescalings

        self.ndeg = ndeg
        self.nhidden = nhidden
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.amsgrad = amsgrad

        self.Nsim = Nsim
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.save_path = save_path
        #self.lace_path = utils.ls_level(os.getcwd(), 1)
        #self.models_dir = os.path.join(self.lace_path, "lya_pk/")

        self.target_space = target_space
        self.epsilon = 1e-3

        if train == True:
            self._train()

        if train == False:
            if self.model_path == None:
                raise Exception("If train==False, model path is required.")

            """else:
                pretrained_weights = torch.load(
                    os.path.join(self.models_dir, self.model_path),
                    map_location="cpu",
                )
                self.emulator = P3D_emuv0(nhidden=self.nhidden)
                self.emulator.load_state_dict(pretrained_weights)
                self.emulator.to(self.device)
                print("Model loaded. No training needed")

                kMpc_train = self._obtain_sim_params()

                self.log_KMpc = torch.log10(kMpc_train).to(self.device)

                return"""

        if self.save_path != None:
            # saves the model in the predefined path after training
            self._save_emulator()

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

    def _obtain_sim_params(self):
        """
        Obtain simulation parameters.

        Returns:
            k_Mpc (np.ndarray): Simulation k values.
            Nz (int): Number of redshift values.
            Nk (int): Number of k values.
            k_Mpc_train (tensor): k values in the k training range
        """

        self.k_Mpc = self.training_data[0]["k3d_Mpc"]
        self.mu = self.training_data[0]["mu3d"]

        k_mask = (self.k_Mpc < self.kmax_Mpc) & (self.k_Mpc > 0)
        self.k_mask = k_mask

        k_Mpc_train = self.k_Mpc[self.k_mask]
        mu_train = self.mu[self.k_mask]

        k_Mpc_train = torch.Tensor(k_Mpc_train).to(self.device)
        mu_train = torch.Tensor(mu_train).to(self.device)

        data = [
            {
                key: value
                for key, value in self.training_data[i].items()
                if key in self.emuparams
            }
            for i in range(len(self.training_data))
        ]
        data = self._sort_dict(
            data, self.emuparams
        )  # sort the data by emulator parameters
        data = [list(data[i].values()) for i in range(len(self.training_data))]
        data = np.array(data)

        paramlims = np.concatenate(
            (
                data.min(0).reshape(len(data.min(0)), 1),
                data.max(0).reshape(len(data.max(0)), 1),
            ),
            1,
        )
        self.paramLims = paramlims

        training_label = [
            {
                key: value
                for key, value in self.training_data[i].items()
                if key in ["p3d_Mpc"]
            }
            for i in range(len(self.training_data))
        ]
        training_label = [
            list(training_label[i].values())[0].tolist()
            for i in range(len(self.training_data))
        ]
        training_label = np.array(training_label)
        training_label = training_label[:, self.k_mask]

        self.yscalings = np.median(training_label)

        return k_Mpc_train, mu_train

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
        training_data = (training_data - self.paramLims[:, 0]) / (
            self.paramLims[:, 1] - self.paramLims[:, 0]
        ) - 0.5
        training_data = torch.Tensor(training_data)

        return training_data

    def _get_training_p3d(self, target_space="Arinyo"):
        """
        Given an archive and key_av, it obtains the p1d_Mpc values from the training data and scales it.
        Finally, it returns the scaled values as a torch.Tensor object along with the scaling factor.
        """

        if target_space == "p3d":
            training_label = [
                {
                    key: value
                    for key, value in self.training_data[i].items()
                    if key in ["p3d_Mpc"]
                }
                for i in range(len(self.training_data))
            ]
            training_label = [
                list(training_label[i].values())[0].tolist()
                for i in range(len(self.training_data))
            ]
            training_label = np.array(training_label)
            training_label = training_label[:, self.k_mask]

            # training_label = np.log10(training_label / self.yscalings)
            training_label = torch.Tensor(training_label)

            training_Plin = [
                {
                    key: value
                    for key, value in self.training_data[i].items()
                    if key in ["Plin"]
                }
                for i in range(len(self.training_data))
            ]
            training_Plin = np.array(training_Plin)
            training_Plin = [
                list(training_Plin[i].values())[0].tolist()
                for i in range(len(self.training_data))
            ]

            training_Plin = torch.Tensor(training_Plin)

            return training_label, training_Plin

        if target_space == "Arinyo":
            training_label = np.array(
                [
                    list(self.training_data[i]["Arinyo"].values())
                    for i in range(len(self.training_data))
                ]
            )
            training_label_25 = np.array(
                [
                    list(self.training_data[i]["Arinyo_25"].values())
                    for i in range(len(self.training_data))
                ]
            )
            training_label_75 = np.array(
                [
                    list(self.training_data[i]["Arinyo_75"].values())
                    for i in range(len(self.training_data))
                ]
            )

            training_label_err = training_label_75 - training_label_25

            training_label = torch.Tensor(training_label)
            training_label_err = torch.Tensor(training_label_err)

            return training_label, training_label_err

    def P3D_Mpc(self, linP, k, mu, params, params_std, epoch=None):
        """
        Compute the model for the 3D flux power spectrum in units of Mpc^3.

        Parameters:
            z (float): Redshift.
            k (float): Wavenumber.
            mu (float): Cosine of the angle between the line-of-sight and the wavevector.
            parameters (dict, optional): Additional parameters for the model. Defaults to {}.

        Returns:
            float: Computed value of the 3D flux power spectrum.
        """

        k = torch.Tensor(k)
        # avoid the code to crash (log)
        mu[mu == 0] += self.epsilon
        linP = torch.Tensor(linP).to(self.device)

        bias, beta, q1, q2, av, bv, kvav, kp = (
            params[:, 0].unsqueeze(1),
            params[:, 1].unsqueeze(1),
            params[:, 2].unsqueeze(1),
            params[:, 3].unsqueeze(1),
            params[:, 4].unsqueeze(1),
            params[:, 5].unsqueeze(1),
            params[:, 6].unsqueeze(1),
            params[:, 7].unsqueeze(1),
        )

        #bias = torch.clamp(bias, -2, 2)
        #beta = torch.clamp(beta, 0.01, 3)
        #q1 = torch.clamp(q1, 1e-15, 5)
        #q2 = torch.clamp(q2, 1e-15, 5)
        #kvav = torch.clamp(kvav, 0.1, 5)
        #av = torch.clamp(av, 1e-5, 3)
        #bv = torch.clamp(bv, 0.5, 3)
        #kp = torch.clamp(kp, 5, 40)

        ## CALCULATE P3D

        # model large-scales biasing for delta_flux(k)
        linear_rsd = 1 + beta * mu**2
        lowk_bias = bias * linear_rsd

        # model small-scales correction (D_NL in Arinyo-i-Prats 2015)
        delta2 = (1 / (2 * torch.pi**2)) * k**3 * linP
        nonlin = delta2 * (q1 + q2 * delta2)
        nonvel = ((k**av) / kvav) * mu**bv
        nonpress = (k / kp) ** 2

        D_NL = torch.exp(nonlin * (1 - nonvel) - nonpress)

        p3d_plin =  lowk_bias**2 * D_NL #*linP *

        ## CALCULATE P3D UNCERTAINTY

        dp3d_dbias = (2 / bias) * p3d_plin
        dp3d_dbeta = (2 * mu**2 / (1 + beta * mu**2)) * p3d_plin
        dp3d_dq1 = (delta2 * (1 - nonvel)) * p3d_plin
        dp3d_dq2 = (delta2**2 * (1 - nonvel)) * p3d_plin
        dp3d_dav = (-nonlin * nonvel * torch.log(k)) * p3d_plin
        dp3d_dbv = (-nonlin * nonvel * torch.log(mu)) * p3d_plin
        dp3d_dkvav = (nonlin * nonvel / kvav) * p3d_plin
        dp3d_dkp = (2 * nonpress / kp) * p3d_plin

        dp3d_dparam = torch.concat(
            (
                dp3d_dbias.unsqueeze(1),
                dp3d_dbeta.unsqueeze(1),
                dp3d_dq1.unsqueeze(1),
                dp3d_dq2.unsqueeze(1),
                dp3d_dav.unsqueeze(1),
                dp3d_dbv.unsqueeze(1),
                dp3d_dkvav.unsqueeze(1),
                dp3d_dkp.unsqueeze(1),
            ),
            1,
        )
        # print(dp3d_dparam.shape)

        p3d_plin_std = torch.sqrt(
            (dp3d_dparam**2 * params[:, :, None] ** 2).sum(1)
        )

        return p3d_plin, p3d_plin_std

    def _train_ArinyoSpace(self, loader_train):
        # define optimizer
        if self.adamw:
            optimizer = optim.AdamW(
                self.emulator.parameters(),
                lr=self.lr,
                amsgrad=self.amsgrad,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = optim.Adam(
                self.emulator.parameters(),
                lr=self.lr,
                amsgrad=self.amsgrad,
                weight_decay=self.weight_decay,
            )

        # define scheduler
        scheduler = lr_scheduler.StepLR(
            optimizer, self.step_size, gamma=self.gamma
        )

        self.loss_arr = []
        t0 = time.time()
        for epoch in range(self.nepochs):
            _loss_arr = []
            for datain, bestfit_coeff, bestfit_coeff_err in loader_train:
                optimizer.zero_grad()

                # make predictions for the Arinyo parameters
                coeffs_arinyo, logerr_coeffs_arinyo = self.emulator(
                    datain.to(self.device)
                )

                # limit and measure the predicted uncertainty
                logerr_coeffs_arinyo = torch.clamp(logerr_coeffs_arinyo, -10, 5)
                coeffserr_arinyo = torch.exp(logerr_coeffs_arinyo)

                # loss function in Arinyo space
                log_prob = (
                    (coeffs_arinyo - bestfit_coeff.to(self.device))
                    / torch.sqrt(
                        0.5
                        * (
                            coeffserr_arinyo**2
                            + bestfit_coeff_err.to(self.device) ** 2
                        )
                    )
                ).pow(2) + 2 * torch.sqrt(
                    0.5
                    * (
                        coeffserr_arinyo**2
                        + bestfit_coeff_err.to(self.device) ** 2
                    )
                )  #

                loss = torch.nansum(log_prob, 1)
                loss = torch.nanmean(loss, 0)

                # backpropagation
                loss.backward()
                optimizer.step()

                _loss_arr.append(loss.item())

            self.loss_arr.append(_loss_arr)

            scheduler.step()
        print(f"Emualtor trained in {time.time() - t0} seconds")

    def _train_p3dSpace(self, loader_train, kMpc_train, mu_train):
        if self.rerr_p3d is None:
            raise ValueError(
                "Training in p3d spcae requires the argument rerr_p3d"
            )

        # define optimizer

        if self.adamw:
            optimizer = optim.AdamW(
                self.emulator.parameters(),
                lr=self.lr,
                amsgrad=self.amsgrad,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = optim.Adam(
                self.emulator.parameters(),
                lr=self.lr,
                amsgrad=self.amsgrad,
                weight_decay=self.weight_decay,
            )

        # define scheduler
        scheduler = lr_scheduler.StepLR(
            optimizer, self.step_size, gamma=self.gamma
        )

        self.loss_arr = []
        t0 = time.time()

        self.rerr_p3d = self.rerr_p3d[self.k_mask]

        p3d0_rerr = torch.Tensor(self.rerr_p3d)
        for epoch in range(self.nepochs):
            _loss_arr = []
            for datain, p3d0_plin, linP in loader_train:
                linP = linP[:, self.k_mask]
                p3d0_plin_err = p3d0_rerr[None, :] * p3d0_plin

                optimizer.zero_grad()

                # make predictions for the Arinyo parameters
                coeffs_arinyo, logerr_coeffs_arinyo = self.emulator(
                    datain.to(self.device)
                )

                logerr_coeffs_arinyo = torch.clamp(logerr_coeffs_arinyo, -10, 5)
                coeffserr_arinyo = torch.exp(logerr_coeffs_arinyo)

                p3d_plin, p3dstd = self.P3D_Mpc(
                    linP,
                    kMpc_train,
                    mu_train,
                    coeffs_arinyo,
                    coeffserr_arinyo**2,
                    epoch=epoch,
                )

                # loss function in p3d space
                log_prob = (
                    p3d_plin - p3d0_plin.to(self.device)
                ) ** 2  # / torch.sqrt(0.5*(p3d0_err.to(self.device) ** 2 + p3dstd**2)) + 2 * torch.sqrt(0.5*(p3d0_err.to(self.device) ** 2 + p3dstd**2))

                loss = torch.nansum(log_prob, 1)
                loss = torch.nanmean(loss, 0)

                # backpropagation
                loss.backward()
                optimizer.step()

                _loss_arr.append(loss.item())

            self.loss_arr.append(_loss_arr)

            scheduler.step()
        print(f"Emualtor trained in {time.time() - t0} seconds")

    def _train(self):
        """
        Trains the emulator with given key_list using the archive data.
        Args:
        Returns:None
        """

        # set the emulator
        self.emulator = P3D_emuv0(nhidden=self.nhidden)
        if self.init_xavier:
            self.emulator.apply(init_xavier)
        self.emulator.to(self.device)

        kMpc_train, mu_train = self._obtain_sim_params()
        self.kMpc_train = kMpc_train
        self.mu_train = mu_train
        # get the training data and create dataloader
        training_data = self._get_training_data()

        if self.target_space == "Arinyo":
            training_label, training_label_error = self._get_training_p3d(
                target_space="Arinyo"
            )
            trainig_dataset = TensorDataset(
                training_data, training_label, training_label_error
            )
            loader_train = DataLoader(
                trainig_dataset, batch_size=self.batch_size, shuffle=True
            )

            self._train_ArinyoSpace(loader_train)

        elif self.target_space == "p3d":
            training_label, training_Plin = self._get_training_p3d(
                target_space="p3d"
            )

            trainig_dataset = TensorDataset(
                training_data, training_label, training_Plin
            )
            loader_train = DataLoader(
                trainig_dataset, batch_size=self.batch_size, shuffle=True
            )

            self._train_p3dSpace(loader_train, kMpc_train, mu_train)

        else:
            raise ValueError("Valid target spaces are 'Ariño' and 'p3d'")

    def _save_emulator(self):
        if self.drop_sim != None:
            torch.save(
                self.emulator.state_dict(),
                os.path.join(self.save_path, f"emulator_{self.drop_sim}.pt"),
            )
        else:
            torch.save(self.emulator.state_dict(), self.save_path)

    def _get_p3D_Mpc(self, testing_data, data_emu):
        data = [
            {
                key: value
                for key, value in testing_data[i].items()
                if key in self.emuparams
            }
            for i in range(len(testing_data))
        ]
        data = self._sort_dict(
            data, self.emuparams
        )  # sort the data by emulator parameters
        data = [list(data[i].values()) for i in range(len(testing_data))]
        data = np.array(data)

        test_data = np.array(data)
        test_data = (test_data - self.paramLims[:, 0]) / (
            self.paramLims[:, 1] - self.paramLims[:, 0]
        ) - 0.5
        test_data = torch.Tensor(test_data)

        coeffs_Arinyo, coeffslogerr_Arinyo = self.emulator(
            test_data.to(self.device)
        )  #
        coeffslogerr_Arinyo = torch.clamp(coeffslogerr_Arinyo, -10, 5)
        coeffserr_Arinyo = torch.exp(coeffslogerr_Arinyo) ** 2

        testing_Plin = [
            {
                key: value
                for key, value in testing_data[i].items()
                if key in ["Plin"]
            }
            for i in range(len(testing_data))
        ]

        testing_Plin = np.array(testing_Plin)
        testing_Plin = [
            list(testing_Plin[i].values())[0].tolist()
            for i in range(len(testing_data))
        ]

        testing_Plin = torch.Tensor(testing_Plin)

        testing_Plin = testing_Plin[:, self.k_mask]

        if self.target_space == "Arinyo":
            coeffslogerr_Arinyo = torch.clamp(coeffslogerr_Arinyo, -10, 5)

            coeffs_Arinyo = coeffs_Arinyo.detach().cpu().numpy()

            p3d_predicted = np.zeros(shape=(11, 20, 16))
            p3d_true = np.zeros(shape=(11, 20, 16))

            for k in range(11):
                best_pars = params_numpy2dict(coeffs_Arinyo.reshape(11, 8)[k])
                p3d_predicted[k] = data_emu["model"][k].get_model_3d(best_pars)

            return coeffs_Arinyo, p3d_predicted

        elif self.target_space == "p3d":
            p3d, p3dstd = self.P3D_Mpc(
                testing_Plin,
                self.kMpc_train,
                self.mu_train,
                coeffs_Arinyo,
                coeffserr_Arinyo,
                epoch=None,
            )
            return p3d.detach().cpu().numpy() * testing_Plin.numpy(), p3dstd.detach().cpu().numpy()* testing_Plin.numpy()

    def get_coeff(self, input_emu):
        test_data = np.array(input_emu)
        test_data = (test_data - self.paramLims[:, 0]) / (
            self.paramLims[:, 1] - self.paramLims[:, 0]
        ) - 0.5
        test_data = torch.Tensor(test_data)

        coeffs_Arinyo, coeffslogerr_Arinyo = self.emulator(
            test_data.to(self.device)
        )  #
        coeffslogerr_Arinyo = torch.clamp(coeffslogerr_Arinyo, -10, 5)
        coeffserr_Arinyo = torch.exp(coeffslogerr_Arinyo) ** 2

        coeffs_Arinyo = coeffs_Arinyo.detach().cpu().numpy()

        return coeffs_Arinyo
