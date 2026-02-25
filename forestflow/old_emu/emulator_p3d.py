import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys

# LaCE modules
from lace.emulator import utils

from forestflow.emulator_p3d_architecture import P3D_emuv0, P3D_emuv1
from forestflow.input_emu import params_numpy2dict
from forestflow.utils import memorize, _sort_dict

import torch
from torch.utils.data import DataLoader, dataset, TensorDataset
from torch import nn, optim
from torch.optim import lr_scheduler


def init_xavier(m):
    """Initialization of the NN.
    This is quite important for a faster training
    """
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


# def norm_params(xx, direction="R"):
#     pmin = torch.Tensor([1, 0.11, 0, 0.3, 0.01, 0.9, 9, 0])
#     pdiff = torch.Tensor([1.5, 2.6, 2.6, 3.6, 1.1, 1.7, 30, 2.4])

#     if direction == "R":
#         xx = (xx - pmin[None, :]) / pdiff[None, :] - 0.5
#     else:
#         xx = (xx + 0.5) * pdiff[None, :] + pmin[None, :]

#     return xx


class P3DEmulator:
    """A class for training an emulator.

    Args:
        emuparams (dict): A dictionary of emulator parameters.
        kmax_Mpc (float): The maximum k in Mpc^-1 to use for training. Default is 3.5.
        nepochs (int): The number of epochs to train for. Default is 200.
        model_path (str): The path to a pretrained model. Default is None.
        train (bool): Wheather to train the emulator or not. Default True. If False, a model path must is required.
    """

    def __init__(
        self,
        training_data,
        paramList,
        rerr_p3d=None,
        rerr_p1d=None,
        target_space="p3d",
        k3d_max_Mpc=5,
        k1d_max_Mpc=5,
        nhidden=1,
        new_arch=True,
        nwidth1=10,
        nwidth2=100,
        nepochs=100,
        batch_size=100,
        lr=1e-3,
        weight_decay=1e-4,
        gamma=0.1,
        amsgrad=False,
        step_size=75,
        init_xavier=True,
        adamw=False,
        train=True,
        save_path=None,
        model_path=None,
        epsilon=1e-7,
    ):
        self.training_data = training_data
        self.rerr_p3d = rerr_p3d
        self.rerr_p1d = rerr_p1d
        self.emuparams = paramList
        self.k3d_max_Mpc = k3d_max_Mpc
        self.k1d_max_Mpc = k1d_max_Mpc
        self.nepochs = nepochs
        self.step_size = step_size
        self.model_path = model_path
        self.init_xavier = init_xavier
        self.adamw = adamw
        self.new_arch = new_arch
        self.nwidth1 = nwidth1
        self.nwidth2 = nwidth2

        self.nhidden = nhidden
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.amsgrad = amsgrad

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.save_path = save_path
        # self.lace_path = utils.ls_level(os.getcwd(), 1)
        # self.models_dir = os.path.join(self.lace_path, "lya_pk/")

        self.target_space = target_space
        self.epsilon = epsilon

        if train == True:
            self._train()

        """if train == False:
            if self.model_path == None:
                raise Exception("If train==False, model path is required.")

            else:
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

    def _obtain_sim_params(self):
        """
        Obtain simulation parameters.

        Returns:
            k_Mpc (np.ndarray): Simulation k values.
            Nz (int): Number of redshift values.
            Nk (int): Number of k values.
            k_Mpc_train (tensor): k values in the k training range
        """

        data = [
            {
                key: value
                for key, value in self.training_data[i].items()
                if key in self.emuparams
            }
            for i in range(len(self.training_data))
        ]
        data = _sort_dict(data, self.emuparams)
        data = [list(data[i].values()) for i in range(len(self.training_data))]
        data = np.array(data)
        self.paramLims = np.concatenate(
            (
                data.min(0).reshape(len(data.min(0)), 1),
                data.max(0).reshape(len(data.max(0)), 1),
            ),
            1,
        )
        params_norm = (data - self.paramLims[:, 0]) / (
            self.paramLims[:, 1] - self.paramLims[:, 0]
        ) - 0.5
        self.norm_params = torch.Tensor(params_norm).to(self.device)

    def _get_training_p3d(self, target_space="Arinyo"):
        """
        Given an archive and key_av, it obtains the p1d_Mpc values from the training data and scales it.
        Finally, it returns the scaled values as a torch.Tensor object along with the scaling factor.
        """

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

        if (target_space == "p3d") | (target_space == "p3dp1d"):
            # get k3d
            self.k3d_Mpc = self.training_data[0]["k3d_Mpc"]
            self.k3d_mask = (self.k3d_Mpc < self.k3d_max_Mpc) & (
                self.k3d_Mpc > 0
            )
            k3d_Mpc_train = torch.Tensor(self.k3d_Mpc[self.k3d_mask]).to(
                self.device
            )

            # get mu
            self.mu = self.training_data[0]["mu3d"]
            # self.mu[self.mu == 0] += self.epsilon  # so the model does not break
            mu_train = torch.Tensor(self.mu[self.k3d_mask]).to(self.device)

            # get relative error
            self.p3d0_rerr = torch.Tensor(self.rerr_p3d[self.k3d_mask]).to(
                self.device
            )

            if target_space == "p3dp1d":
                # get k1d
                self.k1d_Mpc = self.training_data[0]["k_Mpc"]
                self.k1d_mask = (self.k1d_Mpc < self.k1d_max_Mpc) & (
                    self.k1d_Mpc > 0
                )
                self.k1d_Mpc_train = torch.Tensor(
                    self.k1d_Mpc[self.k1d_mask]
                ).to(self.device)

                # get relative error
                self.p1d0_rerr = torch.Tensor(self.rerr_p1d[self.k1d_mask]).to(
                    self.device
                )

            # p3d
            nelem = len(self.training_data)
            training_p3d = np.zeros(
                (nelem, *self.training_data[0]["p3d_Mpc"].shape)
            )
            training_Plin = np.zeros_like(training_p3d)
            norm3d = self.training_data[0]["k3d_Mpc"] ** 3 / 2 / np.pi**2

            for ii in range(nelem):
                training_p3d[ii] = self.training_data[ii]["p3d_Mpc"] * norm3d
                training_Plin[ii] = self.training_data[ii]["Plin"]
            training_p3d = torch.Tensor(training_p3d[:, self.k3d_mask]).to(
                self.device
            )
            training_Plin = torch.Tensor(training_Plin[:, self.k3d_mask]).to(
                self.device
            )

            if target_space == "p3dp1d":
                self._P1D_stuff()
                # p1d
                training_p1d = np.zeros(
                    (
                        nelem,
                        self.training_data[0]["p1d_Mpc"].shape[0],
                    )
                )
                training_Plin_for_p1d = np.zeros(
                    (nelem, *self.training_data[0]["Plin_for_p1d"].shape)
                )
                norm1d = self.training_data[0]["k_Mpc"] / np.pi
                for ii in range(nelem):
                    training_p1d[ii] = (
                        self.training_data[ii]["p1d_Mpc"] * norm1d
                    )
                    training_Plin_for_p1d[ii] = self.training_data[ii][
                        "Plin_for_p1d"
                    ]

                training_p1d = torch.Tensor(training_p1d[:, self.k1d_mask]).to(
                    self.device
                )
                training_Plin_for_p1d = training_Plin_for_p1d[
                    :, self.k1d_mask, :
                ]
                training_Plin_for_p1d = training_Plin_for_p1d.reshape(nelem, -1)
                training_Plin_for_p1d = torch.Tensor(training_Plin_for_p1d).to(
                    self.device
                )

                training_data = (
                    k3d_Mpc_train,
                    mu_train,
                    training_p3d,
                    training_Plin,
                    training_p1d,
                    training_Plin_for_p1d,
                )
            else:
                training_data = (
                    k3d_Mpc_train,
                    mu_train,
                    training_p3d,
                    training_Plin,
                )
            return training_data

    # xxx
    def P3D_Mpc(self, linP, k, mu, params, params_std):
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

        #         k = torch.Tensor(k)
        #         linP = torch.Tensor(linP).to(self.device)

        # stop model from breaking
        params = torch.abs(params) + self.epsilon

        bias, beta, q1, kvav, av, bv, kp, q2 = (
            params[:, 0].unsqueeze(1),
            params[:, 1].unsqueeze(1),
            params[:, 2].unsqueeze(1),
            params[:, 3].unsqueeze(1),
            params[:, 4].unsqueeze(1),
            params[:, 5].unsqueeze(1),
            params[:, 6].unsqueeze(1),
            params[:, 7].unsqueeze(1),
        )

        ## CALCULATE P3D

        # model large-scales biasing for delta_flux(k)
        linear_rsd = 1 + beta * mu**2
        lowk_bias = bias * linear_rsd

        # model small-scales correction (D_NL in Arinyo-i-Prats 2015)
        delta2 = (1 / (2 * torch.pi**2)) * k**3 * linP
        nonlin = delta2 * (q1 + q2 * delta2)
        nonvel = k**av / kvav * mu**bv
        nonpress = (k / kp) ** 2

        expo = nonlin * (1 - nonvel) - nonpress

        # stop from breaking
        expo = torch.clamp(expo, min=-50, max=50)

        D_NL = torch.exp(expo)

        p3d = delta2 * lowk_bias**2 * D_NL
        p3d_no_norm = linP * lowk_bias**2 * D_NL

        # to be implemented
        p3d_std = 0.1 * p3d

        ## CALCULATE P3D UNCERTAINTY

        # dp3d_dbias = (2 / bias) * p3d
        # dp3d_dbeta = (2 * mu**2 / (1 + beta * mu**2)) * p3d
        # dp3d_dq1 = (delta2 * (1 - nonvel)) * p3d
        # dp3d_dq2 = (delta2**2 * (1 - nonvel)) * p3d
        # dp3d_dav = (-nonlin * nonvel * torch.log(k)) * p3d
        # dp3d_dbv = (-nonlin * nonvel * torch.log(mu)) * p3d
        # dp3d_dkvav = (nonlin * nonvel / kvav) * p3d
        # dp3d_dkp = (2 * nonpress / kp) * p3d

        # dp3d_dparam = torch.concat(
        #     (
        #         dp3d_dbias.unsqueeze(1),
        #         dp3d_dbeta.unsqueeze(1),
        #         dp3d_dq1.unsqueeze(1),
        #         dp3d_dkvav.unsqueeze(1),
        #         dp3d_dav.unsqueeze(1),
        #         dp3d_dbv.unsqueeze(1),
        #         dp3d_dkp.unsqueeze(1),
        #         dp3d_dq2.unsqueeze(1),
        #     ),
        #     1,
        # )

        # p3d_std = torch.sqrt(
        #     (dp3d_dparam**2 * params[:, :, None] ** 2).sum(1)
        # )

        return p3d, p3d_std, p3d_no_norm

    def _P1D_stuff(self, k_perp_min=0.001, k_perp_max=100, n_k_perp=99):
        ln_k_perp = torch.linspace(
            np.log(k_perp_min), np.log(k_perp_max), n_k_perp
        )

        k_perp = torch.exp(ln_k_perp)
        _k = torch.sqrt(self.k1d_Mpc_train[None, :] ** 2 + k_perp[:, None] ** 2)
        _mu = self.k1d_Mpc_train[None, :] / _k
        self._k_p1d = _k.swapaxes(0, 1)
        self._mu_p1d = _mu.swapaxes(0, 1)

        fact = (1 / (2 * torch.pi)) * k_perp[:, None] ** 2
        self.fact_p1d = fact.swapaxes(0, 1)

        self.norm_p1d = self.k1d_Mpc_train / torch.pi
        self.ln_k_perp_p1d = ln_k_perp

    # yyy
    def P1D_Mpc(
        self,
        linP_for_p1d,
        params,
        params_std,
    ):
        _k = self._k_p1d.reshape(-1)
        _mu = self._mu_p1d.reshape(-1)
        _, _, p3d_fix_k_par = self.P3D_Mpc(
            linP_for_p1d,
            _k,
            _mu,
            params,
            params_std,
        )

        p3d_fix_k_par = p3d_fix_k_par.reshape(
            p3d_fix_k_par.shape[0], *self._k_p1d.shape
        )
        p3d_fix_k_par = p3d_fix_k_par * self.fact_p1d[None, :]
        p1d = (
            torch.trapz(p3d_fix_k_par, self.ln_k_perp_p1d, dim=2)
            * self.norm_p1d[None, :]
        )
        # to be implemented
        p1d_std = p1d * 0.1

        return p1d, p1d_std

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
                if self.new_arch:
                    coeffs_arinyo = self.emulator(datain.to(self.device))
                else:
                    coeffs_arinyo, logerr_coeffs_arinyo = self.emulator(
                        datain.to(self.device)
                    )
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
        print(f"Emulator trained in {time.time() - t0} seconds")

    def _train_p3dSpace(self, loader_train, kMpc_train, mu_train):
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
            for training in loader_train:
                if self.target_space == "p3d":
                    datain, p3d0, linP = training
                else:
                    datain, p3d0, linP, p1d0, linP_for_p1d = training
                    p1d0_err = self.p1d0_rerr[None, :] * p1d0
                p3d0_err = self.p3d0_rerr[None, :] * p3d0

                optimizer.zero_grad()

                # make predictions for the Arinyo parameters
                if self.new_arch:
                    coeffs_arinyo = self.emulator(datain.to(self.device))
                    coeffserr_arinyo = 1e-2 * coeffs_arinyo
                else:
                    coeffs_arinyo, logerr_coeffs_arinyo = self.emulator(
                        datain.to(self.device)
                    )
                    # errors on log space
                    logerr_coeffs_arinyo = torch.clamp(
                        logerr_coeffs_arinyo, -10, 5
                    )
                    coeffserr_arinyo = torch.exp(logerr_coeffs_arinyo)

                # get p3d
                p3d, p3dstd, _ = self.P3D_Mpc(
                    linP,
                    kMpc_train,
                    mu_train,
                    coeffs_arinyo,
                    coeffserr_arinyo**2,
                )
                # xxx
                if self.target_space == "p3dp1d":
                    # get p1d
                    p1d, p1dstd = self.P1D_Mpc(
                        linP_for_p1d,
                        coeffs_arinyo,
                        coeffserr_arinyo**2,
                    )

                # xxx
                # loss function in p3d space
                log_prob_p3d = ((p3d - p3d0) / p3d0_err) ** 2
                loss3d = torch.nansum(log_prob_p3d, 1) / p3d0.shape[0]

                if self.target_space == "p3dp1d":
                    log_prob_p1d = ((p1d - p1d0) / p1d0_err) ** 2
                    loss1d = torch.nansum(log_prob_p1d, 1) / p1d0.shape[0]
                    loss = 0.5 * (loss3d + loss1d)
                else:
                    loss = loss3d

                # / torch.sqrt(0.5*(p3d0_err.to(self.device) ** 2 + p3dstd**2)) + 2 * torch.sqrt(0.5*(p3d0_err.to(self.device) ** 2 + p3dstd**2))

                loss = torch.nanmean(torch.sqrt(loss), 0)

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
        if self.new_arch:
            self.emulator = P3D_emuv1(self.nhidden, self.nwidth1, self.nwidth2)
        else:
            self.emulator = P3D_emuv0(nhidden=self.nhidden)

        if self.init_xavier:
            self.emulator.apply(init_xavier)
        else:
            self.emulator.apply(init_zeros)

        self.emulator.to(self.device)

        self._obtain_sim_params()

        training_data = self._get_training_p3d(target_space=self.target_space)

        if self.target_space == "Arinyo":
            trainig_dataset = TensorDataset(self.norm_params, *training_data)
            loader_train = DataLoader(
                trainig_dataset, batch_size=self.batch_size, shuffle=True
            )

            self._train_ArinyoSpace(loader_train)

        elif (self.target_space == "p3d") | (self.target_space == "p3dp1d"):
            kMpc_train, mu_train = training_data[0], training_data[1]
            trainig_dataset = TensorDataset(
                self.norm_params, *training_data[2:]
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

    def get_coeff(self, input_emu):
        test_data = np.array(input_emu)
        test_data = (test_data - self.paramLims[:, 0]) / (
            self.paramLims[:, 1] - self.paramLims[:, 0]
        ) - 0.5
        test_data = torch.Tensor(test_data)

        if self.new_arch:
            coeffs_Arinyo = self.emulator(test_data.to(self.device))
        else:
            coeffs_Arinyo, coeffslogerr_Arinyo = self.emulator(
                test_data.to(self.device)
            )
            coeffslogerr_Arinyo = torch.clamp(coeffslogerr_Arinyo, -10, 5)
            coeffserr_Arinyo = torch.exp(coeffslogerr_Arinyo) ** 2

        coeffs_Arinyo = coeffs_Arinyo.detach().cpu().numpy()

        return coeffs_Arinyo
