import torch
from torch import nn, optim
import numpy as np


class P3D_emuv0(torch.nn.Module):
    def __init__(self, nhidden):
        super().__init__()
        self.inputlay = torch.nn.Sequential(nn.Linear(6, 10), nn.LeakyReLU(0.1))

        params = np.linspace(10, 100, nhidden)
        modules = []
        for k in range(nhidden - 1):
            modules.append(nn.Linear(int(params[k]), int(params[k + 1])))
            modules.append(nn.LeakyReLU(0.1))
        self.hiddenlay = nn.Sequential(*modules)

        self.means = torch.nn.Sequential(
            nn.Linear(100, 50), nn.LeakyReLU(0.1), nn.Linear(50, 8)
        )
        self.stds = torch.nn.Sequential(
            nn.Linear(100, 50), nn.LeakyReLU(0.1), nn.Linear(50, 8)
        )

    def forward(self, inp):
        x = self.inputlay(inp)
        x = self.hiddenlay(x)
        coeffs_arinyo = self.means(x)
        logerr_coeffs_arinyo = self.stds(x)

        return coeffs_arinyo, logerr_coeffs_arinyo


class P3D_emuv1(torch.nn.Module):
    def __init__(self, nhidden, nwidth1, nwidth2, input_pars=6, output_pars=8):
        super().__init__()

        params = np.linspace(nwidth1, nwidth2, nhidden + 1, dtype=int)

        modules = []
        modules.append(nn.Linear(input_pars, nwidth1))
        modules.append(nn.SELU())
        for k in range(nhidden):
            modules.append(nn.Linear(params[k], params[k + 1]))
            modules.append(nn.SELU())
        modules.append(nn.Linear(nwidth2, output_pars))
        self.hiddenlay = nn.Sequential(*modules)

    def forward(self, inp):
        coeffs_arinyo = self.hiddenlay(inp)
        return coeffs_arinyo
