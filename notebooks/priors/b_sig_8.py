# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
from getdist import plots, loadMCSamples
import matplotlib.pyplot as plt

# ### Setup fiducial cosmology from DESI DR2 Lya BAO and compute sigma_8

import camb
pars = camb.CAMBparams()
# set background cosmology
pars.set_cosmology(H0=67.36, ombh2=0.02237, omch2=0.12, mnu=0.06)
# set primordial power
pars.InitPower.set_params(As=2.1e-9, ns=0.9649)
# compute sig_8 at z=2.33 and at z=0
pars.set_matter_power(redshifts=[2.33, 0.0]);
results = camb.get_results(pars)
sig_8, sig_8_z0 = np.array(results.get_sigma8())
print(sig_8, sig_8_z0)
# same for f sig_8
f_sig_8, f_sig_8_z0 = results.get_fsigma8()
print(f_sig_8, f_sig_8_z0)

# ### Read chains from the DESI DR2 Lya BAO analysis

# chains_dir='/global/cfs/cdirs/desi/science/lya/y3/loa/final_results/sampler_runs/output_sampler/'
chains_dir='/home/jchaves/Proyectos/projects/lya/data/lya_bao/output_sampler/'

chains_file=chains_dir+'/lyaxlya_lyaxlyb_lyaxqso_lybxqso-final_base'

samples = loadMCSamples(chains_file)

params = samples.getParams()

params

g = plots.getSubplotPlotter(width_inch=6)
g.settings.axes_fontsize = 9
g.settings.legend_fontsize = 11
g.triangle_plot([samples], ['bias_LYA', 'beta_LYA'], legend_labels=[r'DESI DR2 LYA BAO'])

Nsamp,Npar=samples.samples.shape
print(Nsamp, Npar)

new_params_names=['bias_eta_LYA', 'bias_LYA_sig_8', 'bias_eta_LYA_f_sig_8']
# this will collect a dictionary for each sample in the chain
new_params_entries=[]

test=samples.getParamSampleDict(0)

test

for i in range(Nsamp):
    verbose = (i%1000==0)
    verbose = False
    if verbose: print('sample point',i)
    # get point from original chain
    sample = samples.getParamSampleDict(i)
    bias = sample['bias_LYA']
    beta = sample['beta_LYA']
    # compute derived parameters
    new_params = {}
    new_params['bias_eta_LYA'] = beta * bias / f_sig_8 * sig_8
    new_params['bias_LYA_sig_8'] = bias * sig_8
    # beta * bias * sig_8 = b_eta f sig_8
    new_params['bias_eta_LYA_f_sig_8'] = beta * bias * sig_8
    # add them to the list of entries
    new_params_entries.append({k: new_params[k] for k in new_params_names})
    if verbose: print('new params', new_params_entries[-1])

# setup numpy arrays with new parameters
b_eta = np.array([new_params_entries[i]['bias_eta_LYA'] for i in range(Nsamp)])
b_sig_8 = np.array([new_params_entries[i]['bias_LYA_sig_8'] for i in range(Nsamp)])
b_eta_f_sig_8 = np.array([new_params_entries[i]['bias_eta_LYA_f_sig_8'] for i in range(Nsamp)])

# mean and uncertainties
mean_b_eta = np.mean(b_eta)
var_b_eta = np.var(b_eta)
print('b_eta = {} +/- {}'.format(mean_b_eta, np.sqrt(var_b_eta)))
mean_b_sig_8 = np.mean(b_sig_8)
var_b_sig_8 = np.var(b_sig_8)
print('b_sig_8 = {} +/- {}'.format(mean_b_sig_8, np.sqrt(var_b_sig_8)))
mean_b_eta_f_sig_8 = np.mean(b_eta_f_sig_8)
var_b_eta_f_sig_8 = np.var(b_eta_f_sig_8)
print('b_eta_f_sig_8 = {} +/- {}'.format(mean_b_eta_f_sig_8, np.sqrt(var_b_eta_f_sig_8)))

# +
# from P1D we have:
#      b sig_8 = -0.0363 +/- 0.0018 
#      b_eta_f_sig_8 = -0.0518 +/ 0.0012
# but this can't be true... 
# -

np.corrcoef(b_sig_8, b_eta_f_sig_8)

# add derived parameters 
samples.addDerived(b_eta,'bias_eta_LYA',label='b_\\eta')
samples.addDerived(b_sig_8,'bias_LYA_sig_8',label='b_\\alpha \\, \\sigma_8(z)')
samples.addDerived(b_eta_f_sig_8,'bias_eta_LYA_f_sig_8',label='b_\\eta \\, f \\, \\sigma_8(z)')

g = plots.getSubplotPlotter(width_inch=6)
g.settings.axes_fontsize = 9
g.settings.legend_fontsize = 11
g.triangle_plot([samples], ['bias_LYA', 'beta_LYA', 'bias_eta_LYA'], legend_labels=[r'DESI DR2 LYA BAO'])

g = plots.getSubplotPlotter(width_inch=6)
g.settings.axes_fontsize = 9
g.settings.legend_fontsize = 11
g.triangle_plot([samples], ['bias_LYA_sig_8', 'bias_eta_LYA_f_sig_8'], legend_labels=[r'DESI DR2 LYA BAO'])




