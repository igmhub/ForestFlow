# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Vega
#     language: python
#     name: vega
# ---

# %% [markdown]
# # BAO fits for different HCD biases (high-snr spectra only)

# %%
import numpy as np
import matplotlib.pyplot as plt
from vega import VegaInterface, Wedge, FitResults
from vega.analysis import Analysis
from astropy.io import fits
from getdist import MCSamples, plots

# %% [markdown]
# ### Reproduce the official results

# %%
# official results (data split in the DR2 Lya BAO paper)
official_dir = '/global/cfs/cdirs/desi/science/lya/y3/loa/validation_tests/3-0-0-0/data_splits/high_snr/'
official_fname = official_dir + '/fits/results/fit_output.fits'
official_fit = FitResults(official_fname)

# %%
# my version of that fit (exactly same configuration)
basedir = '/global/cfs/cdirs/desicollab/users/font/high_snr_fits/'
my_fit = FitResults(basedir + 'results/fit_output.fits')

# %%
for fit in [official_fit, my_fit]:
    print('b_hcd = {:.5f} , b_Lya = {:.5f}, chi2 = {:.3f}'.format(fit.params['bias_hcd'], fit.params['bias_LYA'], fit.chisq))

# %% [markdown]
# ### Impact of small variations (fixing beta_hcd, L_hcd)

# %%
b_hcd_free = FitResults(basedir + 'results/fit_output_b_hcd_free.fits')

# %%
for fit in [my_fit, b_hcd_free]:
    print('b_hcd = {:.5f} , b_Lya = {:.5f}, chi2 = {:.3f}'.format(fit.params['bias_hcd'], fit.params['bias_LYA'], fit.chisq))

# %% [markdown]
# ### New fits for fixed b_hcd

# %%
fits = {}
#for key in ['0', '0.01', '0.016', '0.02', '0.03', '0.04', '0.05', '0.06', '0.08', '0.1']:
for key in ['0', '0.01', '0.016', '0.02', '0.03', '0.04', '0.05', '0.06']:
#for key in ['0', '0.01', '0.02', '0.03', '0.04', '0.05', '0.06']:
    fits[key] = FitResults(basedir+'/results/fit_output_b_hcd_{}.fits'.format(key))

# %% [markdown]
# ### Compare b_hcd=0 with free b_hcd (with best-fit 0)

# %%
for fit in [b_hcd_free, fits['0']]:
    print('b_Lya = {:.5f}, chi2 = {:.3f}'.format(fit.params['bias_LYA'], fit.chisq))

# %% [markdown]
# ### Study the impact of b_hcd on b_Lya, beta_Lya and chi2

# %%
for key, fit in fits.items():
    print('b_hcd = -{} , b_Lya = {:.3f}, beta_Lya = {:.3f}, chi2 = {:.1f}'.format(
            key, fit.params['bias_LYA'], fit.params['beta_LYA'], fit.chisq))

# %%
b_hcd = np.array([-float(key) for key in fits.keys()])
b_Lya = np.array([fit.params['bias_LYA'] for fit in fits.values()])
beta_Lya = np.array([fit.params['beta_LYA'] for fit in fits.values()])
chi2 = np.array([fit.chisq for fit in fits.values()])
minchi2 = np.min(chi2)

# %%
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel(r'$b_{\rm HCD}$')
ax1.set_ylabel(r'$b_{\rm LYA}$', color=color)
ax1.plot(b_hcd, b_Lya, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel(r'$\chi^2$', color=color)  # we already handled the x-label with ax1
ax2.plot(b_hcd, chi2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

# %%
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel(r'$b_{\rm HCD}$')
ax1.set_ylabel(r'$b_{\rm LYA}$', color=color)
ax1.plot(b_hcd, b_Lya, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel(r'$\beta_{\rm LYA}$', color=color)  # we already handled the x-label with ax1
ax2.plot(b_hcd, beta_Lya, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

# %% [markdown]
# ### Contours of bias/beta for the best-guess values of bias_HCD = -0.016

# %%
keys = ['0','0.01','0.016','0.02']
labels = ['b_hcd = -{}'.format(key) for key in keys]
chains = [fits[key].chain for key in keys]

# %%
g = plots.getSubplotPlotter(width_inch=6)
g.plot_2d(chains, ['bias_LYA', 'beta_LYA'])
g.add_legend(labels, legend_loc='upper left', colored_text=True)

# %% [markdown]
# ### Results when using a prior on b_hcd = -0.017 +/- 0.002

# %%
b_hcd_prior = FitResults(basedir + 'results/fit_output_b_hcd_prior.fits')

# %%
g = plots.getSubplotPlotter(width_inch=6)
g.plot_2d([b_hcd_free.chain, b_hcd_prior.chain], ['bias_hcd', 'bias_LYA', 'beta_LYA'])
g.add_legend(['free HCD bias', 'prior b_hcd = -0.017+/-0.002'], legend_loc='upper left', colored_text=True)

# %%
g = plots.getSubplotPlotter(width_inch=6)
g.triangle_plot([b_hcd_free.chain, b_hcd_prior.chain],
                ['bias_hcd', 'bias_LYA', 'beta_LYA'],
                legend_labels=['free HCD bias', 'prior b_hcd = -0.017+/-0.002'])

# %%
g = plots.getSubplotPlotter(width_inch=6)
g.triangle_plot([b_hcd_free.chain, b_hcd_prior.chain],
                ['bias_hcd', 'bias_LYA', 'beta_LYA'],
                legend_labels=['free HCD bias', 'prior b_hcd = -0.017+/-0.002'])

# %%

# %%
