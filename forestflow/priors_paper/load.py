import os
import sys
import numpy as np
from getdist import loadMCSamples
from vega import FitResults
from forestflow.priors_paper import set_samples

from lace.cosmo import cosmology


def load_BAO_data():

    zeff = 2.33
    class_planck = cosmology.Cosmology(cosmo_label="Planck18_noBAO")
    planck_sig8 = class_planck.get_sigma8(zeff)
    planck_f = class_planck.get_growth_rate(zeff)

    BAO = {}

    # load baseline

    chains_dir = "/home/jchaves/Proyectos/projects/lya/data/lya_bao/dr1/"
    chains_file_dr1 = chains_dir + "/lyaxlya_lyaxlyb_lyaxqso_lybxqso-baseline_combined"

    chains_dir = "/home/jchaves/Proyectos/projects/lya/data/lya_bao/dr2/"
    chains_file_dr2 = chains_dir + "/lyaxlya_lyaxlyb_lyaxqso_lybxqso-final_base"

    chains_file = [chains_file_dr1, chains_file_dr2]
    labels = ["dr1", "dr2"]

    for ii, file in enumerate(chains_file):
        samples = loadMCSamples(file)

        BAO[labels[ii]] = {}
        BAO[labels[ii]]["bias_delta_sig_8_z"] = (
            samples.getParams().bias_LYA * planck_sig8
        )
        BAO[labels[ii]]["bias_eta_f_sig_8_z"] = (
            samples.getParams().beta_LYA * samples.getParams().bias_LYA * planck_sig8
        )
        BAO[labels[ii]]["bias_delta"] = samples.getParams().bias_LYA
        BAO[labels[ii]]["bias_eta"] = (
            samples.getParams().bias_LYA * samples.getParams().beta_LYA / planck_f
        )
        BAO[labels[ii]]["beta"] = samples.getParams().beta_LYA
        BAO[labels[ii]]["bias_hcd"] = samples.getParams().bias_hcd
        BAO[labels[ii]]["weights"] = samples.weights

    # load high SNR, HCD prior

    # number of realizations to sample the covariance of the fitter
    n = 10000
    basedir = "/home/jchaves/Proyectos/projects/lya/data/lya_bao/fits_andreu/"
    fit_file_dr1 = basedir + "fit_output_dr1_mid_prior.fits"
    fit_file_dr2 = basedir + "fit_output_mid_prior.fits"

    fit_files = [fit_file_dr1, fit_file_dr2]
    labels = ["dr1_hsnr", "dr2_hsnr"]

    for ii, file in enumerate(fit_files):

        samples = FitResults(file)
        params = samples.params
        cov = samples.cov

        # mean vector (ordered consistently with the covariance)
        names = list(params.keys())
        mean = np.array([params[k] for k in names])

        # draw samples
        samples = np.random.multivariate_normal(mean, cov, size=n)

        # dictionary of arrays (each array has length n)
        samples_dict = {name: samples[:, i] for i, name in enumerate(names)}

        BAO[labels[ii]] = {}
        BAO[labels[ii]]["bias_delta_sig_8_z"] = samples_dict["bias_LYA"] * planck_sig8
        BAO[labels[ii]]["bias_eta_f_sig_8_z"] = (
            samples_dict["beta_LYA"] * samples_dict["bias_LYA"] * planck_sig8
        )
        BAO[labels[ii]]["bias_delta"] = samples_dict["bias_LYA"]
        BAO[labels[ii]]["bias_eta"] = (
            samples_dict["bias_LYA"] * samples_dict["beta_LYA"] / planck_f
        )
        BAO[labels[ii]]["beta"] = samples_dict["beta_LYA"]
        BAO[labels[ii]]["bias_hcd"] = samples_dict["bias_hcd"]

    return BAO


def load_p1d_data(zeff=2.33):

    class_planck = cosmology.Cosmology(cosmo_label="Planck18")
    planck_f = class_planck.get_growth_rate(zeff)

    dict_out_all = load_map_igm_p3d()
    dict_out_all["forest_out"]["bias"] = -np.abs(dict_out_all["forest_out"]["bias"])

    P1D = {}

    P1D["bias_delta_sig_8_z"] = (
        dict_out_all["emu_params"]["sig_8"] * dict_out_all["forest_out"]["bias"][:, 1]
    )
    P1D["bias_eta_f_sig_8_z"] = (
        dict_out_all["emu_params"]["sig_8"]
        * dict_out_all["forest_out"]["bias"][:, 1]
        * dict_out_all["forest_out"]["beta"][:, 1]
    )
    P1D["sig_8"] = dict_out_all["emu_params"]["sig_8"]
    P1D["sig_8_z0"] = dict_out_all["emu_params"]["sig_8_z0"]
    P1D["fsig_8"] = P1D["sig_8"] * planck_f

    for par in dict_out_all["forest_out"].keys():
        if par == "bias":
            lab = "bias_delta"
        else:
            lab = par

        P1D[lab] = dict_out_all["forest_out"][par][:, 1]

    P1D["bias_eta"] = P1D["bias_delta"] * P1D["beta"] / planck_f

    P1D["Delta2star"] = dict_out_all["emu_params"]["Delta2star"]
    P1D["nstar"] = dict_out_all["emu_params"]["nstar"]

    return P1D


def load_p1d_chain_for_forestflow():
    try:
        data = np.load(
            "int_data_figs/priors_cosmo_IGM_from_p1d.npy", allow_pickle=True
        ).item()
    except:
        set_samples.set_process_p1d_chain()
        data = np.load(
            "int_data_figs/priors_cosmo_IGM_from_p1d.npy", allow_pickle=True
        ).item()
    return data


def load_map_igm_p3d(zeff=2.33):
    try:
        data = np.load(
            "int_data_figs/arinyo_from_desi_p1d.npy", allow_pickle=True
        ).item()
    except:
        pars_data = load_p1d_chain_for_forestflow()
        set_samples.set_map_igm_p3d(pars_chain)
        data = np.load(
            "int_data_figs/arinyo_from_desi_p1d.npy", allow_pickle=True
        ).item()

    class_planck = cosmology.Cosmology(cosmo_label="Planck18")
    planck_f = class_planck.get_growth_rate(zeff)

    data["forest_out"]["bias_delta"] = -np.abs(data["forest_out"]["bias"])
    data["forest_out"]["bias_eta"] = (
        data["forest_out"]["beta"] * data["forest_out"]["bias_delta"] / planck_f
    )

    return data
