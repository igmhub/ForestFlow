import numpy as np
from getdist import loadMCSamples
from vega import FitResults


def load_BAO_data(sig8, f):

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
        BAO[labels[ii]]["bias_delta_sig_8_z"] = samples.getParams().bias_LYA * sig8
        BAO[labels[ii]]["bias_eta_f_sig_8_z"] = (
            samples.getParams().beta_LYA * samples.getParams().bias_LYA * sig8
        )
        BAO[labels[ii]]["bias_delta"] = samples.getParams().bias_LYA
        BAO[labels[ii]]["bias_eta"] = (
            samples.getParams().bias_LYA * samples.getParams().beta_LYA / f
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
        BAO[labels[ii]]["bias_delta_sig_8_z"] = samples_dict["bias_LYA"] * sig8
        BAO[labels[ii]]["bias_eta_f_sig_8_z"] = (
            samples_dict["beta_LYA"] * samples_dict["bias_LYA"] * sig8
        )
        BAO[labels[ii]]["bias_delta"] = samples_dict["bias_LYA"]
        BAO[labels[ii]]["bias_eta"] = (
            samples_dict["bias_LYA"] * samples_dict["beta_LYA"] / f
        )
        BAO[labels[ii]]["beta"] = samples_dict["beta_LYA"]
        BAO[labels[ii]]["bias_hcd"] = samples_dict["bias_hcd"]

    return BAO


def load_p1d_data(f):
    dict_out_all = np.load(
        "int_data_figs/arinyo_from_desi_p1d.npy", allow_pickle=True
    ).item()
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
    P1D["fsig_8"] = P1D["sig_8"] * f

    for par in dict_out_all["forest_out"].keys():
        if par == "bias":
            lab = "bias_delta"
        else:
            lab = par

        P1D[lab] = dict_out_all["forest_out"][par][:, 1]

    P1D["bias_eta"] = P1D["bias_delta"] * P1D["beta"] / f

    P1D["Delta2star"] = dict_out_all["emu_params"]["Delta2star"]
    P1D["nstar"] = dict_out_all["emu_params"]["nstar"]

    return P1D
