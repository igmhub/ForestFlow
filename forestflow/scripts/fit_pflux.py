import numpy as np
import sys, os
from pyDOE2 import lhs

# mamba install -c conda-forge pydoe2
# mamba install -c conda-forge emcee

from lace.emulator import pd_archive
from lace.cosmo import camb_cosmo
from lace.setup_simulations import read_genic
from lya_pk import model_p3d_arinyo, fit_p3d
from lya_pk.input_emu import read_extra_sims, get_flag_out


def get_std_kp1d(ind_sim, ind_tau, ind_z, err_p1d):
    _sim = np.argwhere(err_p1d["u_ind_sim"] == ind_sim)[0, 0]
    _tau = np.argwhere(err_p1d["u_ind_tau"] == ind_tau)[0, 0]
    _z = np.argwhere(err_p1d["u_ind_z"] == ind_z)[0, 0]
    av_pk = err_p1d["p1d_sim_tau_z"][_sim, _tau, _z] * err_p1d["k"] / np.pi
    std_kpk = err_p1d["sm_rel_err"] * av_pk
    return std_kpk


def get_std_kp3d(ind_sim, ind_tau, ind_z, err_p3d, sm_pk=False):
    _sim = np.argwhere(err_p3d["u_ind_sim"] == ind_sim)[0, 0]
    _tau = np.argwhere(err_p3d["u_ind_tau"] == ind_tau)[0, 0]
    _z = np.argwhere(err_p3d["u_ind_z"] == ind_z)[0, 0]
    if sm_pk:
        pk = err_p3d["sm_p3d_sim_tau_z"][_sim, _tau, _z]
    else:
        pk = err_p3d["p3d_sim_tau_z"][_sim, _tau, _z]

    av_pk = pk * err_p3d["k"] ** 3 / 2 / np.pi**2
    std_kpk = err_p3d["sm_rel_err"] * av_pk
    return std_kpk


def get_default_params():
    parameters_q = {
        "bias": -0.12,
        "beta": 1.4,
        "d1_q1": 0.4,
        "d1_kvav": 0.6,
        "d1_av": 0.3,
        "d1_bv": 1.5,
        "d1_kp": 18.0,
    }
    parameters_q2 = {
        "bias": -0.12,
        "beta": 1.4,
        "d1_q1": 0.4,
        "d1_kvav": 0.6,
        "d1_av": 0.3,
        "d1_bv": 1.5,
        "d1_kp": 18.0,
        "d1_q2": 0.2,
    }

    priors_q = {
        "bias": [-1, 0.5],
        "beta": [0, 5.0],
        "d1_q1": [0, 5],
        "d1_kvav": [0.1, 5.0],
        "d1_av": [0, 2],
        "d1_bv": [0, 5],
        "d1_kp": [1, 50],
    }
    priors_q2 = {
        "bias": [-1, 0.5],
        "beta": [0, 5.0],
        "d1_q1": [0, 5],
        "d1_kvav": [0.1, 5.0],
        "d1_av": [0, 2],
        "d1_bv": [0, 5],
        "d1_kp": [1, 50],
        "d1_q2": [0, 5],
    }

    return parameters_q, priors_q, parameters_q2, priors_q2


def get_input_data(data, err_p3d, err_p1d, sim_suite):
    data_dict = {}
    data_dict["units"] = "N"
    data_dict["z"] = np.atleast_1d(data["z"])
    data_dict["k3d"] = data["k3_Mpc"]
    data_dict["mu3d"] = data["mu3"]
    data_dict["p3d"] = data["p3d_Mpc"] * data["k3_Mpc"] ** 3 / 2 / np.pi**2
    std_p3d = get_std_kp3d(
        data["ind_sim"], data["ind_tau"], data["ind_z"], err_p3d, sm_pk=False
    )
    data_dict["std_p3d"] = std_p3d

    data_dict["k1d"] = data["k_Mpc"]
    data_dict["p1d"] = data["p1d_Mpc"] * data["k_Mpc"] / np.pi
    std_p1d = get_std_kp1d(data["ind_sim"], data["ind_tau"], data["ind_z"], err_p1d)
    data_dict["std_p1d"] = std_p1d

    # read cosmology
    assert "LACE_REPO" in os.environ, "export LACE_REPO"
    folder = os.environ["LACE_REPO"] + "/lace/emulator/sim_suites/post_768/"

    if sim_suite == "hypercube":
        flag = "sim_pair_" + str(data["ind_sim"])
    else:
        sim_eq = {
            "100": "sim_pair_h",
            "101": "nu_sim",
            "30": "sim_pair_30",
            "102": "diffSeed",
            "103": "curved_003",
            "104": "P18",
            "105": "running",
        }
        flag = sim_eq[str(data["ind_sim"])]

    genic_fname = folder + flag + "/sim_plus/paramfile.genic"

    sim_cosmo_dict = read_genic.camb_from_genic(genic_fname)
    cosmo = camb_cosmo.get_cosmology_from_dictionary(sim_cosmo_dict)

    # get model
    camb_results = camb_cosmo.get_camb_results(
        cosmo, zs=data_dict["z"], camb_kmax_Mpc=200
    )
    model = model_p3d_arinyo.ArinyoModel(cosmo, data_dict["z"][0], camb_results)
    linp = (
        model.linP_Mpc(z=data_dict["z"][0], k_Mpc=data_dict["k3d"])
        * data_dict["k3d"] ** 3
        / 2
        / np.pi**2
    )

    return data_dict, model, linp


def main():
    # pass sim to use
    ind_sim = int(sys.argv[1])
    test = False

    if test:
        nwalkers = 20
        nsteps = 1
        nburn = 0
    else:
        nwalkers = 500
        nsteps = 1500
        nburn = 750

    # sim_suite = 'hypercube'
    sim_suite = "special"

    # read data sims
    if sim_suite == "hypercube":
        archive = pd_archive.archivePD(nsamples=30)
        archive.average_over_samples(flag="all")
    else:
        archive = read_extra_sims()

    o_ind_sim = archive.data_av_all[ind_sim]["ind_sim"]
    o_ind_tau = archive.data_av_all[ind_sim]["scale_tau"]
    o_ind_z = archive.data_av_all[ind_sim]["z"]
    print(
        "Analyzing simulation "
        + str(ind_sim)
        + " out of "
        + str(len(archive.data_av_all))
    )

    # read errors (from compute__Pflux_variance.ipynb)
    # the following two lines do not work when submitting HTCondor job
    # assert "LACE_PK_REPO" in os.environ, "export LACE_PK_REPO"
    # folder = os.environ["LACE_PK_REPO"] + "/lace_pk/data/"
    # I need to specify the target directory by hand
    folder = "/data/desi/scratch/jchavesm/lya_pk/data/"
    if sim_suite == "hypercube":
        err_p1d = np.load(folder + "p1d_4_fit.npz")
        err_p3d = np.load(folder + "p3d_4_fit.npz")
    else:
        err_p1d = np.load(folder + "p1d_4_fit_extended.npz")
        err_p3d = np.load(folder + "p3d_4_fit_extended.npz")

    # fit options
    kmax_3d = 5
    noise_3d = 0.075
    kmax_1d = 5
    noise_1d = 0.01
    fit_type = "both"
    use_q2 = True

    # get initial parameters for fit
    _ = get_default_params()
    parameters_q, priors_q, parameters_q2, priors_q2 = _
    if use_q2:
        parameters = parameters_q2
        priors = priors_q2
    else:
        parameters = parameters_q
        priors = priors_q

    # folder and name of output file
    folder_out_file = "/data/desi/scratch/jchavesm/p3d_fits_new/"
    out_file = get_flag_out(
        o_ind_sim,
        o_ind_tau,
        o_ind_z,
        kmax_3d,
        noise_3d,
        kmax_1d,
        noise_1d,
    )

    # get input data
    data_dict, model, linp = get_input_data(
        archive.data_av_all[ind_sim], err_p3d, err_p1d, sim_suite
    )

    # set fitting model
    fit = fit_p3d.FitPk(
        data_dict,
        model,
        fit_type=fit_type,
        k3d_max=kmax_3d,
        k1d_max=kmax_1d,
        noise_3d=noise_3d,
        noise_1d=noise_1d,
        priors=priors,
    )

    ## fit data ##
    # get initial solution for sampler
    # we extract 5 samples from the priors
    parameter_names = list(parameters.keys())
    nsam = 5
    seed = 0
    nparams = len(parameter_names)
    design = lhs(
        nparams,
        samples=nsam,
        criterion="c",
        random_state=seed,
    )

    for ii in range(nparams):
        buse = priors[parameter_names[ii]]
        design[:, ii] = (buse[1] - buse[0]) * design[:, ii] + buse[0]

    chia = 1e10
    for ii in range(design.shape[0]):
        pp = {}
        for jj in range(nparams):
            pp[parameter_names[jj]] = design[ii, jj]
        results, _best_fit_params = fit.maximize_likelihood(pp)
        chi2 = fit.get_chi2(_best_fit_params)
        if chi2 < chia:
            chia = chi2
            best_fit_params = _best_fit_params.copy()

    # and values that we know work well for some samples
    results, _best_fit_params = fit.maximize_likelihood(parameters)
    chi2 = fit.get_chi2(_best_fit_params)
    if chi2 < chia:
        chia = chi2
        best_fit_params = _best_fit_params.copy()
    # the output is chia and best_fit_params
    print("Initial chi2", chia)
    print("and best_params", best_fit_params)

    # using best_fit_params as input, we run the chain
    lnprob, chain = fit.explore_likelihood(
        best_fit_params,
        seed=0,
        nwalkers=nwalkers,
        nsteps=nsteps,
        nburn=nburn,
        plot=False,
        attraction=0.4,
    )

    bp1 = chain[np.argmax(lnprob)]
    pars = {}
    for ii in range(len(fit.names)):
        pars[fit.names[ii]] = bp1[ii]
    chib = fit.get_chi2(pars)

    # select best-fitting parameters
    if chia < chib:
        chi2 = chia
        bp = np.array(list(best_fit_params.values()))
    else:
        chi2 = chib
        bp = bp1
        best_fit_params = pars

    # check if we can improve the fit by starting at the MLE
    results, _best_fit_params = fit.maximize_likelihood(pars)
    chi2a = fit.get_chi2(_best_fit_params)
    if chi2a < chi2:
        chi2 = chi2a
        best_fit_params = _best_fit_params.copy()
        bp = np.array(list(best_fit_params.values()))

    print("chi2", chi2)
    print("best_params", bp)

    # store chains to file
    np.savez(
        folder_out_file + out_file,
        chain=chain,
        lnprob=lnprob,
        chi2=chi2,
        best_params=bp,
    )


if __name__ == "__main__":
    main()
