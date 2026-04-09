import numpy as np
from getdist import MCSamples

from cup1d.likelihood.pipeline import Pipeline
from lace.cosmo import cosmology, rescale_cosmology
from forestflow.P3D_cINN import P3DEmulator


def set_getdist_samples(BAO, P1D):

    names = [
        "b_delta_sigma8",
        "b_eta_f_sigma8",
        "bias_delta",
        "bias_eta",
        "beta",
        "bias_hcd",
    ]
    labels = [
        r"b_\delta \sigma_8",
        r"b_{\eta} f \sigma_8",
        r"b_\delta",
        r"b_\eta",
        r"\beta",
        r"b_\mathrm{HCD}",
    ]
    label_sample = {
        "dr1": "BAO DR1 low SNR",
        "dr2": "BAO DR2 low SNR",
        "dr1_hsnr": "BAO DR1",
        "dr2_hsnr": "BAO DR2",
    }
    all_samples = {}
    for key in BAO.keys():
        vstack = np.vstack(
            [
                BAO[key]["bias_delta_sig_8_z"],
                BAO[key]["bias_eta_f_sig_8_z"],
                BAO[key]["bias_delta"],
                BAO[key]["bias_eta"],
                BAO[key]["beta"],
                BAO[key]["bias_hcd"],
            ]
        ).T

        if "weights" in BAO[key]:
            all_samples[key] = MCSamples(
                samples=vstack.copy(),
                names=names,
                labels=labels,
                weights=BAO[key]["weights"].copy(),
                label=label_sample[key],
            )
        else:
            all_samples[key] = MCSamples(
                samples=vstack.copy(),
                names=names,
                labels=labels,
                label=label_sample[key],
            )

    p1d = np.vstack(
        [
            P1D["bias_delta_sig_8_z"],
            P1D["bias_eta_f_sig_8_z"],
            P1D["sig_8"],
            P1D["sig_8_z0"],
            P1D["fsig_8"],
            P1D["bias_delta"],
            P1D["beta"],
            P1D["q1"],
            P1D["kvav"],
            P1D["av"],
            P1D["bv"],
            P1D["kp"],
            P1D["q2"],
            P1D["bias_eta"],
            P1D["Delta2star"],
            P1D["nstar"],
        ]
    ).T
    names = [
        "b_delta_sigma8",
        "b_eta_f_sigma8",
        "sigma8",
        "sigma8_z0",
        "fsigma8",
        "bias_delta",
        "beta",
        "q1",
        "kvav",
        "av",
        "bv",
        "kp",
        "q2",
        "bias_eta",
        "Delta2star",
        "nstar",
    ]
    labels = [
        r"b_\delta \sigma_8",
        r"b_\eta f \sigma_8",
        r"\sigma_8(z=2.33)",
        r"\sigma_8(z=0)",
        r"f \sigma_8",
        r"b_\delta",
        r"\beta",
        r"q_1",
        r"k_\mathrm{vav}",
        r"a_\mathrm{v}",
        r"b_\mathrm{v}",
        r"k_\mathrm{p}",
        r"q_2",
        r"b_\eta",
        r"\Delta^2_\star",
        r"n_\star",
    ]

    label_sample = {"p1d": "P1D"}

    for key in ["p1d"]:
        all_samples[key] = MCSamples(
            samples=p1d.copy(),
            names=names,
            labels=labels,
            label=label_sample[key],
        )

    return all_samples


def set_process_p1d_chain(nn=1000, seed=12345, store_p1d=False, zeff=2.33):

    pip = Pipeline()
    # local
    base = "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1"
    folder = os.path.join(base, "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7/")
    # nersc
    # folder = "/global/cfs/cdirs/desi/users/jjchaves/p1d"

    fname = os.path.join(folder, "chain.npy")
    chain = np.array(np.load(fname))
    chain = chain.reshape(-1, 53)

    blobs = np.load(os.path.join(folder, "blobs.npy"), allow_pickle=True)
    d2star = blobs["Delta2_star"].reshape(-1)
    nstar = blobs["n_star"].reshape(-1)

    zs = np.array(
        [2.2, zeff, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2]
    )  # adding 2.33 for the priors

    # get a random sample
    rng = np.random.default_rng(seed=seed)
    ind = rng.permutation(np.arange(chain.shape[0]))[:nn]
    pars_chain = {}

    ## create holder for p1d chain
    pars_chain["z"] = zs

    # cosmo (for Arinyo model)
    pars_chain["As"] = np.zeros((ind.shape[0]))
    pars_chain["ns"] = np.zeros((ind.shape[0]))
    pars_chain["Delta2star"] = d2star[ind]
    pars_chain["nstar"] = nstar[ind]

    # cosmo for this project
    pars_chain["sig_8"] = np.zeros((ind.shape[0]))
    pars_chain["sig_8_z0"] = np.zeros((ind.shape[0]))
    pars_chain["f_sig_8"] = np.zeros((ind.shape[0]))

    # input ForestFlow
    pars_chain["Delta2_p"] = np.zeros((ind.shape[0], zs.shape[0]))
    pars_chain["n_p"] = np.zeros((ind.shape[0], zs.shape[0]))
    pars_chain["mF"] = np.zeros((ind.shape[0], zs.shape[0]))
    pars_chain["gamma"] = np.zeros((ind.shape[0], zs.shape[0]))
    pars_chain["sigT_Mpc"] = np.zeros((ind.shape[0], zs.shape[0]))
    pars_chain["kF_Mpc"] = np.zeros((ind.shape[0], zs.shape[0]))

    if store_p1d:
        # store p1d w/ and w/o contaminants
        ii = 0
        p1d = pip.fitter.like.get_p1d_kms(
            pip.fitter.like.data.z, pip.fitter.like.data.k_kms, chain[ind[ii], :]
        )
        pars_chain["k_kms"] = np.zeros((zs.shape[0], len(p1d[0][-1])))
        pars_chain["p1d"] = np.zeros((ind.shape[0], zs.shape[0], len(p1d[0][-1])))
        pars_chain["p1d_nocont"] = np.zeros(
            (ind.shape[0], zs.shape[0], len(p1d[0][-1]))
        )

        # same k at 2.33 as at 2.2
        for jj in range(2):
            nelem = len(p1d[0][0])
            pars_chain["k_kms"][jj, :nelem] = pip.fitter.like.data.k_kms[0]

        # the others the same
        for jj in range(2, zs.shape[0]):
            nelem = len(p1d[0][jj - 1])
            pars_chain["k_kms"][jj, :nelem] = pip.fitter.like.data.k_kms[jj - 1]

    ## fill the holder
    # set cosmology
    class_planck = cosmology.Cosmology(cosmo_label="Planck18")
    planck_f = class_planck.get_growth_rate(zeff)
    dkms_dMpc_zs = class_planck.get_dkms_dMpc(pars_chain["z"])

    for ii in range(ind.shape[0]):
        if ii % 100 == 0:
            print(ii)

        chain_params = pip.fitter.like.parameters_from_sampling_point(chain[ind[ii], :])

        pars_chain["mF"][ii] = pip.fitter.like.theory.model_igm.models[
            "F_model"
        ].get_mean_flux(zs, like_params=chain_params)

        pars_chain["gamma"][ii] = pip.fitter.like.theory.model_igm.models[
            "T_model"
        ].get_gamma(zs, like_params=chain_params)

        sigT_kms = pip.fitter.like.theory.model_igm.models["T_model"].get_sigT_kms(
            zs, like_params=chain_params
        )
        pars_chain["sigT_Mpc"][ii] = sigT_kms / dkms_dMpc_zs

        kF_kms = pip.fitter.like.theory.model_igm.models["P_model"].get_kF_kms(
            zs, like_params=chain_params
        )
        pars_chain["kF_Mpc"][ii] = kF_kms * dkms_dMpc_zs

        pars_chain["As"][ii] = chain_params[0].value_from_cube(chain[ind[ii], 0])
        pars_chain["ns"][ii] = chain_params[1].value_from_cube(chain[ind[ii], 1])

        class_new = rescale_cosmology.RescaledCosmology(
            fid_cosmo=class_planck,
            new_params_dict={"As": pars_chain["As"][ii], "ns": pars_chain["ns"][ii]},
        )

        pars_chain["sig_8"][ii] = class_new.get_sigma8(zeff)
        pars_chain["sig_8_z0"][ii] = class_new.get_sigma8(0)
        pars_chain["f_sig_8"][ii] = planck_f * pars_chain["sig_8"][ii]

        for jj in range(zs.shape[0]):
            linP_params = class_planck.get_linP_Mpc_params(
                pars_chain["z"][jj], kp_Mpc=0.7
            )
            pars_chain["Delta2_p"][ii, jj] = linP_params["Delta2_p"]
            pars_chain["n_p"][ii, jj] = linP_params["n_p"]

        if store_p1d:
            p1d = pip.fitter.like.get_p1d_kms(
                pip.fitter.like.data.z, pip.fitter.like.data.k_kms, chain[ind[ii], :]
            )
            p1d_no = pip.fitter.like.get_p1d_kms(
                pip.fitter.like.data.z,
                pip.fitter.like.data.k_kms,
                chain[ind[ii], :],
                no_contaminants=True,
            )
            # same k at 2.33 as at 2.2
            for jj in range(2):
                nelem = len(p1d[0][0])
                pars_chain["p1d"][ii, jj, :nelem] = p1d[0][0]
                pars_chain["p1d_nocont"][ii, jj, :nelem] = p1d_no[0][0]

            # the others the same
            for jj in range(2, zs.shape[0]):
                nelem = len(p1d[0][jj - 1])
                pars_chain["p1d"][ii, jj, :nelem] = p1d[0][jj - 1]
                pars_chain["p1d_nocont"][ii, jj, :nelem] = p1d_no[0][jj - 1]

    np.save("int_data_figs/inter_chain.npy", pars_chain)

    dict_save_file = {"zs": pars_chain["z"]}
    for par in pars_chain.keys():
        if par not in ["k_kms", "p1d", "p1d_nocont", "z", "As", "ns"]:
            dict_save_file[par] = pars_chain[par]

    np.save("int_data_figs/priors_cosmo_IGM_from_p1d.npy", dict_save_file)

    return


def set_cmbspa_sig8z(nsamples=200, nz=20):

    # CMB-SPA Table 1 https://arxiv.org/abs/2506.20707v1
    cosmo_full_cmbspa = {
        "H0": 67.24,
        "err_H0": 0.35,
        "mnu": 0.06,
        "omch2": 0.12009,
        "err_omch2": 0.00086,
        "ombh2": 0.022381,
        "err_ombh2": 0.000093,
        "omk": 0,
        "log_As": 3.0479,
        "err_log_As": 0.0099,
        "ns": 0.9684,
        "err_ns": 0.0030,
        "nrun": 0.0,
        "pivot_scalar": 0.05,
        "w": -1.0,
    }

    cmb_spa_samples = sample_cosmo_dict(cosmo_full_cmbspa, n_samples=nsamples)
    zplot = np.linspace(0, 3, nz)
    sig8 = np.zeros((nsamples, nz))
    f = np.zeros((nsamples, nz))

    for ii, dict_cosmo in enumerate(cmb_spa_samples):
        class_cosmo = cosmology.Cosmology(cosmo_params_dict=dict_cosmo)
        sig8[ii] = class_cosmo.get_sigma8(zplot)
        f[ii] = class_cosmo.get_growth_rate(zplot)

    dict_out = {"z": zplot, "sig8": sig8, "f": f}
    np.save("int_data_figs/sig8_cmb_spa.npy", dict_out)

    return


def sample_cosmo_dict(base, n_samples=1, rng=None):
    rng = np.random.default_rng(rng)

    # parameters with errors
    params = {k: v for k, v in base.items() if not k.startswith("err_")}
    errs = {k[4:]: v for k, v in base.items() if k.startswith("err_")}

    samples = []
    for _ in range(n_samples):
        d = {}
        # set parameters without errors
        for p in params:
            if p != "log_As":
                d[p] = params[p]
        # for those with errors, sample a normal distribution
        for p, err in errs.items():
            _par = rng.normal(base[p], err)
            if p == "log_As":
                d["As"] = np.exp(_par) * 1e-10
            else:
                d[p] = _par
        samples.append(d)

    return samples


def set_desifs_sig8z(nsamples=5000):

    # DESI FS DR1  redshifts https://arxiv.org/abs/2411.12021

    # Table1
    desi_fs_zeff = np.array([0.295, 0.510, 0.706, 0.930, 1.317, 1.491])

    datasets = ["BGS", "LRG1", "LRG2", "LRG3", "ELG2", "QSO"]

    Omega_m = np.array(
        [
            (0.284, 0.024, 0.024),
            (0.307, 0.018, 0.020),
            (0.287, 0.020, 0.020),
            (0.304, 0.023, 0.023),
            (0.310, 0.027, 0.034),
            (0.314, 0.029, 0.039),
        ]
    )

    H0 = np.array(
        [
            (68.3, 2.4, 2.4),
            (68.8, 1.3, 1.5),
            (70.9, 1.6, 1.6),
            (66.8, 1.2, 1.2),
            (68.5, 2.1, 2.1),
            (69.4, 3.1, 3.1),
        ]
    )

    ln10As = np.array(
        [
            (2.73, 0.40, 0.40),
            (3.05, 0.22, 0.22),
            (3.17, 0.21, 0.24),
            (3.12, 0.22, 0.22),
            (2.86, 0.17, 0.19),
            (3.26, 0.18, 0.18),
        ]
    )

    ns = np.array(
        [
            (0.962, 0.040),
            (0.964, 0.039),
            (0.979, 0.038),
            (0.972, 0.038),
            (0.969, 0.039),
            (0.976, 0.038),
        ]
    )

    cosmo_full_desi = {}

    for ii in range(len(datasets)):
        cosmo_full_desi[datasets[ii]] = {
            "H0": H0[ii][0],
            "err_H0": np.mean(H0[ii][1:]),
            "mnu": 0.0,
            "omch2": Omega_m[ii][0] * (H0[ii][0] / 100) ** 2,
            "err_omch2": np.mean(Omega_m[ii][1:]) * (H0[ii][0] / 100) ** 2,
            "ombh2": 0.02218,
            "err_ombh2": 0.00055,  # Table4
            "omk": 0,
            "log_As": ln10As[ii][0],
            "err_log_As": np.mean(ln10As[ii][1:]),
            "ns": ns[ii][0],
            "err_ns": ns[ii][1],
            "nrun": 0.0,
            "pivot_scalar": 0.05,
            "w": -1.0,
        }
        print(datasets[ii], cosmo_full_desi[datasets[ii]])

    nmods = len(datasets)
    sig8 = np.zeros((nsamples, nmods))
    sig8_z0 = np.zeros((nsamples, nmods))
    f = np.zeros((nsamples, nmods))

    for jj in range(nmods):
        print(datasets[jj])
        desi_samples = sample_cosmo_dict(
            cosmo_full_desi[datasets[jj]], n_samples=nsamples
        )

        for ii, dict_cosmo in enumerate(desi_samples):
            if ii % 100 == 0:
                print(ii, nsamples)
            class_cosmo = cosmology.Cosmology(cosmo_params_dict=dict_cosmo)
            sig8[ii, jj] = class_cosmo.get_sigma8(desi_fs_zeff[jj])
            f[ii, jj] = class_cosmo.get_growth_rate(desi_fs_zeff[jj])
            sig8_z0[ii, jj] = class_cosmo.get_sigma8(0)

    dict_out = {"sig8": sig8, "f": f, "datasets": datasets, "zeff": desi_fs_zeff}
    np.save("int_data_figs/sig8_desi.npy", dict_out)

    return


def set_map_igm_p3d(pars_chain, store_p1d=False):

    import forestflow
    from forestflow.model_p3d_arinyo import ArinyoModel

    path_repo = forestflow.__path__[0]

    emulator = P3DEmulator(
        model_path=path_repo + "/data/emulator_models/forest_mpg",
    )

    if store_p1d:
        class_planck = cosmology.Cosmology(cosmo_label="Planck18")
        dkms_dMpc_zs = class_planck.get_dkms_dMpc(pars_chain["zs"])

    # initiate Arinyo model, needed to compute P1D
    fid_cosmo = {
        "H0": 67.66,
        "mnu": 0,
        "omch2": 0.119,
        "ombh2": 0.0224,
        "omk": 0,
        "As": 2.105e-09,
        "ns": 0.9665,
        "nrun": 0.0,
        "pivot_scalar": 0.05,
        "w": -1.0,
    }
    model_Arinyo = ArinyoModel(fid_cosmo)

    pars = pars_chain.keys()

    out_ari = {}
    for par in emulator.Arinyo_params:
        out_ari[par] = np.zeros_like(pars_chain["mF"])

    for ii in range(pars_chain["mF"].shape[0]):
        if ii % 100 == 0:
            print(ii)

        new_cosmo = {
            "H0": 67.66,
            "mnu": 0,
            "omch2": 0.119,
            "ombh2": 0.0224,
            "omk": 0,
            # 'As': 2.105e-09,
            "As": pars_chain["As"][ii],
            # 'ns': 0.9665,
            "ns": pars_chain["ns"][ii],
            "nrun": 0.0,
            "pivot_scalar": 0.05,
            "w": -1.0,
        }

        # redshift
        for jj in range(pars_chain["mF"].shape[1]):
            input_emu = {}
            for par in pars:
                if par not in ["Delta2_p", "n_p", "mF", "gamma", "sigT_Mpc", "kF_Mpc"]:
                    continue
                input_emu[par] = pars_chain[par][ii, jj]

            par_ari = emulator.predict_Arinyos(emu_params=input_emu)
            for par in par_ari:
                out_ari[par][ii, jj] = par_ari[par]

            if store_p1d:
                _ = pars_chain["k_kms"][jj] != 0
                # we can use this one because we only chage the priordial power
                k_Mpc = pars_chain["k_kms"][jj, _] * dkms_dMpc_zs[jj]
                P1D_Mpc = model_Arinyo.P1D_Mpc(
                    pars_chain["z"][jj],
                    k_Mpc,
                    par_ari,
                    cosmo_new=new_cosmo,
                )
                out_ari["p1d"][ii, jj, : np.sum(_)] = P1D_Mpc * dkms_dMpc_zs[jj]

    dict_out_all = {}
    dict_out_all["emu_params"] = pars_chain
    dict_out_all["forest_out"] = out_ari
    dict_out_all["zs"] = pars_chain["zs"]
    np.save("int_data_figs/arinyo_from_desi_p1d.npy", dict_out_all)

    dict_save_file = {
        "zs": dict_out_all["zs"],
    }
    for par in out_ari.keys():
        if par != "p1d":
            dict_save_file[par] = out_ari[par]

    folder = os.path.join(os.path.dirname(forestflow.__path__[0]), "data", "priors")

    np.save(folder + "priors_arinyo_from_p1d.npy", dict_save_file)

    return
