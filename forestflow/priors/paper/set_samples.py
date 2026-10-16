"""
Construct and transform samples for the priors analysis.
"""

from typing import Any
from numpy.typing import ArrayLike, NDArray

import os
import numpy as np
from getdist import MCSamples

from lace.cosmo import cosmology, rescale_cosmology


def set_getdist_samples(BAO: Any, P1D: ArrayLike) -> NDArray[Any]:

    """
    Construct named GetDist containers for BAO and P1D prior samples.

    Parameters
    ----------
    BAO : mapping
        BAO sample arrays keyed by survey configuration, as returned by
        :func:`forestflow.priors.paper.load.load_BAO_data`.
    P1D : mapping
        P1D-derived arrays as returned by
        :func:`forestflow.priors.paper.load.load_p1d_data`.

    Returns
    -------
    dict of getdist.MCSamples
        Named BAO configurations plus the ``p1d`` sample collection.
    """
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
        "dr1": "DESI DR1 BAO (w/ low SNR)",
        "dr2": "DESI DR2 BAO (w/ low SNR)",
        "dr1_hsnr": "DESI DR1 BAO",
        "dr2_hsnr": "DESI BAO",
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

    label_sample = {"p1d": r"DESI $P_\mathrm{1D}$"}

    for key in ["p1d"]:
        all_samples[key] = MCSamples(
            samples=p1d.copy(),
            names=names,
            labels=labels,
            label=label_sample[key],
        )

    return all_samples


# cup1d P1D chains mapped to ForestFlow inputs. The chains store unit-cube
# coordinates; they are decoded with the parameter definitions saved next to
# them (fitter_results.npy), not with the bounds of the current cup1d config.
P1D_CHAINS = {
    "desi": {
        "folder": "/home/jchaves/Proyectos/projects/lya/data/out_DESI_DR1/"
        "DESIY1_QMLE3/global_opt/CH24_mpgcen_gpr/chain_7/",
        # nersc: "/global/cfs/cdirs/desi/users/jjchaves/p1d"
        "zs": [2.2, None, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2],
    },
    "accel2": {
        "folder": "/home/jchaves/Proyectos/projects/lya/data/accel2/chains/chain_1/",
        "zs": [2.2, None, 2.6, 3.0, 3.6, 4.0],
    },
}

IGM_PARAMS = ("tau_eff", "sigT_kms", "gamma", "kF_kms")


def set_process_p1d_chain(
    lab_sample: ArrayLike,
    nn: int | None=10000,
    seed: int | None=12345,
    store_p1d: bool | None=False,
    out_folder: str | None="int_data_figs",
) -> None:

    """
    Subsample a cup1d P1D chain and write derived ForestFlow priors.

    Parameters
    ----------
    lab_sample : {"desi", "accel2"}
        Source-chain configuration, see ``P1D_CHAINS``.
    nn : int, default=10000
        Number of posterior samples selected without replacement.
    seed : int, default=12345
        Random-number generator seed.
    store_p1d : bool, default=False
        Also store the P1D predicted by cup1d (contaminants included) for each
        sample. Only implemented for ``"desi"``.
    out_folder : str, default="int_data_figs"
        Folder for the output products.

    Notes
    -----
    The function writes intermediate ``.npy`` products in ``out_folder``.
    IGM histories are converted from km/s to Mpc with the Planck18 cosmology.
    """
    from cup1d.likelihood.parameter import values_from_cube_batch

    pip, chain, chain_params, d2star, nstar, zs, zeff = set_input_process_p1d_chain(
        lab_sample
    )
    if store_p1d and lab_sample != "desi":
        raise NotImplementedError("store_p1d only implemented for lab_sample='desi'")

    # get a random sample
    rng = np.random.default_rng(seed=seed)
    ind = rng.permutation(np.arange(chain.shape[0]))[:nn]
    # physical values of the selected samples, shape (nn,) per parameter
    values = values_from_cube_batch(chain_params, np.asarray(chain[ind]))
    pars_chain = {}

    ## create holder for p1d chain
    pars_chain["z"] = zs

    # cosmo (for Arinyo model)
    pars_chain["As"] = values["As"]
    pars_chain["ns"] = values["ns"]
    pars_chain["Delta2star"] = d2star[ind]
    pars_chain["nstar"] = nstar[ind]

    # cosmo for this project
    pars_chain["sig_8"] = np.zeros((ind.shape[0]))
    pars_chain["sig_8_z0"] = np.zeros((ind.shape[0]))
    pars_chain["f_sig_8"] = np.zeros((ind.shape[0]))

    # input ForestFlow
    for par in ["Delta2_p", "n_p", "mF", "gamma", "sigT_Mpc", "kF_Mpc"]:
        pars_chain[par] = np.zeros((ind.shape[0], zs.shape[0]))

    igm_models = pip.like.theory.model_igm.models

    if store_p1d:
        # p1d at zeff is that of the first data redshift (2.2), as the k grid
        data_label = pip.args.data_label[0]
        k_kms = pip.data[data_label].k_kms
        iz_data = [0, 0] + list(range(1, zs.shape[0] - 1))
        nk = max(len(k_kms[iz]) for iz in iz_data)
        pars_chain["k_kms"] = np.zeros((zs.shape[0], nk))
        pars_chain["p1d"] = np.zeros((ind.shape[0], zs.shape[0], nk))
        for jj, iz in enumerate(iz_data):
            pars_chain["k_kms"][jj, : len(k_kms[iz])] = k_kms[iz]

    ## fill the holder
    # set cosmology
    class_planck = cosmology.Cosmology(cosmo_label="Planck18")
    planck_f = class_planck.get_growth_rate(zeff)
    dkms_dMpc_zs = class_planck.get_dkms_dMpc(pars_chain["z"])

    for ii in range(ind.shape[0]):
        if ii % 100 == 0:
            print(ii)

        like_params = {name: values[name][ii] for name in values}

        pars_chain["mF"][ii] = igm_models["F_model"].get_mean_flux(
            zs, like_params=like_params
        )
        pars_chain["gamma"][ii] = igm_models["T_model"].get_gamma(
            zs, like_params=like_params
        )
        sigT_kms = igm_models["T_model"].get_sigT_kms(zs, like_params=like_params)
        pars_chain["sigT_Mpc"][ii] = sigT_kms / dkms_dMpc_zs
        kF_kms = igm_models["P_model"].get_kF_kms(zs, like_params=like_params)
        pars_chain["kF_Mpc"][ii] = kF_kms * dkms_dMpc_zs

        class_new = rescale_cosmology.RescaledCosmology(
            fid_cosmo=class_planck,
            new_params_dict={"As": pars_chain["As"][ii], "ns": pars_chain["ns"][ii]},
        )

        pars_chain["sig_8"][ii] = class_new.get_sigma8(zeff)
        pars_chain["sig_8_z0"][ii] = class_new.get_sigma8(0)
        pars_chain["f_sig_8"][ii] = planck_f * pars_chain["sig_8"][ii]

        for jj in range(zs.shape[0]):
            linP_params = class_new.get_linP_Mpc_params(pars_chain["z"][jj], kp_Mpc=0.7)
            pars_chain["Delta2_p"][ii, jj] = linP_params["Delta2_p"]
            pars_chain["n_p"][ii, jj] = linP_params["n_p"]

        if store_p1d:
            p1d = pip.like.get_p1d_kms(parameters=like_params)[0][data_label]
            for jj, iz in enumerate(iz_data):
                pars_chain["p1d"][ii, jj, : len(p1d[iz])] = p1d[iz]

    np.save(os.path.join(out_folder, lab_sample + "_inter_chain.npy"), pars_chain)

    dict_save_file = {"zs": pars_chain["z"]}
    for par in pars_chain.keys():
        if par not in ["k_kms", "p1d", "z"]:
            dict_save_file[par] = pars_chain[par]

    np.save(
        os.path.join(out_folder, lab_sample + "_priors_cosmo_IGM_from_p1d.npy"),
        dict_save_file,
    )

    return


def set_input_process_p1d_chain(lab_sample: ArrayLike, zeff: float | None=2.33) -> tuple[Any, ...]:
    """
    Load a cup1d analysis and a P1D posterior chain with its parameters.

    Parameters
    ----------
    lab_sample : {"desi", "accel2"}
        Source chain, see ``P1D_CHAINS``.
    zeff : float, default=2.33
        Effective redshift inserted into the configuration's redshift grid.

    Returns
    -------
    pipeline : cup1d.inference.analysis.Analysis
        CM2026 baseline analysis (``lace_mpg``, DESI DR1). Only its IGM models
        are used to decode the chains; for ``"accel2"`` too, since both chains
        share the baseline IGM parameterisation (checked against the saved
        parameter names).
    chain : numpy.memmap, shape (n_samples, n_params)
        Flattened unit-cube posterior chain, memory-mapped.
    chain_params : dict
        Parameter definitions saved with the chain, in chain order.
    d2star, nstar : ndarray, shape (n_samples,)
        Blob-derived linear-power summaries.
    zs : ndarray
        Analysis redshift grid.
    zeff : float
        Effective redshift used by downstream conversions.
    """
    from cup1d.inference.analysis import Analysis

    folder = P1D_CHAINS[lab_sample]["folder"]
    zs = np.array([zeff if z is None else z for z in P1D_CHAINS[lab_sample]["zs"]])

    saved = np.load(os.path.join(folder, "fitter_results.npy"), allow_pickle=True).item()
    names = saved["like"]["free_param_names"]
    chain_params = {name: saved["like"]["free_params"][name] for name in names}

    pip = Analysis(create_output=False)
    if list(pip.like.free_params) != list(names):
        raise ValueError(
            "free parameters of the cup1d baseline differ from those of the chain"
        )

    # memory-mapped: only the selected samples are read
    chain = np.load(os.path.join(folder, "chain.npy"), mmap_mode="r")
    chain = chain.reshape(-1, len(names))

    blobs = np.load(os.path.join(folder, "blobs.npy"), allow_pickle=True)
    d2star = np.array(blobs["Delta2_star"]).reshape(-1)
    nstar = np.array(blobs["n_star"]).reshape(-1)
    del blobs

    return pip, chain, chain_params, d2star, nstar, zs, zeff


def set_cmbspa_sig8z(nsamples: int | None=200, nz: int | None=20) -> None:

    # CMB-SPA Table 1 https://arxiv.org/abs/2506.20707v1
    """
    Sample CMB-SPA cosmologies and write their sigma8-redshift product.

    Parameters
    ----------
    nsamples : int, default=200
        Number of cosmological draws.
    nz : int, default=20
        Number of redshifts between 2 and 4.4.

    Notes
    -----
    Writes ``int_data_figs/cmbspa_sig8z.npy``.
    """
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


def sample_cosmo_dict(base: Any, n_samples: int | None=1, rng: Any | None=None) -> NDArray[Any]:
    """
    Draw cosmology dictionaries from independent normal input errors.

    Parameters
    ----------
    base : mapping
        Fiducial cosmology; ``err_<name>`` keys set normal standard deviations.
        ``log_As`` is converted to linear ``As`` in each output mapping.
    n_samples : int, default=1
        Number of dictionaries to draw.
    rng : numpy.random.Generator or int, optional
        Generator or seed forwarded to :func:`numpy.random.default_rng`.

    Returns
    -------
    list of dict
        Drawn cosmology dictionaries suitable for ``Cosmology`` construction.
    """
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


def load_k_mu_accel2() -> tuple[Any, ...]:
    """
    Load Accel2 P3D data and construct a uniformly sampled mu grid.

    Returns
    -------
    knew3d, munew3d : ndarray
        Wavenumber and cosine-angle grids of shape ``(n_k, 25)``.
    data_accel2 : mapping
        Data returned by cup1d's Accel2 loader.
    """
    from cup1d.p1ds.simulations.data_accel2 import load_data

    folder = "/home/jchaves/Proyectos/projects/lya/data/accel2/frontier_grid/"
    data_accel2 = load_data(folder)

    # more mu bins, then average
    nmu = 25
    knew3d = np.zeros((data_accel2["k3d_Mpc"].shape[0], nmu))
    munew3d = np.zeros((data_accel2["k3d_Mpc"].shape[0], nmu))
    mnu_uni = np.linspace(0, 1, nmu)
    for ii in range(nmu):
        knew3d[:, ii] = data_accel2["k3d_Mpc"][:, 0]
        munew3d[:, ii] = mnu_uni[ii]

    return knew3d, munew3d, data_accel2


def set_desifs_sig8z(nsamples: int | None=5000) -> None:

    # DESI FS DR1  redshifts https://arxiv.org/abs/2411.12021

    # Table1
    """
    Sample DESI full-shape cosmologies and write sigma8-growth products.

    Parameters
    ----------
    nsamples : int, default=5000
        Number of independent cosmological draws for each tracer sample.

    Notes
    -----
    Writes ``int_data_figs/sig8_desi.npy``.
    """
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


def set_map_igm_p3d(
    lab_sample: ArrayLike,
    pars_chain: ArrayLike,
    store_p1d: bool | None=False,
    store_p3d: bool | None=False,
    emulator_label: str | None="forest_mpg_gp",
    n_store_power: int | None=1000,
    batch_size: int | None=100,
) -> None:

    """
    Map P1D posterior samples to emulator Arinyo and optional power outputs.

    Parameters
    ----------
    lab_sample : str
        Source sample label. ``"accel2"`` provides the implemented P3D grid.
    pars_chain : mapping of ndarray
        Processed P1D posterior product containing IGM and cosmology arrays.
    store_p1d : bool, default=False
        Store modeled P1D arrays where their grid is implemented.
    store_p3d : bool, default=False
        Store modeled P3D and linear-power arrays where their grid is
        implemented.
    emulator_label : str, default="forest_mpg_gp"
        Emulator passed to :func:`forestflow.emulator.load_emulator`. Products
        made with the legacy paper cINN (``mpg_hypercube.pt``) are kept in
        ``int_data_figs/legacy_emu``.
    n_store_power : int, default=1000
        Number of leading samples for which P1D/P3D/linear power are stored
        (in float32). The Arinyo parameters are stored for every sample.
    batch_size : int, default=100
        Samples evaluated together; bounds the peak memory use.

    Raises
    ------
    NotImplementedError
        If power storage is requested for a non-Accel2 sample without grids.

    Notes
    -----
    Writes an intermediate mapping under ``int_data_figs``. Besides the
    emulated ``bias`` and ``bias_eta``, ``beta = bias_eta * f(z) / bias`` is
    stored; ``f(z)`` does not depend on the sampled ``As`` and ``ns``.
    """
    import forestflow
    from forestflow.emulator import load_emulator
    from forestflow.model.arinyo import ArinyoModel

    emulator = load_emulator(emulator_label)
    print("Emulator loaded:", emulator_label)

    if lab_sample == "accel2":
        knew3d, munew3d, _ = load_k_mu_accel2()
        k_Mpc = _["k1d_Mpc"]
    else:
        if store_p1d | store_p3d:
            raise NotImplementedError(
                "k and mu for p1d and p3d not implemented for this sample"
            )

    # initiate Arinyo model, needed to compute P1D and P3D
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
    fid_cosmo = cosmology.Cosmology(cosmo_params_dict=fid_cosmo)
    model_Arinyo = ArinyoModel(fid_cosmo)

    zs = pars_chain["zs"]
    nsam, nz = pars_chain["mF"].shape
    fz = model_Arinyo.linear.get_fz(model_Arinyo.linear.get_linear_theory(zs), zs)

    out_ari = {}
    for par in emulator.output_labels:
        out_ari[par] = np.zeros((nsam, nz))

    n_store_power = min(n_store_power, nsam)
    if store_p1d:
        out_ari["p1d"] = np.zeros((n_store_power, nz, k_Mpc.shape[0]), dtype=np.float32)
    if store_p3d:
        out_ari["plin"] = np.zeros(
            (n_store_power, nz, knew3d.shape[0]), dtype=np.float32
        )
        out_ari["p3d"] = np.zeros(
            (n_store_power, nz) + knew3d.shape, dtype=np.float32
        )

    print("Starting loop over samples", nsam, "samples")

    for i0 in range(0, nsam, batch_size):
        i1 = min(i0 + batch_size, nsam)
        nb = i1 - i0
        if i0 % (10 * batch_size) == 0:
            print(i0)

        rows = [
            {par: pars_chain[par][ii, jj] for par in emulator.input_labels}
            for ii in range(i0, i1)
            for jj in range(nz)
        ]
        pred = emulator.evaluate(rows)
        par_ari = {}
        for par in emulator.output_labels:
            par_ari[par] = np.asarray(pred[par]).reshape(nb, nz)
            out_ari[par][i0:i1] = par_ari[par]

        if i0 >= n_store_power or not (store_p1d | store_p3d):
            continue

        # power spectra only for the first n_store_power samples
        ns_ = min(i1, n_store_power) - i0
        par_ari = {par: val[:ns_] for par, val in par_ari.items()}
        linear = model_Arinyo.linear.get_linear_theory_batch(
            zs,
            [
                {"As": pars_chain["As"][ii], "ns": pars_chain["ns"][ii]}
                for ii in range(i0, i0 + ns_)
            ],
        )
        if store_p1d:
            kk = np.broadcast_to(k_Mpc, (ns_, nz, k_Mpc.shape[0]))
            out_ari["p1d"][i0 : i0 + ns_] = model_Arinyo.P1D_Mpc(
                linear, zs, kk, par_ari
            )
        if store_p3d:
            shape = (ns_, nz) + knew3d.shape
            out_ari["p3d"][i0 : i0 + ns_] = model_Arinyo.P3D_Mpc_k_mu(
                linear,
                zs,
                np.broadcast_to(knew3d, shape),
                np.broadcast_to(munew3d, shape),
                par_ari,
            )
            out_ari["plin"][i0 : i0 + ns_] = model_Arinyo.linear.get_linP_Mpc_batch(
                linear, np.broadcast_to(knew3d[:, 0], (ns_, nz, knew3d.shape[0]))
            )

    out_ari["beta"] = out_ari["bias_eta"] * fz[np.newaxis, :] / out_ari["bias"]

    dict_out_all = {}
    dict_out_all["emu_params"] = pars_chain
    dict_out_all["forest_out"] = out_ari
    dict_out_all["zs"] = zs
    dict_out_all["emulator_label"] = emulator_label
    np.save("int_data_figs/arinyo_from_" + lab_sample + "_p1d.npy", dict_out_all)

    dict_save_file = {
        "zs": dict_out_all["zs"],
        "emulator_label": emulator_label,
    }
    for par in out_ari.keys():
        if par not in ["p1d", "p3d", "plin"]:
            dict_save_file[par] = out_ari[par]

    folder = os.path.join(os.path.dirname(forestflow.__path__[0]), "data", "priors")

    np.save(folder + "_" + lab_sample + "_priors_arinyo_from_p1d.npy", dict_save_file)

    return
