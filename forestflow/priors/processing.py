"""
Construct and transform samples for the priors analysis.

The P1D posterior chains are decoded by cup1d
(``cup1d.postprocessing.forestflow_priors``), which writes the processed
chains read by :func:`forestflow.priors.loaders.load_p1d_chain_for_forestflow`.
This module maps those samples through the ForestFlow emulator and builds the
GetDist sample sets of the analysis. ``getdist`` is optional and imported only
where needed.
"""

from __future__ import annotations

from typing import Any
from numpy.typing import ArrayLike, NDArray

import os
import numpy as np

from lace.cosmo import cosmology
from forestflow.paths import get_data_path

#: Default folder of the intermediate products, relative to the working directory.
INT_DATA_FOLDER = "int_data_figs"


def add_p1d_sys(
    P1D: ArrayLike,
    sys_shift: Any,
    zeff: float | None = 2.33,
    seed: int | None = 12345,
) -> tuple[dict, dict]:
    """
    Add the model systematic error of ``bias_delta`` and ``beta`` to P1D samples.

    Parameters
    ----------
    P1D : mapping
        P1D-derived arrays as returned by
        :func:`forestflow.priors.loaders.load_p1d_data`.
    sys_shift : mapping
        Relative systematic error, ``|model/sim - 1|`` as a fraction, with
        redshifts ``z`` and arrays ``bias_delta`` and ``beta``. If it also has
        ``bias_eta``, the independent errors are those of ``bias_delta`` and
        ``bias_eta``, and ``beta`` follows from them (see Notes).
    zeff : float, default=2.33
        Redshift of the P1D samples.
    seed : int, default=12345
        Random-number generator seed.

    Returns
    -------
    P1D_sys : dict
        Copy of ``P1D`` with ``bias_delta``, ``beta``, ``bias_eta`` and the
        sigma8 combinations perturbed.
    rel_sys : dict
        Relative systematic error at ``zeff`` of the parameters with an
        independent error (``bias_delta`` and ``beta``, or ``bias_delta`` and
        ``bias_eta``).

    Notes
    -----
    The systematic is a linear fit to ``sys_shift`` against redshift, evaluated
    at ``zeff`` and clipped at zero. By default, each sample is multiplied by
    ``1 + eps_b`` (``bias_delta``) and ``1 + eps_beta`` (``beta``), with
    independent zero-mean Gaussian ``eps`` of that width, and ``bias_eta`` by
    their product. If ``sys_shift`` has ``bias_eta``, ``bias_delta`` and
    ``bias_eta`` get independent factors ``1 + eps_b`` and ``1 + eps_eta``,
    and ``beta = bias_eta f / bias_delta`` is multiplied by their ratio. The
    derived ``b_delta sigma8`` and ``b_eta f sigma8`` thus share the
    ``bias_delta`` error and stay correlated.
    """
    eta_independent = "bias_eta" in sys_shift
    pars = ["bias_delta", "bias_eta"] if eta_independent else ["bias_delta", "beta"]
    rel_sys = {}
    for par in pars:
        fit_sys = np.polyfit(sys_shift["z"], sys_shift[par], 1)
        rel_sys[par] = float(np.clip(np.polyval(fit_sys, zeff), 0, None))

    rng = np.random.default_rng(seed=seed)
    nn = P1D["bias_delta"].shape[0]
    fac_b = 1 + rel_sys["bias_delta"] * rng.standard_normal(nn)
    if eta_independent:
        fac_beta = (1 + rel_sys["bias_eta"] * rng.standard_normal(nn)) / fac_b
    else:
        fac_beta = 1 + rel_sys["beta"] * rng.standard_normal(nn)

    P1D_sys = {key: np.copy(val) for key, val in P1D.items()}
    P1D_sys["bias_delta"] = P1D["bias_delta"] * fac_b
    P1D_sys["beta"] = P1D["beta"] * fac_beta
    P1D_sys["bias_eta"] = P1D["bias_eta"] * fac_b * fac_beta
    P1D_sys["bias_delta_sig_8_z"] = P1D["bias_delta_sig_8_z"] * fac_b
    P1D_sys["bias_eta_f_sig_8_z"] = P1D["bias_eta_f_sig_8_z"] * fac_b * fac_beta

    return P1D_sys, rel_sys


def set_getdist_samples(
    BAO: Any, P1D: ArrayLike, P1D_sys: ArrayLike | None = None
) -> NDArray[Any]:

    """
    Construct named GetDist containers for BAO and P1D prior samples.

    Parameters
    ----------
    BAO : mapping
        BAO sample arrays keyed by survey configuration, as returned by
        :func:`forestflow.priors.loaders.load_BAO_data`.
    P1D : mapping
        P1D-derived arrays as returned by
        :func:`forestflow.priors.loaders.load_p1d_data`.
    P1D_sys : mapping, optional
        P1D arrays including the model systematic, as returned by
        :func:`add_p1d_sys`; stored as ``p1d_sys``.

    Returns
    -------
    dict of getdist.MCSamples
        Named BAO configurations plus the ``p1d`` (and ``p1d_sys``) sample
        collections.
    """
    from getdist import MCSamples

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

    def p1d_vstack(P1D: Any) -> NDArray[Any]:
        return np.vstack(
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

    label_sample = {
        "p1d": r"DESI $P_\mathrm{1D}$ w/o syst.",
        "p1d_sys": r"DESI $P_\mathrm{1D}$",
    }
    p1d_data = {"p1d": P1D}
    if P1D_sys is not None:
        p1d_data["p1d_sys"] = P1D_sys

    for key, P1D_key in p1d_data.items():
        all_samples[key] = MCSamples(
            samples=p1d_vstack(P1D_key),
            names=names,
            labels=labels,
            label=label_sample[key],
        )

    return all_samples


def set_cmbspa_sig8z(
    nsamples: int | None = 200, nz: int | None = 20, out_folder: str = INT_DATA_FOLDER
) -> None:

    # CMB-SPA Table 1 https://arxiv.org/abs/2506.20707v1
    """
    Sample CMB-SPA cosmologies and write their sigma8-redshift product.

    Parameters
    ----------
    nsamples : int, default=200
        Number of cosmological draws.
    nz : int, default=20
        Number of redshifts between 0 and 3.
    out_folder : str, default="int_data_figs"
        Folder of the intermediate products.

    Notes
    -----
    Writes ``<out_folder>/sig8_cmb_spa.npy``.
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
    np.save(os.path.join(out_folder, "sig8_cmb_spa.npy"), dict_out)

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


def accel2_k_mu_grid(data_accel2: dict, nmu: int = 25) -> tuple[Any, ...]:
    """
    Build a (k, mu) grid on the ACCEL2 P3D wavenumbers with uniform mu.

    Parameters
    ----------
    data_accel2 : mapping
        ACCEL2 measurements with ``k3d_Mpc`` of shape ``(n_k, n_mu)``, e.g.
        from ``cup1d.p1ds.synthetic.data_accel2.load_data``.
    nmu : int, default=25
        Number of uniformly spaced mu values in [0, 1].

    Returns
    -------
    knew3d, munew3d : ndarray
        Wavenumber and cosine-angle grids of shape ``(n_k, nmu)``.
    """
    k_Mpc = np.asarray(data_accel2["k3d_Mpc"])[:, 0]
    mu = np.linspace(0, 1, nmu)
    knew3d = np.repeat(k_Mpc[:, None], nmu, axis=1)
    munew3d = np.broadcast_to(mu, knew3d.shape).copy()
    return knew3d, munew3d


def set_desifs_sig8z(nsamples: int | None = 5000, out_folder: str = INT_DATA_FOLDER) -> None:

    # DESI FS DR1  redshifts https://arxiv.org/abs/2411.12021

    # Table1
    """
    Sample DESI full-shape cosmologies and write sigma8-growth products.

    Parameters
    ----------
    nsamples : int, default=5000
        Number of independent cosmological draws for each tracer sample.
    out_folder : str, default="int_data_figs"
        Folder of the intermediate products.

    Notes
    -----
    Writes ``<out_folder>/sig8_desi.npy``.
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
    np.save(os.path.join(out_folder, "sig8_desi.npy"), dict_out)

    return


def set_map_igm_p3d(
    lab_sample: ArrayLike,
    pars_chain: ArrayLike,
    store_p1d: bool | None=False,
    store_p3d: bool | None=False,
    emulator_label: str | None="forest_mpg_gp",
    n_store_power: int | None=1000,
    batch_size: int | None=100,
    accel2_data: dict | None = None,
    out_folder: str = INT_DATA_FOLDER,
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
    accel2_data : mapping, optional
        ACCEL2 measurements (``k1d_Mpc`` and ``k3d_Mpc``), required to store
        powers for ``lab_sample="accel2"``; e.g. from
        ``cup1d.p1ds.synthetic.data_accel2.load_data``.
    out_folder : str, default="int_data_figs"
        Folder of the intermediate products.

    Raises
    ------
    NotImplementedError
        If power storage is requested for a non-Accel2 sample without grids.
    ValueError
        If power storage is requested for ACCEL2 without ``accel2_data``.

    Notes
    -----
    Writes ``<out_folder>/arinyo_from_<lab_sample>_p1d.npy`` and the Arinyo
    summary ``<repo>/data/priors_<lab_sample>_priors_arinyo_from_p1d.npy``. Besides the
    emulated ``bias`` and ``bias_eta``, ``beta = bias_eta * f(z) / bias`` is
    stored; ``f(z)`` does not depend on the sampled ``As`` and ``ns``.
    """
    from forestflow.emulator import load_emulator
    from forestflow.model.arinyo import ArinyoModel

    if store_p1d | store_p3d:
        if lab_sample != "accel2":
            raise NotImplementedError(
                "k and mu for p1d and p3d not implemented for this sample"
            )
        if accel2_data is None:
            raise ValueError("storing ACCEL2 powers needs accel2_data")
        knew3d, munew3d = accel2_k_mu_grid(accel2_data)
        k_Mpc = np.asarray(accel2_data["k1d_Mpc"])

    emulator = load_emulator(emulator_label)
    print("Emulator loaded:", emulator_label)

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
    np.save(os.path.join(out_folder, "arinyo_from_" + lab_sample + "_p1d.npy"), dict_out_all)

    dict_save_file = {
        "zs": dict_out_all["zs"],
        "emulator_label": emulator_label,
    }
    for par in out_ari.keys():
        if par not in ["p1d", "p3d", "plin"]:
            dict_save_file[par] = out_ari[par]

    np.save(
        get_data_path() / f"priors_{lab_sample}_priors_arinyo_from_p1d.npy",
        dict_save_file,
    )

    return
