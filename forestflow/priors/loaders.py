"""
Load the external data and the intermediate products of the priors analysis.

External data (BAO chains and fits, BAO redshift weights) are read below
``lya_data_path`` (see :func:`forestflow.paths.get_lya_data_path`).
Intermediate products live in ``int_data_folder`` (by default
``int_data_figs`` relative to the working directory, as used by the
``notebooks/Figures_CM2026`` notebooks).

The P1D-chain products ``<lab_sample>_priors_cosmo_IGM_from_p1d.npy`` are
written by cup1d (:func:`cup1d.postprocessing.forestflow_priors.set_process_p1d_chain`),
since decoding a cup1d chain needs the cup1d likelihood; ForestFlow only reads
them.

``getdist``, ``vega`` and ``astropy`` are optional dependencies, imported only
by the functions that need them.
"""

from __future__ import annotations

import os
from typing import Any
from numpy.typing import NDArray

import numpy as np

from lace.cosmo import cosmology
from forestflow.paths import get_lya_data_path
from forestflow.priors.processing import INT_DATA_FOLDER, set_map_igm_p3d


def load_BAO_data(
    nn: int | None = 10000, lya_data_path: str | os.PathLike | None = None
) -> NDArray[Any]:
    """
    Load hard-coded DR1/DR2 BAO chains and derived bias combinations.

    Parameters
    ----------
    nn : int, default=10000
        Number of multivariate-normal draws for each external high-S/N fit.
    lya_data_path : str or path-like, optional
        Root of the external data, see :func:`forestflow.paths.get_lya_data_path`.
        Reads ``lya_bao/dr1``, ``lya_bao/dr2`` and ``lya_bao/fits_andreu``.

    Returns
    -------
    dict
        Results grouped by ``dr1``, ``dr2``, ``dr1_hsnr``, and ``dr2_hsnr``.
        Entries include derived bias combinations, weights where available,
        and DR2 redshift weights.

    Notes
    -----
    Requires ``getdist`` and ``vega``.
    """
    from getdist import loadMCSamples
    from vega import FitResults

    root = get_lya_data_path(lya_data_path)

    zeff = 2.33
    class_planck = cosmology.Cosmology(cosmo_label="Planck18_noBAO_wnu")
    planck_sig8 = class_planck.get_sigma8(zeff)
    planck_f = class_planck.get_growth_rate(zeff)

    BAO = {}

    # load baseline

    chains_file_dr1 = str(
        root / "lya_bao" / "dr1" / "lyaxlya_lyaxlyb_lyaxqso_lybxqso-baseline_combined"
    )
    chains_file_dr2 = str(
        root / "lya_bao" / "dr2" / "lyaxlya_lyaxlyb_lyaxqso_lybxqso-final_base"
    )

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
    basedir = root / "lya_bao" / "fits_andreu"
    fit_file_dr1 = str(basedir / "fit_output_dr1_mid_prior.fits")
    fit_file_dr2 = str(basedir / "fit_output_mid_prior.fits")

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
        samples = np.random.multivariate_normal(mean, cov, size=nn)

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

    zhist, hist_smooth = load_bao_weights(lya_data_path=root)
    BAO["dr2"]["zhist"] = zhist
    BAO["dr2"]["smooth_weights"] = hist_smooth
    BAO["dr2_hsnr"]["zhist"] = zhist
    BAO["dr2_hsnr"]["smooth_weights"] = hist_smooth

    return BAO


def load_p1d_data(
    lab_sample: str | None = "desi",
    zeff: float | None = 2.33,
    emu_type: str | None = "legacy",
    int_data_folder: str | os.PathLike = INT_DATA_FOLDER,
) -> NDArray[Any]:
    """
    Load P1D-chain-derived cosmology and forest quantities at one redshift.

    Parameters
    ----------
    lab_sample : str, default="desi"
        P1D sample label passed to :func:`load_map_igm_p3d`.
    zeff : float, default=2.33
        Effective redshift used to compute the Planck18 growth rate.
    emu_type : {"legacy", "new"}, default="legacy"
        Product of the legacy paper cINN (``int_data_folder/legacy_emu``) or of
        the current emulator, see :func:`load_map_igm_p3d`.
    int_data_folder : str or path-like, default="int_data_figs"
        Folder of the intermediate products.

    Returns
    -------
    dict
        P1D-derived arrays including bias combinations, IGM parameters, and
        linear-amplitude summary parameters.
    """
    class_planck = cosmology.Cosmology(cosmo_label="Planck18_wonu")
    planck_f = class_planck.get_growth_rate(zeff)

    dict_out_all = load_map_igm_p3d(
        lab_sample=lab_sample,
        zeff=zeff,
        emu_type=emu_type,
        int_data_folder=int_data_folder,
    )
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


def load_p1d_chain_for_forestflow(
    lab_sample: str | None = "desi",
    int_data_folder: str | os.PathLike = INT_DATA_FOLDER,
) -> NDArray[Any]:
    """
    Load the processed P1D chain (ForestFlow inputs per posterior sample).

    Parameters
    ----------
    lab_sample : str, default="desi"
        Sample label incorporated into the intermediate filename.
    int_data_folder : str or path-like, default="int_data_figs"
        Folder of the intermediate products.

    Returns
    -------
    dict
        Serialized P1D prior-chain product.

    Raises
    ------
    FileNotFoundError
        If the product is missing. It is written by cup1d, which can decode
        its chains: ``cup1d.postprocessing.forestflow_priors.set_process_p1d_chain``.
    """
    fname = os.path.join(int_data_folder, lab_sample + "_priors_cosmo_IGM_from_p1d.npy")
    if not os.path.isfile(fname):
        raise FileNotFoundError(
            f"Missing {fname}. Generate it with cup1d: "
            "cup1d.postprocessing.forestflow_priors.set_process_p1d_chain("
            f"{lab_sample!r}, out_folder={str(int_data_folder)!r})"
        )
    return np.load(fname, allow_pickle=True).item()


def load_map_igm_p3d(
    lab_sample: str | None = "desi",
    zeff: float | None = 2.33,
    emu_type: str | None = "legacy",
    int_data_folder: str | os.PathLike = INT_DATA_FOLDER,
    accel2_data: dict | None = None,
) -> NDArray[Any]:
    """
    Load mapped P1D-to-P3D IGM samples, generating them on cache miss.

    Parameters
    ----------
    lab_sample : str, default="desi"
        Sample label incorporated into the intermediate filename.
    zeff : float, default=2.33
        Effective redshift used to convert beta to velocity-gradient bias
        for legacy products that do not store ``bias_eta``.
    emu_type : {"legacy", "new"}, default="legacy"
        ``"legacy"`` reads the product of the legacy paper cINN, kept in
        ``int_data_folder/legacy_emu`` (it cannot be regenerated); otherwise
        the product of the current emulator, generated on cache miss with
        :func:`forestflow.priors.processing.set_map_igm_p3d`.
    int_data_folder : str or path-like, default="int_data_figs"
        Folder of the intermediate products.
    accel2_data : dict, optional
        ACCEL2 measurements, needed to regenerate the ``"accel2"`` product;
        see :func:`forestflow.priors.processing.set_map_igm_p3d`.

    Returns
    -------
    dict
        Mapped emulator and forest parameter samples, augmented with signed
        ``bias_delta`` and ``bias_eta`` arrays.
    """
    if emu_type == "legacy":
        fname = os.path.join(
            int_data_folder, "legacy_emu", "arinyo_from_" + lab_sample + "_p1d.npy"
        )
    else:
        fname = os.path.join(int_data_folder, "arinyo_from_" + lab_sample + "_p1d.npy")

    if not os.path.isfile(fname):
        if emu_type == "legacy":
            raise FileNotFoundError(f"Missing legacy-emulator product {fname}")
        print("Missing file:", fname)
        print("Generating it from the P1D chain...")
        pars_chain = load_p1d_chain_for_forestflow(
            lab_sample=lab_sample, int_data_folder=int_data_folder
        )
        store = lab_sample != "desi"
        set_map_igm_p3d(
            lab_sample=lab_sample,
            pars_chain=pars_chain,
            store_p3d=store,
            store_p1d=store,
            accel2_data=accel2_data,
            out_folder=int_data_folder,
        )
    data = np.load(fname, allow_pickle=True).item()

    # products without a label were made with the legacy paper cINN
    print("Arinyo mapping from emulator:", data.get("emulator_label", "legacy"))

    data["forest_out"]["bias_delta"] = -np.abs(data["forest_out"]["bias"])
    if "bias_eta" not in data["forest_out"]:
        # legacy emulator only predicts beta
        class_planck = cosmology.Cosmology(cosmo_label="Planck18_wonu")
        planck_f = class_planck.get_growth_rate(zeff)
        data["forest_out"]["bias_eta"] = (
            data["forest_out"]["beta"] * data["forest_out"]["bias_delta"] / planck_f
        )

    return data


def load_bao_weights(
    lya_data_path: str | os.PathLike | None = None,
) -> tuple[Any, ...]:
    """
    Load and smooth the DR2 BAO redshift-weight curve.

    Parameters
    ----------
    lya_data_path : str or path-like, optional
        Root of the external data, see :func:`forestflow.paths.get_lya_data_path`.
        Reads ``lya_bao/dr2/delta_attributes.fits.gz``. Requires ``astropy``.

    Returns
    -------
    zhist : ndarray
        Redshifts reconstructed from FITS logarithmic wavelengths.
    hist_smooth : ndarray
        Normalized squared-weight curve smoothed with a Savitzky--Golay filter.
    """
    from astropy.io import fits
    from scipy.signal import savgol_filter

    filename = get_lya_data_path(lya_data_path) / "lya_bao" / "dr2" / "delta_attributes.fits.gz"
    with fits.open(filename) as hdul:
        # Compute z
        zhist = 10 ** (hdul[1].data["LOGLAM"]) / 1215.67 - 1
        # Normalize histogram/weights
        weights = hdul[1].data["WEIGHT"] ** 2
    hist = weights / np.max(weights)

    # Smooth the curve
    hist_smooth = savgol_filter(hist, window_length=101, polyorder=3)

    return zhist, hist_smooth


# fixed relative systematic errors of the "fixed" option of load_sys_shift,
# independent errors on bias_delta and bias_eta; the error of beta is propagated
SYS_FIXED = {"bias_delta": 0.010, "bias_eta": 0.031}


def load_sys_shift(
    option: str = "accel2_offset",
    fname: str = os.path.join(INT_DATA_FOLDER, "legacy_emu", "accel2_priors_sys_offset.npy"),
    fixed: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Load the relative model systematic of ``bias_delta`` and ``beta``.

    Parameters
    ----------
    option : {"accel2_offset", "fixed"}, default="accel2_offset"
        ``"accel2_offset"`` reads the common-offset estimate saved by the
        ``accel2_priors`` notebook, independent errors on ``bias_delta`` and
        ``beta``. ``"fixed"`` uses constant relative errors, ``SYS_FIXED``
        unless ``fixed`` is given, on the redshifts of that file.
    fname : str
        File written by ``accel2_priors``, relative to the working directory.
    fixed : dict, optional
        Independent relative errors, as fractions, of ``bias_delta`` and
        ``bias_eta``.

    Returns
    -------
    dict
        Same layout as ``sys_shift``: redshifts ``z`` and one array of
        relative errors per parameter, as read by ``plots._plot_sys_band`` and
        ``processing.add_p1d_sys``. For ``"fixed"``, it also contains
        ``bias_eta``, which tells ``add_p1d_sys`` that the independent errors
        are those of ``bias_delta`` and ``bias_eta``, and ``beta`` holds the
        propagated error of ``beta = bias_eta f / bias_delta``,
        ``sqrt(s_bdelta^2 + s_beta^2)`` to first order, used for the plotted
        band.
    """
    sys_offset = np.load(fname, allow_pickle=True).item()
    if option == "accel2_offset":
        return sys_offset
    if option == "fixed":
        values = SYS_FIXED if fixed is None else fixed
        zz = np.asarray(sys_offset["z"], dtype=float)
        s_b, s_eta = float(values["bias_delta"]), float(values["bias_eta"])
        s_beta = np.hypot(s_b, s_eta)
        return {
            "z": zz,
            "bias_delta": np.full(len(zz), s_b),
            "bias_eta": np.full(len(zz), s_eta),
            "beta": np.full(len(zz), s_beta),
            "meta": {
                "method": "fixed relative errors on bias_delta and bias_eta, "
                          "beta propagated to first order",
                "values": {"bias_delta": s_b, "bias_eta": s_eta, "beta": s_beta},
            },
        }
    raise ValueError(f"Unknown option {option!r}; use 'accel2_offset' or 'fixed'")
