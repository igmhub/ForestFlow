import numpy as np
from lace.cosmo import camb_cosmo


def rescale_pklin(
    z, k_Mpc, fun_linpower, fid_cosmo, tar_cosmo, kp_Mpc=0.7, ks_Mpc=0.05
):
    """
    Rescales the linear power spectrum from the fiducial cosmology to the target cosmology.

    Parameters:
    z (float): Redshift of the linear power spectrum.
    k_Mpc (float): Array of k in Mpc^-1.
    fun_linpower (function): Linear power spectrum function.
        It takes as input a redshift and an array of k in Mpc^-1,
        and returns the linear power spectrum.
    fid_cosmo (dictionary): Cosmology object representing the fiducial cosmology.
    tar_cosmo (dictionary): Cosmology object representing the target cosmology.
    """

    ratio_As = tar_cosmo["As"] / fid_cosmo["As"]
    delta_ns = tar_cosmo["ns"] - fid_cosmo["ns"]
    delta_nrun = tar_cosmo["nrun"] - fid_cosmo["nrun"]

    ln_kp_ks = np.log(kp_Mpc / ks_Mpc)
    delta_alpha_p = delta_nrun
    delta_n_p = delta_ns + delta_nrun * ln_kp_ks
    ln_ratio_A_p = (
        np.log(ratio_As) + (delta_ns + 0.5 * delta_nrun * ln_kp_ks) * ln_kp_ks
    )

    rotk = np.log(k_Mpc / kp_Mpc)
    pklin = fun_linpower(z, k_Mpc, grid=False)

    pklin_rescaled = pklin * np.exp(
        ln_ratio_A_p + delta_n_p * rotk + 0.5 * delta_alpha_p * rotk**2
    )

    return pklin, pklin_rescaled


def get_linP_interp(cosmo, zmin=0, zmax=10, nz=256, camb_kmax_Mpc=200.0):
    """
    Obtain an interpolator of the linear power spectrum from CAMB.

    Parameters:
        cosmo (Cosmology): Cosmology object representing the cosmological parameters.
        zs (list): List of redshifts at which to obtain the linear power spectrum.
        camb_results (CAMBResults): CAMBResults object containing the precomputed CAMB results.
        camb_kmax_Mpc (float, optional): Maximum k in Mpc^-1 to consider for the linear power spectrum.
            Defaults to 30.

    Returns:
        linP_interp (interpolator): Interpolator for the linear power spectrum.
    """

    inst_camb = camb_cosmo.get_cosmology_from_dictionary(cosmo)

    camb_results = camb_cosmo.get_camb_results(
        inst_camb, zs=np.linspace(zmin, zmax, nz), camb_kmax_Mpc=camb_kmax_Mpc
    )

    # get interpolator from CAMB
    # meaning of var1 and var2 here
    # https://camb.readthedocs.io/en/latest/transfer_variables.html#transfer-variables
    linP_interp = camb_results.get_matter_power_interpolator(
        nonlinear=False,
        var1=8,
        var2=8,
        hubble_units=False,
        k_hunit=False,
        log_interp=True,
    )

    return linP_interp
