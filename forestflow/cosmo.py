import numpy as np


def rescale_pklin(z, k_Mpc, fun_linpower, tar_cosmo, kp_Mpc=0.7, ks_Mpc=0.05):
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

    fid_cosmo = fun_linpower.cosmo

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
    pklin = fun_linpower(z, k_Mpc)

    pklin_rescaled = pklin * np.exp(
        ln_ratio_A_p + delta_n_p * rotk + 0.5 * delta_alpha_p * rotk**2
    )

    return pklin, pklin_rescaled
