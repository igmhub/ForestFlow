import numpy as np
from scipy.integrate import simpson


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


def rescale_linP(fid_cosmo, tar_cosmo, linP_zs, kp_Mpc=0.7, ks_Mpc=0.05):

    ratio_As = 1.0
    delta_ns = 0.0
    delta_nrun = 0.0

    for par in tar_cosmo:
        ratio_As = tar_cosmo["As"] / fid_cosmo["As"]
        delta_ns = tar_cosmo["ns"] - fid_cosmo["ns"]
        delta_nrun = tar_cosmo["nrun"] - fid_cosmo["nrun"]

        ln_kp_ks = np.log(kp_Mpc / ks_Mpc)

        # compute scalings
        delta_alpha_p = delta_nrun
        delta_n_p = delta_ns + delta_nrun * ln_kp_ks
        ln_ratio_A_p = (
            np.log(ratio_As) + (delta_ns + 0.5 * delta_nrun * ln_kp_ks) * ln_kp_ks
        )

        # update values of linP_params at emulator pivot point, at each z
        linP_Mpc_params = []
        for zlinP in linP_zs:
            linP_Mpc_params.append(
                {
                    "Delta2_p": zlinP["Delta2_p"] * np.exp(ln_ratio_A_p),
                    "n_p": zlinP["n_p"] + delta_n_p,
                    "alpha_p": zlinP["alpha_p"] + delta_alpha_p,
                }
            )
        return linP_Mpc_params


def fft_top_hat(k, R):
    x = k * R
    return 3 / x**3 * (np.sin(x) - x * np.cos(x))


def sig8(k, pk, R):
    res = (k**3 * pk / 2 / np.pi**2) * fft_top_hat(k, R) ** 2
    return np.sqrt(simpson(res, x=np.log(k)))


def sig8_fsig8_with_res(camb_results, fid_cosmo, tar_cosmo, kp_Mpc=0.7, ks_Mpc=0.05):
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

    # f_growth_z0 = camb_results.get_fsigma8()[0] / camb_results.get_sigma8()[0]
    f_growth_z = camb_results.get_fsigma8()[1] / camb_results.get_sigma8()[1]

    k_hMpc, z_pk, pk = camb_results.get_linear_matter_power_spectrum(k_hunit=True)

    pk_hMpc_z0 = pk[0]
    pk_hMpc_z = pk[1]
    k_Mpc = k_hMpc * (fid_cosmo["H0"] / 100)

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

    pklin_rescaled_z0 = pk_hMpc_z0 * np.exp(
        ln_ratio_A_p + delta_n_p * rotk + 0.5 * delta_alpha_p * rotk**2
    )
    pklin_rescaled_z = pk_hMpc_z * np.exp(
        ln_ratio_A_p + delta_n_p * rotk + 0.5 * delta_alpha_p * rotk**2
    )

    sigma8_z0 = sig8(k_hMpc, pklin_rescaled_z0, 8)
    sigma8_z = sig8(k_hMpc, pklin_rescaled_z, 8)

    dict_out = {
        "sigma8_z0": sigma8_z0,
        "sigma8_z": sigma8_z,
        # "f_growth_z0": f_growth_z * sigma8_z0,
        "f_growth_z": f_growth_z * sigma8_z,
    }

    return dict_out
