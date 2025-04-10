import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
# os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
import numpy as np
from mpi4py import MPI
from lace.cosmo.thermal_broadening import thermal_broadening_kms
from lace.cosmo import camb_cosmo, fit_linP


def main():
    """Produce prior samples for DESI-DR1"""

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # which DESI constraints to use
    use_w0wa = False

    # number of samples
    nn = 100000
    emu_params = {
        "Delta2_p": np.zeros(nn),
        "n_p": np.zeros(nn),
        "mF": np.zeros(nn),
        "gamma": np.zeros(nn),
        "sigT_Mpc": np.zeros(nn),
        "kF_Mpc": np.zeros(nn),
    }

    kp_Mpc = 0.7

    ## Redshift
    # DESI-DR1 KP6
    z = 2.33

    ## IGM
    # Table 4 https://arxiv.org/pdf/1808.04367
    T0 = 0.5 * (1.014 + 1.165) * 1e4
    err_T0 = 0.25 * (0.25 + 0.15 + 0.29 + 0.19) * 1e4

    gamma = 0.5 * (1.74 + 1.63)
    err_gamma = 0.25 * (0.15 + 0.21 + 0.16 + 0.19)

    mF = 0.5 * (0.825 + 0.799)
    err_mF = 0.25 * (0.009 + 0.008 + 0.008 + 0.008)

    lambdap = 0.5 * (79.4 + 81.1)  # [kpc]
    err_lambdap = 0.25 * (5.1 + 5.0 + 4.6 + 4.7)

    err_T0_use = np.random.normal(size=nn) * err_T0
    err_gamma_use = np.random.normal(size=nn) * err_gamma
    err_mF_use = np.random.normal(size=nn) * err_mF
    err_lambdap_use = np.random.normal(size=nn) * err_lambdap

    ## COSMO
    if use_w0wa == False:
        # TABLE V
        # DESI DR2 + CMB LCDM
        Om = 0.3027
        Om_err = 0.0036
        H0 = 68.17
        H0_err = 0.28
        w0 = -1
        w0_err = 0
        wa = 0
        wa_err = 0
    else:
        # DESI+CMB+DESY5 w0waCDM
        Om = 0.3191
        Om_err = 0.0056
        H0 = 66.74
        H0_err = 0.56
        w0 = -0.752
        w0_err = 0.057
        wa = -0.86
        wa_err = 0.22

    omnuh2 = 0.0006  # fixed
    mnu = omnuh2 * 93.14

    # Planck2018 Table 1
    ombh2 = 0.02233
    ombh2_err = 0.00015
    # omch2 = 0.1198
    # omch2_err = 0.0012
    ln_As1010 = 3.043
    ln_As1010_err = 0.014
    ns = 0.9652
    ns_err = 0.0042

    err_ln_As1010_use = np.random.normal(size=nn) * ln_As1010_err
    err_ns_use = np.random.normal(size=nn) * ns_err
    err_Om_use = np.random.normal(size=nn) * Om_err
    err_ombh2_use = np.random.normal(size=nn) * ombh2_err
    err_H0_use = np.random.normal(size=nn) * H0_err
    err_w0_use = np.random.normal(size=nn) * w0_err
    err_wa_use = np.random.normal(size=nn) * wa_err

    ind = np.arange(nn)
    ind_use = np.array_split(ind, size)[rank]

    for ii in ind_use:
        if rank == 0:
            if ii % 10 == 0:
                print(ii)
        _H0 = H0 + err_H0_use[ii]
        _Om = Om + err_Om_use[ii]
        _ombh2 = ombh2 + err_ombh2_use[ii]
        _omch2 = _Om * (_H0 / 100) ** 2 - _ombh2
        _ln_As1010 = np.exp(ln_As1010 + err_ln_As1010_use[ii]) * 1e-10
        _ns = ns + err_ns_use[ii]
        _w0 = w0 + err_w0_use[ii]
        _wa = wa + err_wa_use[ii]

        cosmo = {
            "H0": _H0,
            "omch2": _omch2,
            "ombh2": _ombh2,
            "mnu": mnu,
            "omk": 0,
            "As": _ln_As1010,
            "ns": _ns,
            "nrun": 0.0,
            "w": _w0,
            "wa": _wa,
        }

        sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(cosmo)
        # compute linear power parameters at each z (in Mpc units)
        linP_zs = fit_linP.get_linP_Mpc_zs(sim_cosmo, [z], kp_Mpc)
        dkms_dMpc_zs = camb_cosmo.dkms_dMpc(sim_cosmo, z=np.array([z]))

        emu_params["Delta2_p"][ii] = linP_zs[0]["Delta2_p"]
        emu_params["n_p"][ii] = linP_zs[0]["n_p"]

        emu_params["mF"][ii] = mF + err_mF_use[ii]
        emu_params["gamma"][ii] = gamma + err_gamma_use[ii]

        sigma_T_kms = thermal_broadening_kms(T0 + err_T0_use[ii])
        sigT_Mpc = sigma_T_kms / dkms_dMpc_zs[0]
        emu_params["sigT_Mpc"][ii] = sigT_Mpc

        kF_Mpc = 1 / ((lambdap + err_lambdap_use[ii]) / 1000)
        emu_params["kF_Mpc"][ii] = kF_Mpc

    np.save("out/input_priors" + str(rank) + ".npy", emu_params)


if __name__ == "__main__":
    main()
