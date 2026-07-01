import numpy as np

from lace.cosmo import cosmology
from forestflow.model_p3d_arinyo import ArinyoModel


# compute power for Arinyo model
def get_arinyo_power(
    sim,
    n3d=50,
    n1d=100,
    kmax_1d_fit=4,
    kmax_3d_fit=5,
    noise={"n_noise": 0, "keep_all_noise": False, "Lbox_Mpc": 100},
):

    data = {}

    mask_1d = (sim["k_Mpc"] <= kmax_1d_fit) & (sim["k_Mpc"] > 0)
    k1d_Mpc = sim["k_Mpc"][mask_1d]
    p1d_Mpc = sim["p1d_Mpc"][mask_1d]
    data["sim_k1d_Mpc"] = k1d_Mpc
    data["sim_p1d_Mpc"] = p1d_Mpc

    mask_3d = (sim["k3d_Mpc"] <= kmax_3d_fit) & np.isfinite(sim["p3d_Mpc"])
    k3d_Mpc = sim["k3d_Mpc"][mask_3d]
    p3d_Mpc = sim["p3d_Mpc"][mask_3d]
    mu3d = sim["mu3d"][mask_3d]
    data["sim_k3d_Mpc"] = k3d_Mpc
    data["sim_p3d_Mpc"] = p3d_Mpc
    data["sim_mu3d"] = mu3d

    # mu3d
    ari_mu = np.zeros((n3d, 2))
    ari_mu[:, 1] = 1
    ari_mu = ari_mu.T.reshape(-1)
    data["model_mu3d"] = ari_mu

    # k3d
    # min_k3d = 0.01
    # max_k3d = 0.3
    min_k3d = k3d_Mpc.min()
    max_k3d = k3d_Mpc.max()
    _ari_k3d_Mpc = np.geomspace(min_k3d, max_k3d, n3d)
    ari_k3d_Mpc = np.zeros((n3d, 2))
    ari_k3d_Mpc[:, 0] = _ari_k3d_Mpc
    ari_k3d_Mpc[:, 1] = _ari_k3d_Mpc
    ari_k3d_Mpc = ari_k3d_Mpc.T.reshape(-1)
    data["model_k3d_Mpc"] = ari_k3d_Mpc

    # k1d
    ari_k1d_Mpc = np.linspace(k1d_Mpc.min(), k1d_Mpc.max(), n1d)
    data["model_k1d_Mpc"] = ari_k1d_Mpc

    # set Arinyo at z3
    cosmo_params_dict = {}
    for par in sim["cosmo_params"]:
        if par != "omk":
            cosmo_params_dict[par] = sim["cosmo_params"][par]
        else:
            cosmo_params_dict[par] = 0.0

    fid_cosmo = cosmology.Cosmology(cosmo_params_dict=cosmo_params_dict)
    model_Arinyo = ArinyoModel(fid_cosmo)

    plin_central_z3 = model_Arinyo.linP_Mpc(sim["z"], ari_k3d_Mpc[:n3d])
    data["Plin_Mpc"] = plin_central_z3

    pars_use = {}
    for par in sim["Arinyo_min"]:
        pars_use[par] = sim["Arinyo_min"][par]

    p3d_central_z3 = model_Arinyo.P3D_Mpc_k_mu(sim["z"], ari_k3d_Mpc, ari_mu, pars_use)
    data["ari_P3D_Mpc"] = p3d_central_z3
    p1d_central_z3 = model_Arinyo.P1D_Mpc(sim["z"], ari_k1d_Mpc, pars_use)
    data["ari_P1D_Mpc"] = p1d_central_z3

    # get kaiser
    pars_use = {}
    for par in sim["Arinyo_min"]:
        if par in ["q1", "q2"]:
            pars_use[par] = 0
        elif par == "kp":
            pars_use[par] = 1e6
        else:
            pars_use[par] = sim["Arinyo_min"][par]

    p3d_central_z3_kai = model_Arinyo.P3D_Mpc_k_mu(
        sim["z"], ari_k3d_Mpc, ari_mu, pars_use
    )
    p1d_central_z3_kai = model_Arinyo.P1D_Mpc(sim["z"], ari_k1d_Mpc, pars_use)
    data["kai_P3D_Mpc"] = p3d_central_z3_kai
    data["kai_P1D_Mpc"] = p1d_central_z3_kai

    if noise["n_noise"] > 0:
        ari_noise_P1D_Mpc = np.zeros((noise["n_noise"], ari_k1d_Mpc.shape[0]))
        for ii in range(noise["n_noise"]):
            ari_noise_P1D_Mpc[ii] = model_Arinyo.P1D_Mpc_Gaussian_noise(
                sim["z"],
                ari_k1d_Mpc,
                sim["Arinyo_min"],
                seed=ii,
                Lbox_Mpc=noise["Lbox_Mpc"],
            )

        if noise["keep_all_noise"]:
            data["ari_noise_P1D_Mpc"] = ari_noise_P1D_Mpc

        data["ari_std_P1D_Mpc"] = np.std(ari_noise_P1D_Mpc, axis=0)

    return data
