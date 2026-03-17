import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

from getdist import plots
from lace.cosmo import cosmology


def plot_bsig8_betafsigma8(samples, ftsize=18):

    # --- plotting ---
    g = plots.get_subplot_plotter(width_inch=8)
    g.settings.lab_fontsize = ftsize
    g.settings.axes_fontsize = ftsize
    g.settings.legend_fontsize = ftsize
    g.settings.num_plot_contours = 2  # 68%, 95%

    colors = ["C0", "C1", "C2", "C3", "C4"]

    g.triangle_plot(
        [
            samples["p1d"],
            samples["dr1"],
            samples["dr2"],
            samples["dr1_hsnr"],
            samples["dr2_hsnr"],
        ],
        params=["b_delta_sigma8", "b_eta_f_sigma8"],
        filled=[
            True,
            False,
            False,
            True,
            True,
        ],
        contour_colors=colors,
        contour_ls=[
            "-.",
            "--",
            "--",
            "-",
            "-",
        ],
        # contour_args=[
        #     {"hatches": ["", ""]},
        #     {"hatches": ["", ""]},
        #     {"hatches": ["", ""]},
        #     {"hatches": ["/", "/"]},
        #     {"hatches": ["\\", "\\"]},
        # ],
        contour_lws=[3, 3, 3, 3, 3],
    )

    plt.tight_layout()
    plt.savefig("figs/nocomb_bdsig8_befsig8.pdf")
    plt.savefig("figs/nocomb_bdsig8_befsig8.png")


def plot_bao_biases(samples, ftsize=18):

    g = plots.get_subplot_plotter(width_inch=8)
    g.settings.lab_fontsize = ftsize
    g.settings.axes_fontsize = ftsize
    g.settings.legend_fontsize = ftsize
    g.settings.num_plot_contours = 2  # 68%, 95%

    g.triangle_plot(
        [
            # samples["p1d"],
            samples["dr1"],
            samples["dr2"],
            samples["dr1_hsnr"],
            samples["dr2_hsnr"],
        ],
        params=["bias_delta", "bias_eta", "beta", "bias_hcd"],
        filled=[
            False,
            False,
            True,
            True,
        ],
        contour_colors=["C1", "C2", "C3", "C4"],
        contour_ls=[
            "--",
            "--",
            "-",
            "-",
        ],
        contour_lws=[3, 3, 3, 3],
    )

    plt.tight_layout()
    plt.savefig("figs/bao_biases.pdf")
    plt.savefig("figs/bao_biases.png")


def plot_comb_bdsig8_befsig8(samples, ftsize=20):

    # --- plotting ---
    g = plots.get_subplot_plotter(width_inch=8)
    g.settings.lab_fontsize = ftsize
    g.settings.axes_fontsize = ftsize
    g.settings.legend_fontsize = ftsize
    g.settings.num_plot_contours = 2  # 68%, 95%

    g.triangle_plot(
        [
            samples["p1d"],
            samples["dr1_hsnr"],
            samples["dr2_hsnr"],
            samples["p1d_dr1"],
            samples["p1d_dr2"],
        ],
        filled=[False, False, False, True, True, True],
        params=["b_delta_sigma8", "b_eta_f_sigma8"],
        contour_colors=["C0", "C9", "C3", "C1", "C2"],
        contour_ls=[
            "-",
            ":",
            ":",
            "--",
            "-.",
        ],
        contour_lws=[3.0, 3.0, 3.0, 3.0, 3.0, 2],
    )

    plt.tight_layout()
    plt.savefig("figs/comb_bdsig8_befsig8.pdf")
    plt.savefig("figs/comb_bdsig8_befsig8.png")


def plot_sig8(samples, ftsize=20):

    # DESY6 Table IV https://arxiv.org/pdf/2601.14559
    # DES 3x2pt LCDM
    mu_des = 0.751
    sigma_des = 0.035
    # DES all LCDM
    mu_des_all = 0.771
    sigma_des_all = 0.020

    # CMB-SPA https://arxiv.org/abs/2506.20707v1 LCDM
    mu_cmb = 0.8137
    sigma_cmb = 0.0038

    # DESI BAO + full shape LCDM https://arxiv.org/abs/2602.18761
    mu_desi = 0.822
    sigma_desi = 0.034

    # --- GetDist chains ---
    chains = [samples["p1d"], samples["p1d_dr1"], samples["p1d_dr2"]]
    chain_labels = [
        r"DESI $P_\mathrm{1D}$",
        r"DESI $P_\mathrm{1D}$ & Ly$\alpha$ BAO DR1",
        r"DESI $P_\mathrm{1D}$ & Ly$\alpha$ BAO DR2",
    ]

    # chains = [
    #     samples["p1d"],
    #     samples["p1d_dr1"],
    #     samples["p1d_dr2"],
    #     samples["p1d_dr1_low"],
    #     samples["p1d_dr2_low"],
    # ]
    # chain_labels = [
    #     r"DESI $P_\mathrm{1D}$",
    #     r"DESI $P_\mathrm{1D}$ & Ly$\alpha$ BAO DR1",
    #     r"DESI $P_\mathrm{1D}$ & Ly$\alpha$ BAO DR2",
    #     r"DESI $P_\mathrm{1D}$ & Ly$\alpha$ BAO DR1 low SNR",
    #     r"DESI $P_\mathrm{1D}$ & Ly$\alpha$ BAO DR2 low SNR",
    # ]

    mus = []
    sigmas = []

    for s in chains:
        m = s.getMeans()[s.index["sigma8_z0"]]
        cov = s.getCov()
        i = s.index["sigma8_z0"]
        sig = np.sqrt(cov[i, i])
        mus.append(m)
        sigmas.append(sig)

    # --- external constraints ---
    labels = chain_labels + [r"DESI DR2 BAO & DR1 FS", "DESY6 all probes", "CMB-SPA"]
    mu = np.array(mus + [mu_desi, mu_des_all, mu_cmb])
    sigma = np.array(sigmas + [sigma_desi, sigma_des_all, sigma_cmb])

    # reverse order
    labels = labels[::-1]
    mu = mu[::-1]
    sigma = sigma[::-1]

    colors = [f"C{i}" for i in range(len(mu))]
    colors = colors[::-1]
    y = np.arange(len(labels))

    ftsize = 18
    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(len(mu)):
        ax.errorbar(
            mu[i],
            y[i],
            xerr=sigma[i],
            fmt="o",
            lw=2,
            capsize=4,
            color=colors[i],
        )

    ax.axhline(2.5, linestyle=":", color="k", lw=2)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=ftsize)
    ax.set_xlabel(r"$\sigma_8$", fontsize=ftsize + 2)
    ax.tick_params(labelsize=ftsize)
    # ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig("figs/sig8.pdf")
    plt.savefig("figs/sig8.png")


def plot_sig8z(samples, ftsize=20):

    # load CMB-SPA
    try:
        data = np.load("int_data_figs/sig8_cmb_spa.npy", allow_pickle=True).item()
    except:
        print("setting CMB-SPA")
        set_cmbspa_sig8z()
        data = np.load("int_data_figs/sig8_cmb_spa.npy", allow_pickle=True).item()

    zplot = data["z"]
    sig8 = data["sig8"]
    # it is f, not fsig8
    f = data["fsig8"]
    fsig8 = f * sig8

    # load DESI FS
    try:
        data = np.load("int_data_figs/sig8_desi.npy", allow_pickle=True).item()
    except:
        print("setting DESI FS")
        set_desi_sig8z()
        data = np.load("int_data_figs/sig8_desi.npy", allow_pickle=True).item()

    desi_zplot = data["zeff"]
    desi_sig8 = data["sig8"]
    f = data["f"]
    desi_fsig8 = f * desi_sig8

    # DES 3x2pt LCDM, Fig 12 https://arxiv.org/abs/2207.05766
    bins_zs_DES = np.array([0, 0.4, 0.55, 0.7, 1.5])
    plot_zs_DES = np.array([0.30, 0.48, 0.63, 0.80])
    sig8_DES = np.array([0.64, 0.585, 0.505, 0.46])
    errsig8_DES = np.array([0.045, 0.055, 0.055, 0.065])

    # --- GetDist chains ---
    chains = [samples["p1d"], samples["p1d_dr1"], samples["p1d_dr2"]]
    chain_labels = [
        r"DESI $P_\mathrm{1D}$",
        r"DESI $P_\mathrm{1D}$ & Ly$\alpha$ BAO DR1",
        r"DESI $P_\mathrm{1D}$ & Ly$\alpha$ BAO DR2",
    ]

    mus = []
    sigmas = []
    zeff = 2.33

    for s in chains:
        m = s.getMeans()[s.index["sigma8"]]
        cov = s.getCov()
        i = s.index["sigma8"]
        sig = np.sqrt(cov[i, i])
        mus.append(m)
        sigmas.append(sig)

    fig, ax = plt.subplots(figsize=(8, 6))

    # vertical dotted lines for bin edges
    # for z in bins_zs_DES:
    #     ax.axvline(z, linestyle=':')

    # CMB-SPA
    mean = np.mean(sig8, axis=0)
    std = np.std(sig8, axis=0)
    _ = (zplot > -1) & (zplot < 2.7)
    ax.fill_between(
        zplot[_],
        (mean - std)[_],
        (mean + std)[_],
        alpha=0.3,
        label="CMB-SPA",
        color="C1",
    )

    # DES
    # cmb_at_des = np.interp(plot_zs_DES, zplot, mean)
    ax.errorbar(plot_zs_DES, sig8_DES, yerr=errsig8_DES, fmt="o", label="DESY6 3x2pt")

    # DESI FS
    ax.errorbar(
        desi_zplot,
        np.mean(desi_sig8, axis=0),
        yerr=np.std(desi_sig8, axis=0),
        fmt="o",
        label="DESI DR1 FS",
        color="C2",
    )

    # Ours
    # cmb_at_ours = np.interp(zeff, zplot, mean)
    colors = ["C0", "C3", "C4"]
    shift = [-0.05, 0, 0.05]
    for i in range(len(mus)):
        ax.errorbar(
            zeff + shift[i],
            mus[i],
            yerr=sigmas[i],
            fmt="o",
            color=colors[i],
            label=chain_labels[i],
        )

    ax.set_xlabel(r"$z$", fontsize=ftsize)
    ax.set_ylabel(r"$\sigma_8(z)$", fontsize=ftsize)
    ax.tick_params(labelsize=ftsize)
    ax.legend(fontsize=ftsize - 2)

    plt.tight_layout()
    plt.savefig("figs/sig8z.pdf")
    plt.savefig("figs/sig8z.png")

    return


def plot_fsig8z(samples, ftsize=20):

    # load CMB-SPA
    try:
        data = np.load("sig8_cmb_spa.npy", allow_pickle=True).item()
    except:
        print("setting CMB-SPA")
        set_cmbspa_sig8z()
        data = np.load("sig8_cmb_spa.npy", allow_pickle=True).item()

    zplot = data["z"]
    sig8 = data["sig8"]
    # it is f, not fsig8
    f = data["fsig8"]
    fsig8 = f * sig8

    # Cuceu+2026 Eq. 23 Lya full shape https://arxiv.org/abs/2509.15308
    lya_z_fsig8 = 2.33
    lya_fsig8 = 0.37
    lya_fsig8_err = np.sqrt(0.060**2 + 0.033**2)

    # DESI+2025 full shape all Fig. 14 (extract from figure) https://arxiv.org/abs/2411.12021
    z = np.array([0.30, 0.52, 0.72, 0.95, 1.35, 1.50])
    fs8 = np.array([0.378, 0.515, 0.485, 0.422, 0.375, 0.435])
    err_fs8 = np.array([0.095, 0.060, 0.055, 0.045, 0.035, 0.045])

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        z,
        fs8,
        yerr=err_fs8,
        fmt="o",
        color="C0",
        label="DESI DR1 FS GAL+QSO",
        alpha=0.5,
    )
    ax.errorbar(
        lya_z_fsig8,
        lya_fsig8,
        yerr=lya_fsig8_err,
        fmt="o",
        color="C2",
        label="DESI DR1 FS Lya",
        alpha=0.5,
    )

    # CMB-SPA
    mean = np.mean(fsig8, axis=0)
    std = np.std(fsig8, axis=0)
    _ = (zplot > -1) & (zplot < 2.7)
    ax.fill_between(
        zplot[_],
        (mean - std)[_],
        (mean + std)[_],
        alpha=0.3,
        label="CMB-SPA",
        color="C1",
    )

    # --- GetDist chains ---
    chains = [samples["p1d"], samples["p1d_dr1"], samples["p1d_dr2"]]
    chain_labels = [
        r"DESI $P_\mathrm{1D}$",
        r"DESI $P_\mathrm{1D}$ & Ly$\alpha$ BAO DR1",
        r"DESI $P_\mathrm{1D}$ & Ly$\alpha$ BAO DR2",
    ]

    # ours
    colors = ["C0", "C3", "C4"]
    mus = []
    sigmas = []
    zeff = 2.33

    for s in chains:
        m = s.getMeans()[s.index["sigma8"]]
        cov = s.getCov()
        i = s.index["sigma8"]
        sig = np.sqrt(cov[i, i])
        mus.append(m)
        sigmas.append(sig)

    shift = [-0.05, 0, 0.05]
    for i in range(len(mus)):
        ax.errorbar(
            zeff + shift[i],
            mus[i],
            yerr=sigmas[i],
            fmt="o",
            color=colors[i],
            label=chain_labels[i],
        )

    ax.set_xlabel(r"$z$", fontsize=ftsize)
    ax.set_ylabel(r"$f\sigma_8(z)$", fontsize=ftsize)
    ax.tick_params(labelsize=ftsize)
    ax.legend(fontsize=ftsize - 4)

    plt.tight_layout()
    plt.savefig("figs/fsig8z.pdf")
    plt.savefig("figs/fsig8z.png")


def plot_sig8z233(samples, ftsize=20):
    # --- plotting ---
    g = plots.get_subplot_plotter(width_inch=8)
    g.settings.lab_fontsize = ftsize
    g.settings.axes_fontsize = ftsize
    g.settings.legend_fontsize = ftsize
    g.settings.num_plot_contours = 2  # 68%, 95%

    g.triangle_plot(
        [samples["p1d"], samples["p1d_dr1"], samples["p1d_dr2"]],
        filled=[False, True, True, True],
        params=["sigma8"],
        contour_colors=["C0", "C1", "C2", "C3"],
        contour_ls=[
            "-",
            "--",
            "-.",
            ":",
        ],
        contour_lws=[3.0, 3.0, 3.0, 2.0],
    )

    plt.tight_layout()
    plt.savefig("figs/sig8z233.pdf")
    plt.savefig("figs/sig8z233.png")


def plot_compressed(samples, ftsize=20):
    # --- plotting ---
    g = plots.get_subplot_plotter(width_inch=8)
    g.settings.lab_fontsize = ftsize
    g.settings.axes_fontsize = ftsize
    g.settings.legend_fontsize = ftsize
    g.settings.num_plot_contours = 2  # 68%, 95%

    g.triangle_plot(
        [samples["p1d"], samples["p1d_dr1"], samples["p1d_dr2"]],
        filled=[False, True, True, True],
        # params=["Delta2star", "nstar", "sigma8"],
        params=["Delta2star", "nstar"],
        contour_colors=["C0", "C1", "C2", "C3"],
        contour_ls=[
            "-",
            "--",
            "-.",
            ":",
        ],
        contour_lws=[3.0, 3.0, 3.0, 2.0],
    )

    plt.tight_layout()
    plt.savefig("figs/delta2star_nstar.pdf")
    plt.savefig("figs/delta2star_nstar.png")


def plot_bdelta_beta_beta(samples, ftsize=20):

    # --- plotting ---
    g = plots.get_subplot_plotter(width_inch=8)
    g.settings.lab_fontsize = ftsize
    g.settings.axes_fontsize = ftsize
    g.settings.legend_fontsize = ftsize
    g.settings.num_plot_contours = 2  # 68%, 95%

    g.triangle_plot(
        [samples["p1d"], samples["p1d_dr1"], samples["p1d_dr2"]],
        filled=[False, True, True, True],
        params=["bias_delta", "bias_eta", "beta"],
        contour_colors=["C0", "C1", "C2", "C3"],
        contour_ls=[
            "-",
            "--",
            "-.",
            ":",
        ],
        contour_lws=[3.0, 3.0, 3.0, 2.0],
    )

    plt.tight_layout()
    plt.savefig("figs/bdelta_beta_beta.pdf")
    plt.savefig("figs/bdelta_beta_beta.png")


def plot_P3D_small_params(samples, ftsize=20):

    g = plots.get_subplot_plotter(width_inch=10)
    g.settings.lab_fontsize = ftsize
    g.settings.axes_fontsize = ftsize
    g.settings.legend_fontsize = ftsize
    g.settings.num_plot_contours = 2  # 68%, 95%

    g.triangle_plot(
        [samples["p1d"], samples["p1d_dr1"], samples["p1d_dr2"]],
        filled=[False, True, True, True],
        params=["q1", "q2", "kvav", "av", "bv", "kp"],
        contour_colors=["C0", "C1", "C2", "C3"],
        contour_ls=[
            "-",
            "--",
            "-.",
            ":",
        ],
        contour_lws=[3.0, 3.0, 3.0, 2.0],
    )

    plt.tight_layout()
    plt.savefig("figs/corner_arinyo.pdf")
    plt.savefig("figs/corner_arinyo.png")


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
