import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"

from getdist import plots
from forestflow.priors_paper import set_samples


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
        set_samples.set_cmbspa_sig8z()
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
        set_samples.set_desi_sig8z()
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
        set_samples.set_cmbspa_sig8z()
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


def table_cosmo_igm(dict_out_all):

    def format_pm(val, err, sig=2):
        import math

        if err == 0:
            return f"${val} \\pm 0$"
        exp = math.floor(math.log10(abs(err)))
        decimals = -(exp - (sig - 1))
        err_r = round(err, decimals)
        val_r = round(val, decimals)
        fmt = f"{{:.{max(decimals,0)}f}}"
        return f"${fmt.format(val_r)} \\pm {fmt.format(err_r)}$"

    params = ["Delta2_p", "mF", "sigT_Mpc", "gamma", "kF_Mpc"]

    p = dict_out_all["emu_params"]
    means = {k: np.mean(p[k], axis=0) for k in params}
    stds = {k: np.std(p[k], axis=0) for k in params}

    print("z", params[0], params[1], params[2], params[3], params[4], sep=" & ")

    for i, z in enumerate(dict_out_all["zs"]):
        row = [f"{z:.2f}"]
        for k in params:
            row.append(format_pm(means[k][i], stds[k][i]))
        print(" & ".join(row) + r" \\")

    return


def plot_bias_beta_zev(
    bao_data, dict_mapping, zmax=5, ftsize=24, plot_bias_eta=False, z0=3
):

    import matplotlib.ticker as mticker

    from scipy.optimize import curve_fit

    def fit_pow(z, a, b):
        x = (1 + z) / (1 + z0)
        return a * x**b

    def fit_pol(z, a, b, c, d):
        x = (1 + z) / (1 + z0)
        return a + b * x + c * x**2 + d * x**3

    if plot_bias_eta:
        params_labels = ["bias_delta", "beta", "bias_eta"]
    else:
        params_labels = ["bias_delta", "beta"]
    params_labels_latex = [
        r"$b_\delta$",
        r"$\beta$",
        r"$b_\eta$",
    ]

    fig, ax = plt.subplots(len(params_labels), 1, sharex=True, figsize=(10, 10))

    bao_plot = {}
    for ii, key in enumerate(["dr1_hsnr", "dr2_hsnr"]):
        bao_plot[key] = {}
        bao_plot[key]["zeff"] = 2.33
        bao_plot[key]["label"] = "BAO DR" + str(ii + 1)
        for lab in params_labels:
            bao_plot[key][lab] = {}
            bao_plot[key][lab]["mean"] = bao_data[key][lab].mean()
            bao_plot[key][lab]["std"] = bao_data[key][lab].std()

    xdisp = [-0.01, 0.01]
    data = {}
    for ii, lab in enumerate(params_labels):

        param = dict_mapping["forest_out"][lab]
        percen = np.percentile(param, [16, 84], axis=0)

        ax[ii].fill_between(
            dict_mapping["zs"],
            percen[0],
            percen[1],
            alpha=0.5,
            label=r"$P_\mathrm{1D}$ DR1",
        )

        zplot = np.linspace(np.min(dict_mapping["zs"]), np.max(dict_mapping["zs"]), 100)

        mean = np.mean(param, axis=0)
        if ii == 0:
            fit_func = fit_pol
        else:
            fit_func = fit_pol
        fit_val = curve_fit(fit_func, dict_mapping["zs"], mean)[0]
        data[params_labels_latex[ii]] = fit_val

        # data[params_latex[ii]] = fit_val
        ax[ii].plot(zplot, fit_func(zplot, *fit_val), color="C0", ls="--")

        for jj, key in enumerate(bao_plot):
            dumm = np.zeros(2)
            ax[ii].errorbar(
                dumm + bao_plot[key]["zeff"] + xdisp[jj],
                dumm + bao_plot[key][lab]["mean"],
                dumm + bao_plot[key][lab]["std"],
                fmt="o",
                label=bao_plot[key]["label"],
                color="C" + str(jj + 1),
            )

            if (ii == 0) & (key == "dr2_hsnr"):
                b1 = bao_plot[key][lab]["mean"] - bao_plot[key][lab]["std"]
                b2 = bao_plot[key][lab]["mean"] + bao_plot[key][lab]["std"]
                expo = 2.9
                bz = ((1 + zplot) / (1 + bao_plot[key]["zeff"])) ** expo
                ax[ii].fill_between(zplot, b1 * bz, b2 * bz, color="C2", alpha=0.2)

    ax[0].legend(fontsize=ftsize)

    for ii in range(len(params_labels)):
        ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)
        ax[ii].set_ylabel(params_labels_latex[ii], fontsize=ftsize)
        ax[ii].yaxis.set_major_formatter(mticker.FormatStrFormatter("% .1f"))
    ax[-1].set_xlabel(r"$z$", fontsize=ftsize)

    for key, vals in data.items():
        vals = np.round(vals, 2)
        row = " & ".join(f"{v:.2f}" for v in vals)
        print(f"{key} & {row} \\\\")

    plt.tight_layout()
    plt.savefig("figs/bias_beta_BAOvsP1D.png")
    plt.savefig("figs/bias_beta_BAOvsP1D.pdf")


def plot_p3d_small_z(dict_mapping, ftsize=20, z0=3):

    from scipy.optimize import curve_fit

    def fit_func(z, a, b, c, d):
        x = (1 + z) / (1 + z0)
        return a + b * x + c * x**2 + d * x**3

    params = ["q1", "q2", "av", "bv", "kvav", "kp"]
    params_latex = [
        r"$q_1$",
        r"$q_2$",
        r"$a_\mathrm{v}$",
        r"$b_\mathrm{v}$",
        r"$k_\mathrm{v}\,[\mathrm{Mpc}^{-1}]$",
        r"$k_\mathrm{p}\,[\mathrm{Mpc}^{-1}]$",
    ]

    fig, ax = plt.subplots(len(params) // 2, 2, sharex=True, figsize=(10, 8))
    ax = ax.flatten()

    data = {}

    for ii, par in enumerate(params):
        if par != "kvav":
            val_par = dict_mapping["forest_out"][par]
        else:
            val_par = dict_mapping["forest_out"]["kvav"] ** (
                1 / dict_mapping["forest_out"]["av"]
            )
        percen = np.percentile(val_par, [16, 84], axis=0)
        ax[ii].fill_between(dict_mapping["zs"], percen[0], percen[1], alpha=0.5)
        mean = np.mean(val_par, axis=0)
        fit_val = curve_fit(fit_func, dict_mapping["zs"], mean)[0]

        data[params_latex[ii]] = fit_val
        zplot = np.linspace(np.min(dict_mapping["zs"]), np.max(dict_mapping["zs"]), 100)
        ax[ii].plot(
            zplot,
            fit_func(zplot, *fit_val),
            color="C0",
            ls="--",
        )
        ax[ii].set_ylabel(params_latex[ii], fontsize=ftsize)
        ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)

    for key, vals in data.items():
        vals = np.round(vals, 2)
        row = " & ".join(f"{v:.2f}" for v in vals)
        print(f"{key} & {row} \\\\")

    # ax[-1].set_xticks([2, 3, 4])
    # ax[-1].set_xticklabels(["2", "3", "4"])
    ax[0].set_xlim(1.9, 4.3)

    ax[-2].set_xlabel(r"$z$", fontsize=ftsize)
    ax[-1].set_xlabel(r"$z$", fontsize=ftsize)
    plt.tight_layout()
    plt.savefig("figs/Arinyo_with_z.pdf")
    plt.savefig("figs/Arinyo_with_z.png")
