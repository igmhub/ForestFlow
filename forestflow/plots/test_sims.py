import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

from forestflow.utils import sigma68


def plot_p3d_snap(
    folder_out,
    k_Mpc,
    mu,
    p3d_sim,
    p3d_emu,
    p3d_std_emu,
    mu_bins,
    ftsize=20,
    kmax_3d=4,
    kmax_3d_fit=3,
):
    fig, axs = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, height_ratios=[3, 1]
    )
    labs = []

    for ii in range(2):
        axs[ii].axvline(x=kmax_3d_fit, ls="--", color="k", alpha=0.75, lw=2)

    for mi in range(len(mu_bins) - 1):
        if mi != p3d_sim.shape[1] - 1:
            lab = str(mu_bins[mi]) + r"$\leq\mu<$" + str(mu_bins[mi + 1])
        else:
            lab = str(mu_bins[mi]) + r"$\leq\mu\leq$" + str(mu_bins[mi + 1])
        labs.append(lab)

        color = "C" + str(mi)

        mu_mask = np.isfinite(p3d_sim[:, mi])

        # mu_mask = (
        #     (mu >= mu_lims[mi][0]) & (mu <= mu_lims[mi][1]) & (k_Mpc <= kmax_3d)
        # )
        # mu_lab = np.round(np.nanmedian(mu[mu_mask]), decimals=2)
        # n_modes_masked = n_modes[mu_mask]

        # ind = np.argwhere(n_modes_masked >= nmodes_min)[:, 0]
        # axs[0].axvline(
        #     x=(k_Mpc[mu_mask])[ind].min(), ls="-", color=color, alpha=0.75, lw=2
        # )
        # axs[1].axvline(
        #     x=(k_Mpc[mu_mask])[ind].min(), ls="-", color=color, alpha=0.75, lw=2
        # )

        # labs.append(f"$\mu\simeq{mu_lab}$")

        axs[0].plot(
            k_Mpc[mu_mask, mi],
            p3d_sim[mu_mask, mi],
            ":o",
            color=color,
            lw=3,
        )
        axs[0].plot(
            k_Mpc[mu_mask, mi],
            p3d_emu[mu_mask, mi],
            ls="-",
            color=color,
            lw=3,
            alpha=0.75,
        )
        axs[0].fill_between(
            k_Mpc[mu_mask, mi],
            p3d_emu[mu_mask, mi] - p3d_std_emu[mu_mask, mi],
            p3d_emu[mu_mask, mi] + p3d_std_emu[mu_mask, mi],
            alpha=0.2,
            color=color,
        )

        axs[1].plot(
            k_Mpc[mu_mask, mi],
            p3d_emu[mu_mask, mi] / p3d_sim[mu_mask, mi] - 1,
            ls="-",
            color=color,
            lw=3,
            alpha=0.75,
        )

        axs[1].fill_between(
            k_Mpc[mu_mask, mi],
            (p3d_emu[mu_mask, mi] - p3d_std_emu[mu_mask, mi])
            / p3d_sim[mu_mask, mi]
            - 1,
            (p3d_emu[mu_mask, mi] + p3d_std_emu[mu_mask, mi])
            / p3d_sim[mu_mask, mi]
            - 1,
            alpha=0.2,
            color=color,
        )

    axs[1].axhline(0, ls=":", color="k", alpha=1, lw=1)
    axs[1].axhline(-0.1, ls="--", color="k", alpha=1, lw=1)
    axs[1].axhline(0.1, ls="--", color="k", alpha=1, lw=1)

    axs[0].tick_params(axis="both", which="major", labelsize=ftsize)
    axs[1].tick_params(axis="both", which="major", labelsize=ftsize)
    axs[1].set_xlabel(r"$k\, [\mathrm{Mpc}^{-1}]$", fontsize=ftsize)
    axs[0].set_ylabel(
        r"$P_\mathrm{3D}(k, \mu)/P_{\rm lin}(k)$", fontsize=ftsize
    )
    axs[1].set_ylabel(r"Residual", fontsize=ftsize)
    axs[1].set_ylim(-0.22, 0.22)
    axs[0].set_xscale("log")

    # create manual symbols for legend
    handles = []
    for ii in range(4):
        handles.append(mpatches.Patch(color="C" + str(ii), label=labs[ii]))
    legend1 = axs[0].legend(
        handles=handles, ncol=1, fontsize=ftsize - 2, loc="upper right"
    )

    line1 = Line2D(
        [0], [0], label=r"Simulation", color="gray", ls=":", marker="o", lw=2
    )
    line2 = Line2D([0], [0], label=r"ForestFlow", color="gray", ls="-", lw=2)
    legend2 = axs[0].legend(
        handles=[line1, line2], ncol=1, fontsize=ftsize - 2, loc="upper left"
    )
    axs[0].add_artist(legend1)
    axs[0].add_artist(legend2)

    # axs.legend(loc='upper right', ncol=1, fontsize=ftsize-2)
    plt.tight_layout()

    for ext in [".png", ".pdf"]:
        plt.savefig(folder_out + "p3d_snap" + ext)


def plot_p1d_snap(
    folder_out,
    k_p1d_Mpc,
    p1d_sim,
    p1d_emu,
    p1d_std_emu,
    ftsize=20,
    # fact_kmin=4,
    kmax_1d=4,
    kmax_1d_fit=3,
):
    fig, axs = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, height_ratios=[3, 1]
    )
    # kmin = 2 * np.pi / 67.5 * fact_kmin

    for ii in range(2):
        axs[ii].axvline(x=kmax_1d_fit, ls="--", color="k", alpha=0.75, lw=2)

    # for ii in range(2):
    #     axs[ii].axvline(x=kmin, ls="-", color="C0", alpha=0.75, lw=2)

    mask = k_p1d_Mpc <= kmax_1d

    axs[0].plot(
        k_p1d_Mpc[mask],
        p1d_sim[mask],
        ":o",
        color="C0",
        label="Simulation",
        lw=3,
    )
    axs[0].plot(
        k_p1d_Mpc[mask],
        p1d_emu[mask],
        ls="-",
        color="C1",
        label="ForestFlow",
        lw=3,
        alpha=0.75,
    )
    axs[0].fill_between(
        k_p1d_Mpc[mask],
        p1d_emu[mask] - p1d_std_emu[mask],
        p1d_emu[mask] + p1d_std_emu[mask],
        alpha=0.2,
        color="C1",
    )

    axs[1].axhline(0, ls=":", color="k", alpha=1, lw=1)
    axs[1].axhline(-0.01, ls="--", color="k", alpha=1, lw=1)
    axs[1].axhline(0.01, ls="--", color="k", alpha=1, lw=1)
    axs[1].plot(
        k_p1d_Mpc[mask],
        p1d_emu[mask] / p1d_sim[mask] - 1,
        "-",
        color="C0",
        lw=3,
    )
    axs[1].fill_between(
        k_p1d_Mpc[mask],
        (p1d_emu[mask] - p1d_std_emu[mask]) / p1d_sim[mask] - 1,
        (p1d_emu[mask] + p1d_std_emu[mask]) / p1d_sim[mask] - 1,
        alpha=0.2,
        color="C0",
    )

    axs[0].tick_params(axis="both", which="major", labelsize=ftsize)
    axs[1].tick_params(axis="both", which="major", labelsize=ftsize)
    axs[1].set_xlabel(r"$k_\parallel\, [\mathrm{Mpc}^{-1}]$", fontsize=ftsize)
    axs[0].set_ylabel(
        r"$\pi^{-1}\, k_\parallel\, P_\mathrm{1D}(k_\parallel)$",
        fontsize=ftsize,
    )
    axs[1].set_ylabel(r"Residual", fontsize=ftsize)
    axs[0].set_xscale("log")
    axs[1].set_ylim([-0.052, 0.052])
    # plt.legend()

    axs[0].legend(loc="upper left", ncol=1, fontsize=ftsize - 2)
    plt.tight_layout()

    for ext in [".png", ".pdf"]:
        plt.savefig(folder_out + "p1d_snap" + ext)


def plot_p3d_test_sims(
    sim_labels,
    k_Mpc,
    mu,
    residual,
    mu_bins,
    savename=None,
    fontsize=20,
    kmax_3d_fit=3,
):
    """
    Plot the fractional errors in the P3D statistic for different redshifts and mu bins.

    Parameters:
    - archive: The dataset archive containing the training data.
    - fractional_errors: Fractional errors in the P3D statistic for different redshifts and mu bins.
    - savename: The name of the file to save the plot.

    Returns:
    None

    Plots:
    - Subplots showing fractional errors in P3D for different redshifts and mu bins.

    """

    dict_labels = {
        "mpg_central": "Central",
        "mpg_seed": "Seed",
        "mpg_growth": "Growth",
        "mpg_neutrinos": "Neutrinos",
        "mpg_curved": "Curved",
        "mpg_running": "Running",
        "mpg_reio": "Reionization",
    }

    # Create subplots with shared y-axis and x-axis
    fig, axs = plt.subplots(
        len(sim_labels),
        1,
        figsize=(8, 2 * len(sim_labels)),
        sharey=True,
        sharex=True,
        gridspec_kw={"hspace": 0.05, "wspace": 0.00},
    )
    try:
        l = len(axs)
    except:
        axs = [axs]

    # Loop through redshifts
    for ii in range(len(sim_labels)):
        label = dict_labels[sim_labels[ii]]
        axs[ii].text(1.2, -0.2, label, fontsize=fontsize)
        axs[ii].axhline(y=-0.10, ls="--", color="black")
        axs[ii].axhline(y=0.10, ls="--", color="black")
        axs[ii].axhline(y=0, ls=":", color="black")
        axs[ii].set_xscale("log")
        axs[ii].set_ylim(-0.25, 0.25)
        axs[ii].set_yticks(np.array([-0.2, 0, 0.2]))

        axs[ii].axvline(x=kmax_3d_fit, ls="--", color="k", alpha=1, lw=2)

        # Loop through mu bins
        for mi in range(len(mu_bins) - 1):
            if mi != residual.shape[-1] - 1:
                lab = str(mu_bins[mi]) + r"$\leq\mu<$" + str(mu_bins[mi + 1])
            else:
                lab = str(mu_bins[mi]) + r"$\leq\mu\leq$" + str(mu_bins[mi + 1])
            # labs.append(lab)
            if ii == mi:
                lab = lab
            else:
                lab = None

            color = "C" + str(mi)

            mu_mask = np.isfinite(k_Mpc[:, mi])

            # Calculate fractional error statistics
            frac_err = np.mean(residual[ii, :, mu_mask, mi], axis=1)
            frac_err_err = np.std(residual[ii, :, mu_mask, mi], axis=1)

            # Add a line plot with shaded error region to the current subplot
            axs[ii].plot(
                k_Mpc[mu_mask, mi],
                frac_err,
                label=lab,
                color=color,
                lw=2.5,
            )
            axs[ii].fill_between(
                k_Mpc[mu_mask, mi],
                frac_err - frac_err_err,
                frac_err + frac_err_err,
                color=color,
                alpha=0.2,
            )
            axs[ii].tick_params(axis="both", which="major", labelsize=18)

    # Customize subplot appearance
    for xx, ax in enumerate(axs):
        if xx == len(axs) // 2:  # Centered y-label
            ax.yaxis.set_label_coords(-0.1, 0.5)

    axs[len(axs) - 1].set_xlabel(
        r"$k\, [\mathrm{Mpc}^{-1}]$", fontsize=fontsize
    )

    for ii in range(len(mu_bins) - 1):
        legend = axs[ii].legend(
            loc="upper left", ncols=1, fontsize=fontsize - 6
        )
        legend.get_frame().set_alpha(0.9)

    # Adjust spacing between subplots
    # plt.tight_layout()
    fig.text(
        0.01,
        0.5,
        r"$P_{\rm 3D}^\mathrm{emu}/P_{\rm 3D}^\mathrm{sim}-1$",
        va="center",
        rotation="vertical",
        fontsize=fontsize,
    )

    # Save the plot
    if savename:
        plt.savefig(savename, bbox_inches="tight")


def plot_p1d_test_sims(
    sim_labels,
    k_p1d_Mpc,
    fractional_errors,
    savename=None,
    fontsize=20,
    kmax_1d_fit=3,
):
    """
    Plot the fractional errors in the P1D statistic for different redshifts.

    Parameters:
    - fractional_errors: Fractional errors in the P1D statistic for different redshifts.
    - savename: The name of the file to save the plot.

    Returns:
    None

    Plots:
    - Subplots showing fractional errors in P1D for different redshifts.

    """

    # kmin = 2 * np.pi / 67.5 * fact_kmin

    dict_labels = {
        "mpg_central": "Central",
        "mpg_seed": "Seed",
        "mpg_growth": "Growth",
        "mpg_neutrinos": "Neutrinos",
        "mpg_curved": "Curved",
        "mpg_running": "Running",
        "mpg_reio": "Reionization",
    }

    colors = ["purple"]
    fig, ax = plt.subplots(
        ncols=1,
        nrows=len(sim_labels),
        figsize=(8, 2 * len(sim_labels)),
        sharey=True,
        sharex=True,
        gridspec_kw={"hspace": 0.05, "wspace": 0.00},
    )
    try:
        l = len(ax)
    except:
        ax = [ax]

    for c in range(len(sim_labels)):
        # ax[c].axhspan(-0.01, 0.01, color="gray", alpha=0.3)
        ax[c].axhline(y=0.01, color="black", ls="--", alpha=0.8)
        ax[c].axhline(y=-0.01, color="black", ls="--", alpha=0.8)
        ax[c].axhline(y=0, color="black", ls=":", alpha=0.8)
        label = dict_labels[sim_labels[c]]
        ax[c].text(1.15, -0.05, label, fontsize=fontsize)
        ax[c].axvline(x=kmax_1d_fit, ls="--", color="k", alpha=1, lw=2)

        frac_err = np.median(fractional_errors[c], axis=0)
        # frac_err_err = sigma68(fractional_errors[c])
        frac_err_err = np.std(fractional_errors[c], axis=0)

        ax[c].plot(
            k_p1d_Mpc,
            frac_err,
            ls="-",
            lw=2,
            color=colors[0],
        )
        ax[c].fill_between(
            k_p1d_Mpc,
            frac_err - frac_err_err,
            frac_err + frac_err_err,
            color=colors[0],
            alpha=0.2,
        )
        ax[c].tick_params(axis="both", which="major", labelsize=18)

    ax[0].set_ylim(-0.065, 0.065)
    ax[0].set_xscale("log")

    # Customize subplot appearance
    for xx, axi in enumerate(ax):
        if xx == len(ax) // 2:  # Centered y-label
            axi.yaxis.set_label_coords(-0.1, 0.5)
    ax[-1].set_xlabel(r"$k_\parallel\, [\mathrm{Mpc}^{-1}]$", fontsize=fontsize)

    fig.text(
        0.0,
        0.5,
        r"$P_{\rm 1D}^\mathrm{emu}/P_{\rm 1D}^\mathrm{sim}-1$",
        va="center",
        rotation="vertical",
        fontsize=fontsize,
    )

    # savename = None
    if savename:
        plt.savefig(savename, bbox_inches="tight")
