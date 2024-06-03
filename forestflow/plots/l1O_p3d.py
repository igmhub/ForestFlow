import numpy as np
import matplotlib.pyplot as plt
from forestflow.utils import sigma68


def plot_p3d_L1O(
    z_use,
    k_Mpc,
    mu,
    residual,
    mu_bins,
    savename=None,
    fontsize=20,
    kmax_3d_fit=3,
    legend=False,
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

    # Create subplots with shared y-axis and x-axis
    fig, axs = plt.subplots(
        len(z_use),
        1,
        figsize=(8, len(z_use) * 2),
        sharey=True,
        sharex=True,
        gridspec_kw={"hspace": 0.05, "wspace": 0.00},
    )

    # Loop through redshifts
    for ii, z in enumerate(z_use):
        axs[ii].text(1.8, -0.2, f"$z={z}$", fontsize=fontsize)
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
            frac_err = np.mean(residual[:, ii, mu_mask, mi], axis=0)
            frac_err_err = np.std(residual[:, ii, mu_mask, mi], axis=0)

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

    if legend:
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
