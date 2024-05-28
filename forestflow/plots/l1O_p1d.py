import numpy as np
import matplotlib.pyplot as plt
from forestflow.utils import sigma68


def plot_p1d_L1O(
    z_use,
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

    # Create subplots with shared y-axis
    fig, axs = plt.subplots(
        len(z_use),
        1,
        figsize=(8, len(z_use) * 2),
        sharey=True,
        sharex=True,
        gridspec_kw={"hspace": 0.05, "wspace": 0.00},
    )

    # Loop through redshifts
    color = "purple"

    for ii, z in enumerate(z_use):
        axs[ii].text(1.8, -0.05, f"$z={z}$", fontsize=fontsize)
        axs[ii].axhline(y=-0.01, ls="--", color="black")
        axs[ii].axhline(y=0.01, ls="--", color="black")
        axs[ii].axhline(y=0, ls=":", color="black")
        axs[ii].set_xscale("log")
        axs[ii].set_ylim(-0.065, 0.065)
        axs[ii].axvline(x=kmax_1d_fit, ls="--", color="k", alpha=1, lw=2)

        # Calculate fractional error statistics
        frac_err = np.nanmedian(fractional_errors[:, ii, :], 0)
        frac_err_err = sigma68(fractional_errors[:, ii, :])

        # Add a line plot with shaded error region to the current subplot
        axs[ii].plot(k_p1d_Mpc, frac_err, color=color)
        axs[ii].fill_between(
            k_p1d_Mpc,
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
        r"$k_\parallel\, [\mathrm{Mpc}^{-1}]$", fontsize=fontsize
    )

    # Adjust spacing between subplots
    fig.text(
        -0.025,
        0.5,
        r"$P_{\rm 1D}^\mathrm{emu}/P_{\rm 1D}^\mathrm{sim}-1$",
        va="center",
        rotation="vertical",
        fontsize=fontsize,
    )

    # Save the plot
    if savename:
        plt.savefig(savename, bbox_inches="tight")
