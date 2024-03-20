import numpy as np
import matplotlib.pyplot as plt
from forestflow.utils import sigma68


def plot_p1d_L1O(
    archive,
    z_use,
    fractional_errors,
    savename=None,
    fontsize=20,
    fact_kmin=4,
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

    kmin = 2 * np.pi / 67.5 * fact_kmin

    # Create subplots with shared y-axis
    fig, axs = plt.subplots(
        len(z_use),
        1,
        figsize=(8, 16),
        sharey=True,
        sharex=True,
        gridspec_kw={"hspace": 0.05, "wspace": 0.00},
    )

    # Extract data from Archive3D
    k_Mpc = archive.training_data[0]["k_Mpc"]
    # Apply a mask to select relevant k values
    k1d_mask = (k_Mpc < 5) & (k_Mpc > 0)
    k1d_sim = k_Mpc[k1d_mask]

    test_sim = archive.get_testing_data("mpg_central")
    z_grid = [d["z"] for d in test_sim]

    # Loop through redshifts
    ii = 0
    color = "purple"
    for i0, z in enumerate(z_grid):
        if z not in z_use:
            continue

        axs[ii].text(3.2, 0.025, f"$z={z}$", fontsize=fontsize)
        axs[ii].axhline(y=-0.01, ls="--", color="black")
        axs[ii].axhline(y=0.01, ls="--", color="black")
        axs[ii].axhline(y=0, ls=":", color="black")
        axs[ii].set_xscale("log")
        axs[ii].set_ylim(-0.035, 0.035)
        axs[ii].axvline(x=kmin, ls="-", color=color, alpha=0.5)

        # Calculate fractional error statistics
        frac_err = np.nanmedian(fractional_errors[:, i0, :], 0)
        frac_err_err = sigma68(fractional_errors[:, i0, :])

        # Add a line plot with shaded error region to the current subplot
        axs[ii].plot(k1d_sim, frac_err, color=color)
        axs[ii].fill_between(
            k1d_sim,
            frac_err - frac_err_err,
            frac_err + frac_err_err,
            color=color,
            alpha=0.2,
        )

        axs[ii].tick_params(axis="both", which="major", labelsize=18)

        ii += 1

    # Customize subplot appearance
    for xx, ax in enumerate(axs):
        if xx == len(axs) // 2:  # Centered y-label
            ax.yaxis.set_label_coords(-0.1, 0.5)

    axs[len(axs) - 1].set_xlabel(r"$k_\parallel$ [1/Mpc]", fontsize=fontsize)

    # Adjust spacing between subplots
    fig.text(
        0.0,
        0.5,
        r"$P_{\rm 1D}^\mathrm{emu}/P_{\rm 1D}^\mathrm{sim}-1$",
        va="center",
        rotation="vertical",
        fontsize=fontsize,
    )

    # Save the plot
    if savename:
        plt.savefig(savename, bbox_inches="tight")


def plot_p1d_LzO(
    archive,
    z_use,
    fractional_errors,
    savename=None,
    fontsize=20,
    fact_kmin=4,
):
    kmin = 2 * np.pi / 67.5 * fact_kmin

    # Create subplots with shared y-axis
    fig, axs = plt.subplots(
        len(z_use),
        1,
        figsize=(8, 8),
        sharey=True,
        sharex=True,
        gridspec_kw={"hspace": 0.05, "wspace": 0.00},
    )

    # Extract data from Archive3D
    k_Mpc = archive.training_data[0]["k_Mpc"]
    # Apply a mask to select relevant k values
    k1d_mask = (k_Mpc < 5) & (k_Mpc > 0)
    k1d_sim = k_Mpc[k1d_mask]

    # Loop through redshifts
    color = "purple"
    for ii, z in enumerate(z_use):
        axs[ii].text(3.2, 0.025, f"$z={z}$", fontsize=fontsize)
        axs[ii].axhline(y=-0.01, ls="--", color="black")
        axs[ii].axhline(y=0.01, ls="--", color="black")
        axs[ii].axhline(y=0, ls=":", color="black")
        axs[ii].set_xscale("log")
        axs[ii].set_ylim(-0.035, 0.035)
        axs[ii].axvline(x=kmin, ls="-", color=color, alpha=0.5)

        # Calculate fractional error statistics
        frac_err = np.nanmedian(fractional_errors[:, ii, :], 0)
        frac_err_err = sigma68(fractional_errors[:, ii, :])

        # Add a line plot with shaded error region to the current subplot
        axs[ii].plot(k1d_sim, frac_err, color=color)
        axs[ii].fill_between(
            k1d_sim,
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

    axs[len(axs) - 1].set_xlabel(r"$k_\parallel$ [1/Mpc]", fontsize=fontsize)

    # Adjust spacing between subplots
    fig.text(
        0.0,
        0.5,
        r"$P_{\rm 1D}^\mathrm{emu}/P_{\rm 1D}^\mathrm{sim}-1$",
        va="center",
        rotation="vertical",
        fontsize=fontsize,
    )

    # Save the plot
    if savename:
        plt.savefig(savename, bbox_inches="tight")
