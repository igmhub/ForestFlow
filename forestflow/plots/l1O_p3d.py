import numpy as np
import matplotlib.pyplot as plt
from forestflow.utils import sigma68


def plot_p3d_L1O(
    archive,
    z_use,
    z_grid,
    fractional_errors,
    savename=None,
    fontsize=20,
    # nmodes_min=20,
    kmax_3d_plot=4,
    kmax_3d=3,
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

    # Extract data from Archive3D
    k_Mpc = archive.training_data[0]["k3d_Mpc"]
    mu = archive.training_data[0]["mu3d"]

    # Apply a mask to select relevant k values
    k_mask = (k_Mpc < kmax_3d_plot) & (k_Mpc > 0)
    k_Mpc = k_Mpc[k_mask]
    mu = mu[k_mask]
    # n_modes = get_modes()[k_mask]

    # Create subplots with shared y-axis and x-axis
    fig, axs = plt.subplots(
        len(z_use),
        1,
        figsize=(8, len(z_use) * 2),
        sharey=True,
        sharex=True,
        gridspec_kw={"hspace": 0.05, "wspace": 0.00},
    )

    # Define mu bins
    mu_lims = [[0, 0.06], [0.31, 0.38], [0.62, 0.69], [0.94, 1]]

    # Define colors for different mu bins
    # colors = ["navy", "crimson", "forestgreen", "goldenrod"]
    # test_sim = archive.get_testing_data("mpg_central")
    # z_grid = [d["z"] for d in test_sim]

    # Loop through redshifts
    for ii, z in enumerate(z_use):
        jj = np.argwhere(z_grid == z)[0, 0]

        axs[ii].text(1.8, -0.2, f"$z={z}$", fontsize=fontsize)
        axs[ii].axhline(y=-0.10, ls="--", color="black")
        axs[ii].axhline(y=0.10, ls="--", color="black")
        axs[ii].axhline(y=0, ls=":", color="black")
        axs[ii].set_xscale("log")
        axs[ii].set_ylim(-0.25, 0.25)
        axs[ii].set_yticks(np.array([-0.2, 0, 0.2]))

        axs[ii].axvline(x=kmax_3d, ls="--", color="k", alpha=0.75)

        # Loop through mu bins
        for mi in range(int(len(mu_lims))):
            color = "C" + str(mi)

            mu_mask = (mu >= mu_lims[mi][0]) & (mu <= mu_lims[mi][1])
            mu_lab = np.round(np.nanmedian(mu[mu_mask]), decimals=2)
            k_masked = k_Mpc[mu_mask]
            # n_modes_masked = n_modes[mu_mask]

            # ind = np.argwhere(n_modes_masked >= nmodes_min)[:, 0]
            # axs[ii].axvline(
            #     x=k_masked[ind].min(), ls="-", color=color, alpha=0.5
            # )

            # Calculate fractional error statistics
            frac_err = np.nanmedian(fractional_errors[:, jj, :], 0)
            frac_err_err = sigma68(fractional_errors[:, jj, :])

            frac_err_masked = frac_err[mu_mask]
            frac_err_err_masked = frac_err_err[mu_mask]

            # Add a line plot with shaded error region to the current subplot
            axs[ii].plot(
                k_masked,
                frac_err_masked,
                label=f"$\mu\simeq{mu_lab}$",
                color=color,
                lw=2.5,
            )
            axs[ii].fill_between(
                k_masked,
                frac_err_masked - frac_err_err_masked,
                frac_err_masked + frac_err_err_masked,
                color=color,
                alpha=0.2,
            )
            axs[ii].tick_params(axis="both", which="major", labelsize=18)
        ii += 1

    # Customize subplot appearance
    for xx, ax in enumerate(axs):
        if xx == len(axs) // 2:  # Centered y-label
            ax.yaxis.set_label_coords(-0.1, 0.5)

    axs[len(axs) - 1].set_xlabel(
        r"$k\, [\mathrm{Mpc}^{-1}]$", fontsize=fontsize
    )

    legend = axs[0].legend(loc="upper left", fontsize=fontsize - 6, ncols=4)
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


def plot_p3d_LzO(
    archive,
    z_use,
    fractional_errors,
    savename=None,
    fontsize=20,
    # nmodes_min=20,
    kmax_3d_plot=4,
    kmax_3d=3,
):
    # Extract data from Archive3D
    k_Mpc = archive.training_data[0]["k3d_Mpc"]
    mu = archive.training_data[0]["mu3d"]

    # Apply a mask to select relevant k values
    k_mask = (k_Mpc < kmax_3d_plot) & (k_Mpc > 0)
    k_Mpc = k_Mpc[k_mask]
    mu = mu[k_mask]
    # n_modes = get_modes()[k_mask]

    # Create subplots with shared y-axis and x-axis
    fig, axs = plt.subplots(
        len(z_use),
        1,
        figsize=(8, len(z_use) * 2),
        sharey=True,
        sharex=True,
        gridspec_kw={"hspace": 0.05, "wspace": 0.00},
    )

    # Define mu bins
    mu_lims = [[0, 0.06], [0.31, 0.38], [0.62, 0.69], [0.94, 1]]

    # Define colors for different mu bins
    colors = ["navy", "crimson", "forestgreen", "goldenrod"]

    # Loop through redshifts
    for ii, z in enumerate(z_use):
        axs[ii].text(2.2, -0.2, f"$z={z}$", fontsize=fontsize)
        axs[ii].axhline(y=-0.10, ls="--", color="black")
        axs[ii].axhline(y=0.10, ls="--", color="black")
        axs[ii].axhline(y=0, ls=":", color="black")
        axs[ii].set_xscale("log")
        axs[ii].set_ylim(-0.25, 0.25)

        # ind = np.argwhere(n_modes_masked >= nmodes_min)[:, 0]
        axs[ii].axvline(x=kmax_3d, ls="-", color="k", alpha=0.5)

        # Loop through mu bins
        for mi in range(int(len(mu_lims))):
            color = "C" + str(mi)

            mu_mask = (mu >= mu_lims[mi][0]) & (mu <= mu_lims[mi][1])
            mu_lab = np.round(np.nanmedian(mu[mu_mask]), decimals=2)
            k_masked = k_Mpc[mu_mask]
            n_modes_masked = n_modes[mu_mask]

            # ind = np.argwhere(n_modes_masked >= nmodes_min)[:, 0]
            # axs[ii].axvline(
            #     x=k_masked[ind].min(), ls="-", color=color, alpha=0.5
            # )

            # Calculate fractional error statistics
            frac_err = np.nanmedian(fractional_errors[:, ii, :], 0)
            frac_err_err = sigma68(fractional_errors[:, ii, :])

            frac_err_masked = frac_err[mu_mask]
            frac_err_err_masked = frac_err_err[mu_mask]

            # Add a line plot with shaded error region to the current subplot
            axs[ii].plot(
                k_masked,
                frac_err_masked,
                label=f"$\mu\simeq{mu_lab}$",
                color=color,
                lw=2.5,
            )
            axs[ii].fill_between(
                k_masked,
                frac_err_masked - frac_err_err_masked,
                frac_err_masked + frac_err_err_masked,
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

    legend = axs[0].legend(fontsize=fontsize - 6, loc="upper left", ncols=4)
    legend.get_frame().set_alpha(0.9)

    # Adjust spacing between subplots
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


def get_modes():
    n_modes = np.array(
        [
            [
                4.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                2.0,
            ],
            [
                4.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                8.0,
                0.0,
                8.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                12.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                8.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                8.0,
                2.0,
            ],
            [
                8.0,
                0.0,
                0.0,
                0.0,
                0.0,
                8.0,
                16.0,
                0.0,
                0.0,
                0.0,
                16.0,
                8.0,
                0.0,
                8.0,
                0.0,
                2.0,
            ],
            [
                20.0,
                0.0,
                0.0,
                0.0,
                32.0,
                8.0,
                0.0,
                0.0,
                24.0,
                8.0,
                0.0,
                0.0,
                16.0,
                8.0,
                8.0,
                10.0,
            ],
            [
                48.0,
                0.0,
                0.0,
                88.0,
                0.0,
                24.0,
                40.0,
                24.0,
                16.0,
                32.0,
                16.0,
                48.0,
                24.0,
                24.0,
                16.0,
                34.0,
            ],
            [
                72.0,
                0.0,
                144.0,
                0.0,
                64.0,
                64.0,
                32.0,
                64.0,
                48.0,
                80.0,
                24.0,
                80.0,
                48.0,
                56.0,
                64.0,
                44.0,
            ],
            [
                124.0,
                200.0,
                64.0,
                184.0,
                80.0,
                168.0,
                136.0,
                136.0,
                136.0,
                112.0,
                160.0,
                112.0,
                136.0,
                152.0,
                120.0,
                148.0,
            ],
            [
                212.0,
                424.0,
                296.0,
                264.0,
                320.0,
                312.0,
                312.0,
                304.0,
                288.0,
                312.0,
                288.0,
                336.0,
                272.0,
                304.0,
                312.0,
                302.0,
            ],
            [
                564.0,
                760.0,
                776.0,
                800.0,
                760.0,
                696.0,
                688.0,
                752.0,
                720.0,
                776.0,
                696.0,
                736.0,
                688.0,
                768.0,
                704.0,
                712.0,
            ],
            [
                2088.0,
                1360.0,
                1688.0,
                1624.0,
                1576.0,
                1784.0,
                1600.0,
                1720.0,
                1696.0,
                1608.0,
                1728.0,
                1672.0,
                1704.0,
                1688.0,
                1680.0,
                1676.0,
            ],
            [
                3556.0,
                4320.0,
                3960.0,
                3904.0,
                3968.0,
                3784.0,
                3968.0,
                3904.0,
                3984.0,
                4016.0,
                3928.0,
                3840.0,
                3840.0,
                3968.0,
                4000.0,
                3902.0,
            ],
            [
                9704.0,
                8480.0,
                9320.0,
                9120.0,
                9208.0,
                9232.0,
                9272.0,
                9096.0,
                9104.0,
                9184.0,
                9000.0,
                9344.0,
                9128.0,
                9256.0,
                9104.0,
                9164.0,
            ],
            [
                21280.0,
                21888.0,
                20928.0,
                21392.0,
                21576.0,
                21520.0,
                21288.0,
                21312.0,
                21568.0,
                21464.0,
                21336.0,
                21344.0,
                21408.0,
                21376.0,
                21424.0,
                21410.0,
            ],
            [
                50208.0,
                49688.0,
                49664.0,
                50536.0,
                49648.0,
                49744.0,
                50088.0,
                50072.0,
                49656.0,
                50048.0,
                49944.0,
                49960.0,
                49992.0,
                49920.0,
                49936.0,
                49882.0,
            ],
            [
                116172.0,
                117024.0,
                116776.0,
                115688.0,
                117200.0,
                116904.0,
                116392.0,
                116376.0,
                117128.0,
                116528.0,
                116408.0,
                116616.0,
                116904.0,
                116624.0,
                116672.0,
                116374.0,
            ],
            [
                273084.0,
                271408.0,
                272360.0,
                273304.0,
                271352.0,
                272616.0,
                272248.0,
                272184.0,
                272296.0,
                272504.0,
                272392.0,
                272768.0,
                271832.0,
                272456.0,
                272320.0,
                272332.0,
            ],
            [
                633764.0,
                637856.0,
                636360.0,
                635200.0,
                636256.0,
                635376.0,
                635744.0,
                635792.0,
                635688.0,
                636176.0,
                635872.0,
                635376.0,
                635896.0,
                635472.0,
                636088.0,
                635710.0,
            ],
            [
                1487696.0,
                1481224.0,
                1485136.0,
                1484816.0,
                1484024.0,
                1485416.0,
                1484216.0,
                1485304.0,
                1483928.0,
                1484504.0,
                1484832.0,
                1484600.0,
                1484440.0,
                1484360.0,
                1485464.0,
                1484450.0,
            ],
            [
                3463444.0,
                3470160.0,
                3465064.0,
                3465776.0,
                3465976.0,
                3465456.0,
                3467408.0,
                3465808.0,
                3466528.0,
                3465632.0,
                3466008.0,
                3466152.0,
                3466792.0,
                3466720.0,
                3465000.0,
                3466150.0,
            ],
        ]
    )

    return n_modes
