import matplotlib.pyplot as plt
import numpy as np

from forestflow.likelihood import Likelihood
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from forestflow.plot_routines import plot_template
from forestflow.utils import params_numpy2dict, sigma68


# def norm_params(xx, direction="L"):
#     pmin = np.array([1, 0.11, 0, 0.3, 0.01, 0.9, 9, 0])
#     pdiff = np.array([1.5, 2.6, 2.6, 3.6, 1.1, 1.7, 30, 2.4])

#     if direction == "R":
#         xx = (xx - pmin[None, :]) / pdiff[None, :] - 0.5
#     else:
#         xx = (xx + 0.5) * pdiff[None, :] + pmin[None, :]

#     return xx


def plot_test_parz(Archive3D, p3d_emu, sim_label):
    """
    Precision of emulator for target sim
    """

    # load data
    testing_data = Archive3D.get_testing_data(sim_label)

    # get params
    input_params = np.zeros((len(testing_data), len(testing_data[0]["Arinyo"])))
    predict_params = np.zeros_like(input_params)
    zs = np.zeros((len(testing_data)))

    for jj in range(len(testing_data)):
        zs[jj] = testing_data[jj]["z"]
        #_cosmo_params = np.zeros(len(Archive3D.emu_params))
        #for ii, par in enumerate(Archive3D.emu_params):
        #    _cosmo_params[ii] = testing_data[jj][par]
        predict_params[jj] = p3d_emu.predict_Arinyos([testing_data[jj]])
        input_params[jj] = list(testing_data[jj]["Arinyo"].values())

    # make sure bias negative (dependence on bias square)
    input_params[:, 0] = -np.abs(input_params[:, 0])

    fig, ax = plt.subplots(4, 2, sharex=True)
    ax = ax.reshape(-1)
    for ii in range(predict_params.shape[1]):
        ax[ii].plot(zs, input_params[:, ii], "C0o:")
        ax[ii].plot(zs, predict_params[:, ii], "C1-")
        lab = list(testing_data[0]["Arinyo"].keys())[ii]
        ax[ii].set_ylabel(lab)

    ax[6].set_xlabel(r"$z$")
    ax[7].set_xlabel(r"$z$")

    # plt.plot(eval_params[ii], pred_params[ii, :, 1])

    plt.tight_layout()
    # plt.savefig("params_cosmo_1.png")


def plot_test_p3d(ind_book, Archive3D, p3d_emu, sim_label, plot_emu=True):
    """
    Precision of emulator for target sim
    """

    # load data
    try:
        getattr(Archive3D, "testing_data")
    except:
        testing_data = Archive3D.get_testing_data(sim_label)
    else:
        testing_data = Archive3D.testing_data

    # get params
    input_params = np.zeros((len(testing_data), len(testing_data[0]["Arinyo"])))
    predict_params = np.zeros_like(input_params)
    zs = np.zeros((len(testing_data)))

    for jj in range(len(testing_data)):
        zs[jj] = testing_data[jj]["z"]
        #_cosmo_params = np.zeros(len(Archive3D.emu_params))
        #for ii, par in enumerate(Archive3D.emu_params):
        #    _cosmo_params[ii] = testing_data[jj][par]
        if plot_emu:
            predict_params[jj] = p3d_emu.predict_Arinyos([testing_data[jj]])
        input_params[jj] = list(testing_data[jj]["Arinyo"].values())

    # make sure bias negative (dependence on bias square)
    # input_params[:, 0] = -np.abs(input_params[:, 0])

    like = Likelihood(
        testing_data[ind_book], Archive3D.rel_err_p3d, Archive3D.rel_err_p1d
    )

    emu_params = np.abs(predict_params[ind_book])

    # fit_pars = testing_data[ind_book]["Arinyo"]
    fit_pars = params_numpy2dict(input_params[ind_book])
    emu_pars = params_numpy2dict(emu_params)

    print("fit", fit_pars)
    print("emu", emu_pars)

    # save_fig = "test.png"
    save_fig = None
    if plot_emu:
        plot_compare_p3d_smooth(
            like.like,
            fit_pars,
            emu_pars,
            sim_label=sim_label,
            err_bar_all=True,
            save_fig=save_fig,
        )
    else:
        plot_compare_p3d_smooth(
            like.like,
            fit_pars,
            emu_pars,
            sim_label=sim_label,
            err_bar_all=True,
            save_fig=save_fig,
        )


def plot_compare_p3d_smooth(
    like,
    parameters1,
    parameters2=None,
    save_fig=None,
    err_bar_all=False,
    sim_label="",
    plot_data=True,
    plot_p1d=True,
    plot_legend_1=False,
):
    """
    Compare data and best-fitting model.

    Parameters:
        parameters (dict): Dictionary of fitting parameters.
        error_fit_3d (array, optional): Array of 3D fitting errors (default: None).
        error_fit_1d (array, optional): Array of 1D fitting errors (default: None).
        save_fig (str, optional): File path to save the figure (default: None).
        err_bar_all (bool, optional): Flag to enable error bars for all data points (default: False).

    Note:
        - This method compares the data and the best-fitting model.
        - It plots the comparison using subplots for 3D fitting, 1D fitting, and cosmic variance errors.
        - The `parameters` argument should be a dictionary of fitting parameters required to compute the model.
        - The `error_fit_3d` and `error_fit_1d` arrays provide the fitting errors for the 3D and 1D data, respectively.
        - If `error_fit_3d` is provided and `err_bar_all` is True, error bars will be shown for all data points in the 3D fitting plot.
        - If `error_fit_1d` is provided and `err_bar_all` is True, error bars will be shown for all data points in the 1D fitting plot.
        - If `save_fig` is provided, the plot will be saved to the specified file path.

    Returns:
        None
    """

    if plot_p1d:
        fig, ax = plt.subplots(
            3,
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1, 1]},
            figsize=(8, 8),
        )
    else:
        fig, ax = plt.subplots(
            2,
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
            figsize=(8, 6),
        )

    tit = sim_label + r" $z=$" + str(like.data["z"][0])
    fig.suptitle(tit, fontsize=17)

    # compute best-fitting model
    p3d_best1 = like.get_model_3d(parameters=parameters1)
    if plot_p1d:
        p1d_best1 = like.get_model_1d(parameters=parameters1)
    if parameters2 is not None:
        p3d_best2 = like.get_model_3d(parameters=parameters2)
        if plot_p1d:
            p1d_best2 = like.get_model_1d(parameters=parameters2)

    # iterate over wedges
    nmus = like.data["k3d"].shape[1]
    mubins = np.linspace(0, 1, nmus + 1)
    mu_use = np.linspace(0, nmus - 1, 4, dtype=int)

    for imu, ii in enumerate(mu_use):
        col = "C" + str(imu)

        # only plot when data is not nan
        mask = like.ind_fit3d[:, ii]
        kk = like.data["k3d"][mask, ii]

        ## upper panel ##
        norm = like.data["Plin"][mask, ii] * kk**3 / 2 / np.pi**2
        if plot_data:
            line1 = ax[0].plot(
                kk,
                like.data["p3d"][mask, ii] / norm,
                color=col,
                marker="o",
                linestyle=":",
            )

        line1 = ax[0].plot(
            kk,
            p3d_best1[mask, ii] / norm,
            color=col,
            ls="--",
        )
        if parameters2 is not None:
            line2 = ax[0].plot(
                kk,
                p3d_best2[mask, ii] / norm,
                color=col,
                ls="-",
            )

        ## central panel ##
        ax[1].plot(
            kk,
            like.data["p3d"][mask, ii] / p3d_best1[mask, ii],
            color=col,
            ls=":",
        )

        if parameters2 is not None:
            ax[1].plot(
                kk,
                p3d_best2[mask, ii] / p3d_best1[mask, ii],
                color=col,
                ls="-",
            )

    ## lower panel ##
    if plot_p1d:
        mask = like.ind_fit1d.copy()
        ax[2].plot(
            like.data["k1d"][mask],
            like.data["p1d"][mask] / p1d_best1[mask],
            color=col,
            ls=":",
        )
        if parameters2 is not None:
            ax[2].plot(
                like.data["k1d"][mask],
                p1d_best2[mask] / p1d_best1[mask],
                color=col,
                ls="-",
            )

    ###
    ## plot cosmic variance errors
    # we plot these for mu=0
    ii = 0
    if err_bar_all:
        mask = like.ind_fit3d[:, ii]
        data = like.data["p3d"][mask, ii]
        # err_sta = like.data["std_p3d_sta"][mask, ii] / p3d_best1[mask, ii]
        err_sta = like.data["std_p3d_sta"][mask, ii] / data
        ax[1].fill_between(
            like.data["k3d"][mask, ii],
            -err_sta + 1,
            y2=err_sta + 1,
            color="k",
            alpha=0.25,
        )
        # err_sys = like.data["std_p3d_sys"][mask, ii] / p3d_best1[mask, ii]
        err_sys = like.data["std_p3d_sys"][mask, ii] / data
        ax[1].fill_between(
            like.data["k3d"][mask, ii],
            -err_sys + 1,
            y2=err_sys + 1,
            color="k",
            alpha=0.1,
            hatch="/",
        )

        mask = like.ind_fit1d.copy()
        data = like.data["p1d"][mask]
        err_sta = like.data["std_p1d_sta"][mask] / data
        ax[2].fill_between(
            like.data["k1d"][mask],
            -err_sta + 1,
            y2=err_sta + 1,
            color="k",
            alpha=0.25,
        )
        err_sys = like.data["std_p1d_sys"][mask] / data
        ax[2].fill_between(
            like.data["k1d"][mask],
            -err_sys + 1,
            y2=err_sys + 1,
            color="k",
            alpha=0.1,
            hatch="/",
        )
    ####

    ## set legend
    iax = 0
    ftsize = 15
    patch = []
    for ii, imu in enumerate(mu_use):
        mutag = (
            str(np.round(mubins[imu], 2))
            + r"$\leq\mu\leq$"
            + str(np.round(mubins[imu + 1], 2))
        )
        patch.append(mpatches.Patch(color="C" + str(ii), label=mutag))
    legend1 = ax[iax].legend(handles=patch, loc="upper left", fontsize=ftsize)
    if plot_legend_1:
        ax[iax].add_artist(legend1)

    lines = []
    ls = [":", "--", "-"]
    mm = ["o", "", ""]
    lab = [r"$X=$ Data", r"$X=$ Fit", r"$X=$ Emulator"]
    if plot_data:
        istart = 0
    else:
        istart = 1

    for ii in range(istart, 3):
        lines.append(
            mlines.Line2D(
                [],
                [],
                color="k",
                linestyle=ls[ii],
                marker=mm[ii],
                lw=2,
                label=lab[ii],
            )
        )
    legend2 = ax[iax].legend(handles=lines, loc="lower right", fontsize=ftsize)
    ax[iax].add_artist(legend2)

    ax[iax].set_xscale("log")
    # ylab = r"$(2\pi^2)^{-1} k^3 P_F^X(k)$"
    ylab = r"$P_\mathrm{F}^X/P_\mathrm{F}^\mathrm{lin}$"
    plot_template(
        ax[iax],
        legend_loc="upper left",
        ylabel=ylab,
        ftsize=19,
        ftsize_legend=13,
        legend=1,
        legend_columns=1,
    )

    ## central panel
    iax = 1
    # plot expected precision lines
    ax[iax].axhline(1, color="k", ls=":")
    ax[iax].axhline(1.1, color="k", ls="--")
    ax[iax].axhline(0.9, color="k", ls="--")
    ax[iax].set_ylim([0.8, 1.2])
    # ax[iax].axvline(x=kmax_1d, color="k")
    ax[iax].set_xscale("log")

    plot_template(
        ax[iax],
        #     legend_loc="upper right",
        # xlabel=r"$k\,\left[\mathrm{Mpc}^{-1}\right]$",
        ylabel=r"$P_\mathrm{F}^\mathrm{X}/P_\mathrm{F}^\mathrm{fit}$",
        ftsize=19,
        #     ftsize_legend=17,
        #     legend=0,
        #     legend_columns=1,
    )

    ## lower panel
    iax = 2
    # plot expected precision lines
    ax[iax].axhline(1, color="k", ls=":")
    ax[iax].axhline(1.01, color="k", ls="--")
    ax[iax].axhline(0.99, color="k", ls="--")
    ax[iax].set_ylim([0.95, 1.05])
    ax[iax].set_xscale("log")

    plot_template(
        ax[iax],
        #     legend_loc="upper right",
        xlabel=r"$k\,\left[\mathrm{Mpc}^{-1}\right]$",
        ylabel=r"$P_\mathrm{1D}^\mathrm{X}/P_\mathrm{1D}^\mathrm{fit}$",
        ftsize=19,
        #     ftsize_legend=17,
        #     legend=0,
        #     legend_columns=1,
    )

    #     ax[iax].set_xlim(self.data["k3d"][0, 0] * 0.9, 25)
    plt.tight_layout()

    if save_fig is not None:
        plt.savefig(save_fig)

        
def plot_err_uncertainty(emulator, archive, sim_labels, mu_lims_p3d, z, val_scaling=1.0, colors=['deepskyblue', 'goldenrod']):
    """
    Plot the percent error and uncertainty in P1D and P3D for different simulation labels.

    Parameters:
    - sim_labels (list): List of simulation labels for which the predictions are plotted.
    - mu_lims_p3d (tuple): Tuple defining the range of mu values to consider in the P3D plot.
    - z (float): Redshift value.
    - val_scaling (float, optional): Scaling factor for validation data. Defaults to 1.0.
    - colors (list, optional): List of colors for each simulation label in the plot. Defaults to ['deepskyblue', 'goldenrod'].

    Returns:
    None

    Plots:
    - Two horizontally aligned panels (P1D and P3D) with percent error and uncertainty.

    """

    # Extract data from Archive3D
    k_Mpc = archive.training_data[0]["k3d_Mpc"]
    mu = archive.training_data[0]["mu3d"]

    # Apply a mask to select relevant k values
    k_mask = (k_Mpc < 4) & (k_Mpc > 0)
    k_Mpc = k_Mpc[k_mask]
    mu = mu[k_mask]

    # Define plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=False, sharey=True)

    for ii, sim_label in enumerate(sim_labels):
        
        # Retrieve simulation
        test_sim = archive.get_testing_data(sim_label, force_recompute_plin=False)
        dict_sim = [d for d in test_sim if d['z'] == z and d['val_scaling'] == val_scaling]

        # Predict p1d and p3d
        p1d_arinyo, p1d_cov = emulator.predict_P1D_Mpc(
            sim_label=sim_label,
            z=z,
            test_sim=dict_sim,
            return_cov=True
        )

        p3d_arinyo, p3d_cov = emulator.predict_P3D_Mpc(
            sim_label=sim_label,
            z=z,
            test_sim=dict_sim,
            return_cov=True
        )

        # True p1d and p3d from sim
        p1d_sim, p1d_k = emulator.get_p1d_sim(dict_sim)
        p3d_sim = [d for d in test_sim if d['z'] == z][0]['p3d_Mpc'][emulator.k_mask]

        # Fractional error in p1d
        p1derr_pred = np.sqrt(np.diag(p1d_cov))
        frac_err_p1d = (p1d_arinyo / p1d_sim - 1) * 100
        frac_err_p1d_err = 100 * p1derr_pred

        # Fractional error in p3d
        p3derr_pred = np.sqrt(np.diag(p3d_cov))
        frac_err_p3d = (p3d_arinyo / p3d_sim - 1) * 100
        frac_err_p3d_err = 100 * p3derr_pred
        mu_mask = (mu >= mu_lims_p3d[0]) & (mu <= mu_lims_p3d[1])
        k_masked = k_Mpc[mu_mask]

        frac_err_p3d_masked = frac_err_p3d[mu_mask]
        frac_err_p3d_err_masked = frac_err_p3d_err[mu_mask]

        # Plot
        # Panel 1: P1D Plot
        axes[0].plot(p1d_k, frac_err_p1d, color=colors[ii])
        axes[0].fill_between(p1d_k,
                             frac_err_p1d - frac_err_p1d_err,
                             frac_err_p1d + frac_err_p1d_err,
                             alpha=0.2,
                             color=colors[ii],
                             label=f'{sim_label}')

        # Panel 2: P3D Plot
        axes[1].plot(k_masked, frac_err_p3d_masked, color=colors[ii])
        axes[1].fill_between(k_masked,
                             frac_err_p3d_masked - frac_err_p3d_err_masked,
                             frac_err_p3d_masked + frac_err_p3d_err_masked,
                             alpha=0.2,
                             color=colors[ii],
                             label=f'{sim_label}')

    axes[0].set_title('P1D', fontsize=16)
    axes[0].set_ylabel('Percent error [%]', fontsize=16)
    axes[0].set_xlabel('$k$ [1/Mpc]', fontsize=16)
    axes[0].legend(fontsize=12)
    axes[0].fill_between(p1d_k, -1, 1, alpha=0.1, color='grey')
    axes[1].fill_between(k_masked, -10, 10, alpha=0.1, color='grey')

    axes[1].set_title('P3D', fontsize=16)
    axes[1].set_ylim(-15, 15)
    axes[1].set_xlabel('$k$ [1/Mpc]', fontsize=16)

    # Adjust layout and show the

def plot_p3d_L1O(archive, fractional_errors, savename=None):
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
    k_mask = (k_Mpc < 4) & (k_Mpc > 0)
    k_Mpc = k_Mpc[k_mask]
    mu = mu[k_mask]

    # Create subplots with shared y-axis and x-axis
    fig, axs = plt.subplots(11, 1, figsize=(10, 20), sharey=True, sharex=True)

    # Define mu bins
    mu_lims = [[0, 0.06], [0.31, 0.38], [0.62, 0.69], [0.94, 1]]

    # Define colors for different mu bins
    colors = ["navy", "crimson", "forestgreen", "goldenrod"]
    test_sim =  archive.get_testing_data(
        'mpg_central', force_recompute_plin=True
    )
    z_grid = [d["z"] for d in test_sim]

    # Loop through redshifts
    for ii, z in enumerate(z_grid):
        axs[ii].set_title(f"$z={z}$", fontsize=16)
        axs[ii].axhline(y=-10, ls="--", color="black")
        axs[ii].axhline(y=10, ls="--", color="black")

        # Loop through mu bins
        for mi in range(int(len(mu_lims))):
            mu_mask = (mu >= mu_lims[mi][0]) & (mu <= mu_lims[mi][1])
            k_masked = k_Mpc[mu_mask]

            # Calculate fractional error statistics
            frac_err = np.nanmedian(fractional_errors[:, ii, :], 0)
            frac_err_err = sigma68(fractional_errors[:, ii, :])

            frac_err_masked = frac_err[mu_mask]
            frac_err_err_masked = frac_err_err[mu_mask]

            color = colors[mi]

            # Add a line plot with shaded error region to the current subplot
            axs[ii].plot(
                k_masked,
                frac_err_masked,
                label=f"${mu_lims[mi][0]}\leq \mu \leq {mu_lims[mi][1]}$",
                color=color,
            )
            axs[ii].fill_between(
                k_masked,
                frac_err_masked - frac_err_err_masked,
                frac_err_masked + frac_err_err_masked,
                color=color,
                alpha=0.2,
            )
            axs[ii].tick_params(axis="both", which="major", labelsize=16)

    # Customize subplot appearance
    for xx, ax in enumerate(axs):
        if xx == len(axs) // 2:  # Centered y-label
            ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.set_ylim(-10, 10)

    axs[len(axs) - 1].set_xlabel(r"$k$ [1/Mpc]", fontsize=25)

    axs[0].legend()

    # Adjust spacing between subplots
    plt.tight_layout()
    fig.text(
        0,
        0.5,
        r"Error $P_{\rm 3D}$ [%]",
        va="center",
        rotation="vertical",
        fontsize=16,
    )

    # Save the plot
    if savename:
        plt.savefig(savename, bbox_inches="tight")

def plot_p3d_LzO(archive, fractional_errors, z_test, savename=None):
    # Extract data from Archive3D
    k_Mpc = archive.training_data[0]["k3d_Mpc"]
    mu = archive.training_data[0]["mu3d"]

    # Apply a mask to select relevant k values
    k_mask = (k_Mpc < 4) & (k_Mpc > 0)
    k_Mpc = k_Mpc[k_mask]
    mu = mu[k_mask]

    # Create subplots with shared y-axis and x-axis
    fig, axs = plt.subplots(
        len(z_test), 1, figsize=(6, 8), sharey=True, sharex=True
    )

    # Define mu bins
    mu_lims = [[0, 0.06], [0.31, 0.38], [0.62, 0.69], [0.94, 1]]

    # Define colors for different mu bins
    colors = ["navy", "crimson", "forestgreen", "goldenrod"]

    # Loop through redshifts
    for ii, z in enumerate(z_test):
        axs[ii].set_title(f"$z={z}$", fontsize=14)
        axs[ii].axhline(y=-10, ls="--", color="black")
        axs[ii].axhline(y=10, ls="--", color="black")

        # Loop through mu bins
        for mi in range(int(len(mu_lims))):
            mu_mask = (mu >= mu_lims[mi][0]) & (mu <= mu_lims[mi][1])
            k_masked = k_Mpc[mu_mask]

            # Calculate fractional error statistics
            frac_err = np.nanmedian(fractional_errors[:, ii, :], 0)
            frac_err_err = sigma68(fractional_errors[:, ii, :])

            frac_err_masked = frac_err[mu_mask]
            frac_err_err_masked = frac_err_err[mu_mask]

            color = colors[mi]

            # Add a line plot with shaded error region to the current subplot
            axs[ii].plot(
                k_masked,
                frac_err_masked,
                label=f"${mu_lims[mi][0]}\leq \mu \leq {mu_lims[mi][1]}$",
                color=color,
            )
            axs[ii].fill_between(
                k_masked,
                frac_err_masked - frac_err_err_masked,
                frac_err_masked + frac_err_err_masked,
                color=color,
                alpha=0.2,
            )
            axs[ii].tick_params(axis="both", which="major", labelsize=16)

    # Customize subplot appearance
    for xx, ax in enumerate(axs):
        if xx == len(axs) // 2:  # Centered y-label
            ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.set_ylim(-10, 10)

    axs[len(axs) - 1].set_xlabel(r"$k$ [1/Mpc]", fontsize=16)

    axs[0].legend(fontsize=12)

    # Adjust spacing between subplots
    plt.tight_layout()
    fig.text(
        0,
        0.5,
        r"Error $P_{\rm 3D}$ [%]",
        va="center",
        rotation="vertical",
        fontsize=16,
    )

    # Save the plot
    if savename:
        plt.savefig(savename, bbox_inches='tight')
        
def plot_p1d_L1O(archive, fractional_errors, savename=None):
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

    # Create subplots with shared y-axis
    fig, axs = plt.subplots(11, 1, figsize=(10, 20), sharey=True)
    
    test_sim =  archive.get_testing_data(
        'mpg_central', force_recompute_plin=True
    )
    z_grid = [d["z"] for d in test_sim]
    
    like = Likelihood(test_sim[0], archive.rel_err_p3d, archive.rel_err_p1d)
    k1d_mask = like.like.ind_fit1d.copy()
    k1d_sim = like.like.data["k1d"][k1d_mask]

    # Loop through redshifts
    for ii, z in enumerate(z_grid):
        axs[ii].set_title(f"$z={z}$", fontsize=16)
        axs[ii].axhline(y=-1, ls="--", color="black")
        axs[ii].axhline(y=1, ls="--", color="black")

        # Calculate fractional error statistics
        frac_err = np.nanmedian(fractional_errors[:, ii, :], 0)
        frac_err_err = sigma68(fractional_errors[:, ii, :])


        # Add a line plot with shaded error region to the current subplot
        axs[ii].plot(k1d_sim, frac_err, color="crimson")
        axs[ii].fill_between(
            k1d_sim,
            frac_err - frac_err_err,
            frac_err + frac_err_err,
            color="crimson",
            alpha=0.2,
        )

        axs[ii].tick_params(axis="both", which="major", labelsize=18)

    # Customize subplot appearance
    for xx, ax in enumerate(axs):
        if xx == len(axs) // 2:  # Centered y-label
            ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.set_ylim(-5, 5)

    axs[len(axs) - 1].set_xlabel(r"$k$ [1/Mpc]", fontsize=25)
    axs[0].legend()

    # Adjust spacing between subplots
    plt.tight_layout()
    fig.text(
        0,
        0.5,
        r"Error $P_{\rm 1D}$ [%]",
        va="center",
        rotation="vertical",
        fontsize=16,
    )

    # Save the plot
    if savename:
        plt.savefig(savename, bbox_inches="tight")


def plot_p1d_LzO(archive, fractional_errors, z_test, savename=None):
    # Create subplots with shared y-axis
    fig, axs = plt.subplots(len(z_test), 1, figsize=(6, 8), sharey=True)

    test_sim =  archive.get_testing_data(
        'mpg_central', force_recompute_plin=True
    )    
    like = Likelihood(test_sim[0], archive.rel_err_p3d, archive.rel_err_p1d)
    k1d_mask = like.like.ind_fit1d.copy()
    k1d_sim = like.like.data["k1d"][k1d_mask]
    
    # Loop through redshifts
    for ii, z in enumerate(z_test):
        axs[ii].set_title(f"$z={z}$", fontsize=16)
        axs[ii].axhline(y=-1, ls="--", color="black")
        axs[ii].axhline(y=1, ls="--", color="black")

        # Calculate fractional error statistics
        frac_err = np.nanmedian(fractional_errors[:, ii, :], 0)
        frac_err_err = sigma68(fractional_errors[:, ii, :])


        # Add a line plot with shaded error region to the current subplot
        axs[ii].plot(k1d_sim, frac_err, color="crimson")
        axs[ii].fill_between(
            k1d_sim,
            frac_err - frac_err_err,
            frac_err + frac_err_err,
            color="crimson",
            alpha=0.2,
        )

        axs[ii].tick_params(axis="both", which="major", labelsize=18)

    # Customize subplot appearance
    for xx, ax in enumerate(axs):
        if xx == len(axs) // 2:  # Centered y-label
            ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.set_ylim(-5, 5)

    axs[len(axs) - 1].set_xlabel(r"$k$ [1/Mpc]", fontsize=16)
    axs[0].legend(fontsize=12)

    # Adjust spacing between subplots
    plt.tight_layout()
    fig.text(
        0,
        0.5,
        r"Error $P_{\rm 1D}$ [%]",
        va="center",
        rotation="vertical",
        fontsize=16,
    )

    # Save the plot
    if savename:
        plt.savefig(savename, bbox_inches='tight')
        
        
def plot_paramspace(params_emulator, errors, colourbar_lab=r'$P_{\rm 3D}$ uncertainty', vmin=0, vmax=1 ):
    """
    Plot parameter space with uncertainties in a 2x2 grid.

    Parameters:
    - params_emulator (numpy array): Emulator parameters. (Nsim*Nz,6)
    - errors (numpy array): Uncertainties corresponding to each point. (Nsim*Nz,Nk)

    Returns:
    - None
    """
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(8, 6), sharey='row')

    # Scatter plot in the first panel
    sc1 = axs[0, 0].scatter(params_emulator[:, 0], params_emulator[:, 2], c=np.median(errors, 1), s=10, rasterized=True, vmin=vmin, vmax=vmax)
    axs[0, 0].set_ylabel(r'$\bar{F}$', fontsize=16)
    axs[0, 0].set_xlabel(r'$\Delta_p$', fontsize=16)
    axs[0, 0].tick_params(axis='both', which='major', labelsize=12)

    # Scatter plot in the second panel
    sc2 = axs[0, 1].scatter(params_emulator[:, 3], params_emulator[:, 2], c=np.median(errors, 1), s=10, rasterized=True, vmin=vmin, vmax=vmax)
    axs[0, 1].set_xlabel(r'$\sigma_T$', fontsize=16)
    axs[0, 1].tick_params(axis='both', which='major', labelsize=12)

    # Scatter plot in the third panel
    sc3 = axs[1, 0].scatter(params_emulator[:, 4], params_emulator[:, 2], c=np.median(errors, 1), s=10, rasterized=True, vmin=vmin, vmax=vmax)
    axs[1, 0].set_ylabel(r'$\bar{F}$', fontsize=16)
    axs[1, 0].set_xlabel(r'$\gamma$', fontsize=16)
    axs[1, 0].tick_params(axis='both', which='major', labelsize=12)

    # Scatter plot in the fourth panel
    sc4 = axs[1, 1].scatter(params_emulator[:, 5], params_emulator[:, 2], c=np.median(errors, 1), s=10, rasterized=True, vmin=vmin, vmax=vmax)
    axs[1, 1].set_xlabel(r'$k_F$', fontsize=16)
    axs[1, 1].tick_params(axis='both', which='major', labelsize=12)

    # Adjust layout
    plt.tight_layout()

    # Move the legend to the right
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.11, 0.03, 0.87])
    fig.colorbar(sc4, cax=cbar_ax, label=colourbar_lab)
    cbar_ax.yaxis.label.set_fontsize(16)

    # Show the plot
    plt.show()
