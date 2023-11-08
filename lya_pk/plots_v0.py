import matplotlib.pyplot as plt
import numpy as np

from lya_pk.likelihood import Likelihood
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from lya_pk.plot_routines import plot_template
from lya_pk.utils import params_numpy2dict


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
        _cosmo_params = np.zeros(len(Archive3D.emu_params))
        for ii, par in enumerate(Archive3D.emu_params):
            _cosmo_params[ii] = testing_data[jj][par]
        predict_params[jj] = p3d_emu.get_coeff(_cosmo_params)
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
        _cosmo_params = np.zeros(len(Archive3D.emu_params))
        for ii, par in enumerate(Archive3D.emu_params):
            _cosmo_params[ii] = testing_data[jj][par]
        if plot_emu:
            predict_params[jj] = p3d_emu.get_coeff(_cosmo_params)
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
