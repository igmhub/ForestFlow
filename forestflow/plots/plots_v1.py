"""
Plots v1 utilities.
"""

from typing import Any
from numpy.typing import ArrayLike

import matplotlib.pyplot as plt
import numpy as np

from forestflow.likelihood import Likelihood
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from forestflow.plot_routines import plot_template


def plot_test_parz(Archive3D: Any, p3d_emu: ArrayLike, sim_label: Any) -> None:
    """
    Compare fitted and emulator Arinyo parameters across a test simulation.

    Parameters
    ----------
    Archive3D : object
        Archive exposing ``get_testing_data`` and ``emu_params``.
    p3d_emu : object
        Emulator exposing ``predict_Arinyos``.
    sim_label : str
        Test-simulation label passed to the archive.
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
        predict_params[jj] = p3d_emu.predict_Arinyos(_cosmo_params)
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


def plot_test_p3d(ind_book: Any, Archive3D: Any, p3d_emu: ArrayLike, sim_label: Any) -> None:
    """
    Plot P3D comparison for one snapshot of a test simulation.

    Parameters
    ----------
    ind_book : int
        Index of the requested snapshot in archive testing data.
    Archive3D : object
        Archive exposing testing data and relative-error arrays.
    p3d_emu : object
        Emulator exposing ``predict_Arinyos``.
    sim_label : str
        Test-simulation label passed to the archive.
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
        predict_params[jj] = p3d_emu.predict_Arinyos(_cosmo_params)
        input_params[jj] = list(testing_data[jj]["Arinyo"].values())

    # make sure bias negative (dependence on bias square)
    input_params[:, 0] = -np.abs(input_params[:, 0])

    like = Likelihood(
        testing_data[ind_book], Archive3D.rel_err_p3d, Archive3D.rel_err_p1d
    )

    # fit_pars = testing_data[ind_book]["Arinyo"]
    fit_pars = params_numpy2dict(input_params[ind_book])
    emu_pars = params_numpy2dict(predict_params[ind_book])

    # save_fig = "test.png"
    save_fig = None
    plot_compare_p3d_smooth(
        like.like,
        fit_pars,
        emu_pars,
        sim_label=sim_label,
        err_bar_all=True,
        save_fig=save_fig,
    )


def params_numpy2dict(params: ArrayLike) -> dict[str, Any]:
    """
    Map the legacy eight-component emulator vector to parameter names.

    Parameters
    ----------
    params : array_like of shape (8,)
        Legacy emulator output ordered as bias, beta, and six ``d1_*`` terms.

    Returns
    -------
    dict
        Mapping from legacy parameter names to vector entries.
    """
    param_names = [
        "bias",
        "beta",
        "d1_q1",
        "d1_kvav",
        "d1_av",
        "d1_bv",
        "d1_kp",
        "d1_q2",
    ]
    dict_param = {}
    for ii in range(params.shape[0]):
        dict_param[param_names[ii]] = params[ii]
    return dict_param


def plot_compare_p3d_smooth(
    self,
    parameters1: Any,
    parameters2: Any | None=None,
    error_fit_3d: ArrayLike | None=None,
    error_fit_1d: ArrayLike | None=None,
    save_fig: str | None=None,
    err_bar_all: bool | None=False,
    sim_label: Any="",
    plot_data: bool=False,
) -> Any | None:
    """
    Compare one or two legacy P3D model parameter mappings.

    Parameters
    ----------
    self : object
        Legacy likelihood-like object exposing data, fit masks, and
        ``get_model_3d``.
    parameters1, parameters2 : mapping
        Primary and optional comparison parameter mappings.
    error_fit_3d, error_fit_1d : array_like, optional
        Relative error arrays used for plotted error bars.
    save_fig : path-like, optional
        Filename used to save the generated figure.
    err_bar_all : bool, default=False
        Draw error bars on every plotted point rather than selected points.
    sim_label : str, default=""
        Prefix displayed in the figure title.
    plot_data : bool, default=False
        Plot measured P3D values and ratios when true.

    Notes
    -----
    This helper creates a figure but deliberately returns ``None``. Supplying
    ``save_fig`` writes it with :func:`matplotlib.pyplot.savefig`.
    """

    fig, ax = plt.subplots(
        2,
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
        figsize=(8, 6),
    )

    tit = sim_label + r" $z=$" + str(self.data["z"][0])
    fig.suptitle(tit, fontsize=17)

    # compute best-fitting model
    p3d_best1 = self.get_model_3d(parameters=parameters1)
    #     p1d_best1 = self.get_model_1d(parameters=parameters1)
    if parameters2 is not None:
        p3d_best2 = self.get_model_3d(parameters=parameters2)
    #         p1d_best2 = self.get_model_1d(parameters=parameters2)

    # iterate over wedges
    nmus = self.data["k3d"].shape[1]
    mubins = np.linspace(0, 1, nmus + 1)
    mu_use = np.linspace(0, nmus - 1, 4, dtype=int)

    for imu, ii in enumerate(mu_use):
        col = "C" + str(imu)

        # only plot when data is not nan
        mask = self.ind_fit3d[:, ii]

        if plot_data:
            data = self.data["p3d"][mask, ii]

            line1 = ax[0].plot(
                self.data["k3d"][mask, ii], data, color=col, ls=":", marker="o"
            )
            # ratio
            ax[1].plot(
                self.data["k3d"][mask, ii],
                p3d_best1[mask, ii] / data,
                color=col,
                ls="-",
            )
            ax[1].plot(
                self.data["k3d"][mask, ii],
                p3d_best2[mask, ii] / data,
                color=col,
                ls="--",
            )
        else:
            # ratio
            ax[1].plot(
                self.data["k3d"][mask, ii],
                p3d_best2[mask, ii] / p3d_best1[mask, ii],
                color=col,
                ls="-",
            )

        line1 = ax[0].plot(
            self.data["k3d"][mask, ii],
            p3d_best1[mask, ii],
            color=col,
            ls="-",
        )
        line2 = ax[0].plot(
            self.data["k3d"][mask, ii],
            p3d_best2[mask, ii],
            color=col,
            ls="--",
        )

    ###
    # plot cosmic variance errors
    ii = 0
    iax = 1
    if err_bar_all:
        err_sta = self.data["std_p3d_sta"][mask, ii] / p3d_best1[mask, ii]
        ax[iax].fill_between(
            self.data["k3d"][mask, ii],
            -err_sta + 1,
            y2=err_sta + 1,
            color="k",
            alpha=0.25,
        )
        err_sys = self.data["std_p3d_sys"][mask, ii] / p3d_best1[mask, ii]
        ax[iax].fill_between(
            self.data["k3d"][mask, ii],
            -err_sys + 1,
            y2=err_sys + 1,
            color="k",
            alpha=0.1,
            hatch="/",
        )

    ####

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
    ax[iax].add_artist(legend1)

    lines = []
    ls = [":", "-", "--"]
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
    ylab = r"$(2\pi^2)^{-1} k^3 P_F^X(k)$"
    plot_template(
        ax[iax],
        legend_loc="upper left",
        ylabel=ylab,
        ftsize=19,
        ftsize_legend=13,
        legend=1,
        legend_columns=1,
    )

    iax = 1
    # plot expected precision lines
    ax[iax].axhline(1, color="k", ls=":")
    ax[iax].axhline(1.05, color="k", ls="--")
    ax[iax].axhline(0.95, color="k", ls="--")
    ax[iax].set_ylim([0.85, 1.15])
    # ax[iax].axvline(x=kmax_1d, color="k")
    ax[iax].set_xscale("log")

    plot_template(
        ax[iax],
        #     legend_loc="upper right",
        xlabel=r"$k\,\left[\mathrm{Mpc}^{-1}\right]$",
        ylabel=r"$P_\mathrm{3D}^\mathrm{Emu}/P_\mathrm{3D}^\mathrm{Fit}$",
        ftsize=19,
        #     ftsize_legend=17,
        #     legend=0,
        #     legend_columns=1,
    )

    #     ax[iax].set_xlim(self.data["k3d"][0, 0] * 0.9, 25)
    plt.tight_layout()

    if save_fig is not None:
        plt.savefig(save_fig)
