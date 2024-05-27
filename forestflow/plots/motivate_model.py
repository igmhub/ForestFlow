import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


def plot_motivate_model(
    knew,
    munew,
    mu_bins,
    rebin_p3d,
    rebin_model_p3d,
    rebin_kaiser_p3d,
    rebin_plin,
    folder=None,
    ftsize=20,
    kmax_fit=3,
):
    fig, ax = plt.subplots(2, figsize=(8, 8), sharex=True, height_ratios=[3, 1])

    labs = []
    for ii in range(rebin_p3d.shape[1]):
        col = "C" + str(ii)

        if ii != rebin_p3d.shape[1] - 1:
            lab = str(mu_bins[ii]) + r"$\leq\mu<$" + str(mu_bins[ii + 1])
        else:
            lab = str(mu_bins[ii]) + r"$\leq\mu\leq$" + str(mu_bins[ii + 1])
        labs.append(lab)

        _ = np.isfinite(knew[:, ii])
        x = knew[_, ii]
        y = rebin_p3d[_, ii] / rebin_plin[_, ii]
        kaiser = rebin_kaiser_p3d[_, ii] / rebin_plin[_, ii]
        ax[0].plot(x, y, col + "o:")

        ax[0].plot(x, kaiser, color=col, ls="--", lw=2, alpha=0.8)

        y = rebin_model_p3d[_, ii] / rebin_plin[_, ii]
        ax[0].plot(x, y, col + "-", lw=2, alpha=0.8)

        y = rebin_p3d[_, ii] / rebin_model_p3d[_, ii] - 1
        ax[1].plot(x, y, col + "-", lw=2, alpha=0.8)

        y = rebin_kaiser_p3d[_, ii] / rebin_model_p3d[_, ii] - 1
        ax[1].plot(x, y, col + "--", lw=1, alpha=0.8)

    for ii in range(2):
        ax[ii].axvline(kmax_fit, color="k", ls="--", lw=1.5, alpha=0.8)

    for ii in range(1, 2):
        ax[ii].axhline(0, color="k", ls=":", lw=1.5, alpha=0.8)
        ax[ii].axhline(0.1, color="k", ls="--", lw=1.5, alpha=0.8)
        ax[ii].axhline(-0.1, color="k", ls="--", lw=1.5, alpha=0.8)
        ax[ii].set_ylim(-0.21, 0.21)

    ax[0].set_ylim(bottom=-0.01)
    ax[0].set_xscale("log")
    ax[-1].set_xlabel(r"$k\, [\mathrm{Mpc}^{-1}]$", fontsize=ftsize)
    ax[0].set_ylabel(r"$P_\mathrm{3D}(k, \mu)/P_{\rm L}(k)$", fontsize=ftsize)
    ax[1].set_ylabel(r"Residual", fontsize=ftsize)
    # ax[2].set_ylabel(r"Residual Kaiser", fontsize=ftsize)
    # ax[0].legend(loc='upper right', ncol=1, fontsize=ftsize-2)
    hand = []
    for i in range(4):
        col = "C" + str(i)
        hand.append(mpatches.Patch(color=col, label=labs[i]))
    legend1 = ax[0].legend(
        fontsize=ftsize - 2, loc="upper right", handles=hand, ncols=2
    )

    line1 = Line2D(
        [0], [0], label="Central", color="k", ls=":", marker="o", linewidth=2
    )
    line2 = Line2D([0], [0], label="Fit", color="k", ls="-", linewidth=2)
    line3 = Line2D(
        [0],
        [0],
        label=r"Kaiser",
        color="k",
        ls="--",
        linewidth=2,
    )
    hand = [line1, line2, line3]
    ax[0].legend(fontsize=ftsize - 2, loc="lower left", handles=hand, ncols=3)
    ax[0].add_artist(legend1)

    for ii in range(2):
        ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)
    plt.tight_layout()

    if folder is not None:
        plt.savefig(folder + "motivate.pdf")
        plt.savefig(folder + "motivate.png")
