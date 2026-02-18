import numpy as np
from scipy.stats import gaussian_kde


def plot_template(
    ax,
    ax2=None,
    ay2=None,
    xlabel=None,
    ylabel=None,
    title=None,
    legend=None,
    legend_loc="best",
    ftsize=17,
    extra_xaxis=False,
    extra_yaxis=False,
    xcolor="k",
    ycolor="k",
    xcolor2="k",
    ycolor2="k",
    ylabelpad=None,
    handlelength=2,
    legend_title=None,
    legend_columns=1,
    ftsize_legend=15,
    title_fontsize="x-large",
):
    """Template for all plots"""

    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))

    if legend == 0:
        ax.legend(
            fontsize=ftsize_legend,
            loc=legend_loc,
            handlelength=handlelength,
            title=legend_title,
            title_fontsize=title_fontsize,
            ncol=legend_columns,
        )

    if title:
        ax.set_title(
            title,
            fontsize=ftsize + 2,
        )

    if xlabel:
        ax.set_xlabel(
            xlabel,
            fontsize=ftsize + 2,
            color=xcolor,
        )
    if ylabel:
        ax.set_ylabel(
            ylabel,
            labelpad=ylabelpad,
            fontsize=ftsize + 2,
            color=ycolor,
        )

    """for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(ftsize)
        tick.label.set_color(xcolor)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(ftsize)
        tick.label.set_color(ycolor)

    if extra_xaxis:
        for tick in ax2.xaxis.get_major_ticks():
            tick.label2.set_fontsize(ftsize)
            tick.label2.set_color(xcolor2)
    if extra_yaxis:
        for tick in ay2.yaxis.get_major_ticks():
            tick.label2.set_fontsize(ftsize)
            tick.label2.set_color(ycolor2)"""


def plot_vec(cen, vv, length, ax, label, col, direction=None):
    """Plot vectors"""

    # vectors look up and right
    if direction == "up_right":
        if (vv[0] < 0) & (vv[1] < 0):
            vv = abs(vv)
        if (vv[0] > 0) & (vv[1] < 0):
            vv = -vv

    ax.quiver(
        cen[0],
        cen[1],
        vv[0],
        vv[1],
        color=col,
        width=0.02,
        scale=1 / length,
        scale_units="xy",
        angles="xy",
        alpha=0.5,
        label=label,
    )

    return


def density_estimation(m1, m2, ntt=100j):
    xmin = np.min(m1) * 0.95
    xmax = np.max(m1) * 1.05
    ymin = np.min(m2) * 0.95
    ymax = np.max(m2) * 1.05
    X, Y = np.mgrid[xmin:xmax:ntt, ymin:ymax:ntt]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    # ax.imshow(np.rot90(Z), cmap=cmap,
    # extent=[xmin, xmax, ymin, ymax])
    return X, Y, Z


def find_confidence_interval(x, pdf, confidence_level):
    return pdf[pdf > x].sum() - confidence_level
