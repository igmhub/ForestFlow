import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


def plot_arinyo_z(
    z_central,
    Arinyo_central,
    Arinyo_seed,
    Arinyo_both,
    folder_fig=None,
    ftsize=20,
):
    z_central = np.array(z_central)
    # Create a 2x1 grid for plotting
    fig, ax = plt.subplots(
        2, 1, figsize=(8, 10), sharex=True, height_ratios=[3, 1]
    )
    name_params = ["bias", "bias_eta", "kp", "q1", "av", "bv", "kv"]
    transp = [0.7, 0.6, 0.5, 0.7, 0.6, 0.5, 0.4]
    nparam = 3
    # name_params = list(Arinyo_emu[0].keys())

    name2label = {
        "bias": r"$-b_\delta$",
        "bias_eta": r"$-b_\eta$",
        "q1": r"$q$",
        "kv": r"$k_\mathrm{v}$",
        "av": r"$a_\mathrm{v}$",
        "bv": r"$b_\mathrm{v}$",
        "kp": r"$k_\mathrm{p}/10$",
    }

    # Plot the original and emulator data in the upper panel
    for i in range(len(name_params)):
        col = "C" + str(i)
        ari_both = np.array([d[name_params[i]] for d in Arinyo_both])
        ari_cen = np.array([d[name_params[i]] for d in Arinyo_central])
        ari_seed = np.array([d[name_params[i]] for d in Arinyo_seed])

        if name_params[i] == "kp":
            norm = 0.1
        else:
            norm = 1

        print(name_params[i])

        y = np.concatenate([ari_seed / ari_both, ari_cen / ari_both]) - 1
        s_bias = np.median(y)
        s_pred = np.percentile(y - 1, [16, 84])
        s_pred = 0.5 * (s_pred[1] - s_pred[0])
        print("sim-both", s_bias * 100, s_pred * 100)

        lw = 3

        ax[0].plot(
            z_central, norm * np.abs(ari_cen), ":", color=col, lw=lw, alpha=0.7
        )
        ax[0].plot(
            z_central,
            norm * np.abs(ari_seed),
            "--",
            color=col,
            lw=lw,
            alpha=0.7,
        )
        ax[0].plot(
            z_central,
            norm * np.abs(ari_both),
            color=col,
            ls="-",
            lw=lw,
            alpha=0.7,
        )

        if i < 2:
            ax[1].plot(
                z_central,
                np.abs(ari_cen) / np.abs(ari_both) - 1,
                ":",
                color=col,
                lw=lw,
                alpha=0.7,
            )

            ax[1].plot(
                z_central,
                np.abs(ari_seed) / np.abs(ari_both) - 1,
                "--",
                color=col,
                lw=lw,
                alpha=0.7,
            )

    for ii in range(0, 1):
        ax[ii].set_ylabel("Parameter value", fontsize=ftsize)
        ax[ii].set_yscale("log")
        ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)

    for ii in range(1, 2):
        ax[ii].set_ylabel("Residual", fontsize=ftsize)
        ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)
        ax[ii].axhline(y=0, color="k", ls=":", lw=2)
        # ax[ii].axhline(y=0.1, color="k", ls="--", lw=2)
        # ax[ii].axhline(y=-0.1, color="k", ls="--", lw=2)

    # ax[0].set_ylim(8e-2, 1.6)
    # ax[1].set_ylim(-0.21, 0.21)
    ax[0].set_ylim(5e-2, 3)
    ax[1].set_ylim(-0.11, 0.11)
    # ax[2].set_ylim(-0.21, 0.21)

    ax[-1].set_xlabel("$z$", fontsize=ftsize)

    hand = []
    for i in range(len(name_params)):
        col = "C" + str(i)
        hand.append(mpatches.Patch(color=col, label=name2label[name_params[i]]))
    legend1 = ax[0].legend(
        fontsize=ftsize - 2, loc="lower right", handles=hand, ncols=2
    )

    # line1 = Line2D(
    #     [0], [0], label="Best fit to data", color="k", ls="", marker="o"
    # )
    line1 = Line2D(
        [0], [0], label="Combination", color="k", ls="-", linewidth=2
    )
    line2 = Line2D([0], [0], label="Central", color="k", ls=":", linewidth=2)
    line3 = Line2D([0], [0], label="Seed", color="k", ls="--", linewidth=2)
    hand = [line1, line2, line3]
    ax[0].legend(fontsize=ftsize - 2, loc="upper left", handles=hand, ncols=3)
    ax[0].add_artist(legend1)

    # hand = []
    # for i in range(nparam, len(name_params)):
    #     col = "C" + str(i)
    #     hand.append(mpatches.Patch(color=col, label=name2label[name_params[i]]))
    # legend1 = ax[2].legend(
    #     fontsize=ftsize - 2, loc="lower right", handles=hand, ncols=4
    # )

    # plt.gca().add_artist(legend1)
    # Adjust layout
    plt.tight_layout()
    if folder_fig is not None:
        plt.savefig(folder_fig + "arinyo_z.png")
        plt.savefig(folder_fig + "arinyo_z.pdf")


def plot_forestflow_z(
    z_central,
    Arinyo_central,
    Arinyo_emu,
    Arinyo_emu_std,
    folder_fig=None,
    ftsize=20,
):
    z_central = np.array(z_central)
    # Create a 2x1 grid for plotting
    fig, ax = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True, height_ratios=[3, 1]
    )
    name_params = ["bias", "bias_eta", "bv", "q1", "q2", "kp", "av", "kv"]
    transp = [0.7, 0.6, 0.5, 0.7, 0.6, 0.5, 0.4]
    nparam = 2
    # name_params = list(Arinyo_emu[0].keys())

    name2label = {
        "bias": r"$-b_\delta$",
        "bias_eta": r"$-b_\eta$",
        "q1": r"$q_1$",
        "q2": r"$q_2$",
        "kv": r"$k_\mathrm{v}$",
        "av": r"$a_\mathrm{v}$",
        "bv": r"$b_\mathrm{v}$",
        "kp": r"$0.1\,k_\mathrm{p}$",
    }

    ax1 = ax[0]
    ax2 = ax[1]
    # Plot the original and emulator data in the upper panel
    for i in range(len(name_params)):
        col = "C" + str(i)
        ari_emu = np.array([d[name_params[i]] for d in Arinyo_emu])
        ari_emu_std = np.array([d[name_params[i]] for d in Arinyo_emu_std])
        ari_cen = np.array([d[name_params[i]] for d in Arinyo_central])

        if name_params[i] == "kp":
            norm = 0.1
        else:
            norm = 1

        print(name_params[i])

        s_bias = np.median(ari_emu / ari_cen - 1)
        s_pred = np.percentile(ari_emu / ari_cen - 1, [16, 84])
        s_pred = 0.5 * (s_pred[1] - s_pred[0])
        print("cen-emu", s_bias * 100, s_pred * 100)

        ax1.plot(
            z_central,
            norm * np.abs(ari_cen),
            "o:",
            color=col,
            lw=1.5,
            alpha=0.8,
        )

        ax1.plot(
            z_central,
            norm * np.abs(ari_emu),
            color=col,
            ls="-",
            lw=3,
            alpha=0.8,
        )
        ax1.fill_between(
            z_central,
            norm * (np.abs(ari_emu) - 0.5 * ari_emu_std),
            norm * (np.abs(ari_emu) + 0.5 * ari_emu_std),
            color=col,
            alpha=0.2,
        )

        if (name_params[i] == "bias_eta") | (name_params[i] == "bias"):
            ax2.plot(
                z_central,
                np.abs(ari_emu) / np.abs(ari_cen) - 1,
                color=col,
                lw=3,
                alpha=0.8,
            )
            ax2.fill_between(
                z_central,
                (np.abs(ari_emu) - 0.5 * ari_emu_std) / np.abs(ari_cen) - 1,
                (np.abs(ari_emu) + 0.5 * ari_emu_std) / np.abs(ari_cen) - 1,
                color=col,
                alpha=0.2,
            )

    for ii in range(0, 1):
        ax[ii].set_ylabel("Parameter value", fontsize=ftsize)
        ax[ii].set_yscale("log")
        ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)

    for ii in range(1, 2):
        ax[ii].set_ylabel("Residual", fontsize=ftsize)
        ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)
        ax[ii].axhline(y=0, color="k", ls=":", lw=2)

    ax[0].set_ylim(6e-2, 3)
    ax[1].set_ylim(-0.11, 0.11)
    ax[-1].set_xlabel("$z$", fontsize=ftsize)

    hand = []
    for i in range(len(name_params)):
        col = "C" + str(i)
        hand.append(mpatches.Patch(color=col, label=name2label[name_params[i]]))
    legend1 = ax[0].legend(
        fontsize=ftsize - 2, loc="lower right", handles=hand, ncols=2
    )

    # line1 = Line2D(
    #     [0], [0], label="Best fit to data", color="k", ls="", marker="o"
    # )
    line1 = Line2D([0], [0], label="ForestFlow", color="k", ls="-", linewidth=2)
    line2 = Line2D(
        [0], [0], label="Central", color="k", ls=":", marker="o", linewidth=2
    )
    # line3 = Line2D([0], [0], label="Seed", color="k", ls="--", linewidth=2)
    hand = [line1, line2]
    ax[0].legend(fontsize=ftsize - 2, loc="upper left", handles=hand, ncols=1)
    ax[0].add_artist(legend1)

    plt.tight_layout()
    if folder_fig is not None:
        plt.savefig(folder_fig + "arinyo_z.png")
        plt.savefig(folder_fig + "arinyo_z.pdf")
