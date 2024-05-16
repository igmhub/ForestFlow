import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


def plot_arinyo_z(
    z_central,
    Arinyo_central,
    Arinyo_seed,
    Arinyo_emu,
    Arinyo_emu_std,
    folder_fig=None,
    ftsize=20,
):
    z_central = np.array(z_central)
    # Create a 2x1 grid for plotting
    fig, ax = plt.subplots(
        4, 1, figsize=(8, 12), sharex=True, height_ratios=[3, 1, 3, 1]
    )
    name_params = ["bias", "bias_eta", "q1", "kp", "q2", "kv", "av", "bv"]
    # name_params = list(Arinyo_emu[0].keys())

    name2label = {
        "bias": r"$-b_\delta$",
        "bias_eta": r"$-b_\eta$",
        "q1": r"$0.5(q_1+q_2)$",
        "q2": r"$0.5(q_1-q_2)$",
        "kv": r"$k_\mathrm{v}$",
        "av": r"$a_\mathrm{v}$",
        "bv": r"$b_\mathrm{v}$",
        "kp": r"$k_\mathrm{p}$",
    }

    # Plot the original and emulator data in the upper panel
    for i in range(len(name_params)):
        if i < 4:
            ax1 = ax[0]
            ax2 = ax[1]
        else:
            ax1 = ax[2]
            ax2 = ax[3]
        col = "C" + str(i)
        if name_params[i] == "q1":
            ari_emu1 = np.array([d["q1"] for d in Arinyo_emu])
            ari_emu2 = np.array([d["q2"] for d in Arinyo_emu])
            ari_emu = 0.5 * (ari_emu1 + ari_emu2)

            ari_emu_std1 = np.array([d["q1"] for d in Arinyo_emu_std])
            ari_emu_std2 = np.array([d["q2"] for d in Arinyo_emu_std])
            ari_emu_std = 0.5 * np.sqrt(ari_emu_std1**2 + ari_emu_std2**2)

            ari_cen1 = np.array([d["q1"] for d in Arinyo_central])
            ari_cen2 = np.array([d["q2"] for d in Arinyo_central])
            ari_cen = 0.5 * (ari_cen1 + ari_cen2)

            ari_cen1 = np.array([d["q1"] for d in Arinyo_seed])
            ari_cen2 = np.array([d["q2"] for d in Arinyo_seed])
            ari_seed = 0.5 * (ari_cen1 + ari_cen2)
        elif name_params[i] == "q2":
            ari_emu1 = np.array([d["q1"] for d in Arinyo_emu])
            ari_emu2 = np.array([d["q2"] for d in Arinyo_emu])
            ari_emu = 0.5 * (ari_emu1 - ari_emu2)

            ari_emu_std1 = np.array([d["q1"] for d in Arinyo_emu_std])
            ari_emu_std2 = np.array([d["q2"] for d in Arinyo_emu_std])
            ari_emu_std = 0.5 * np.sqrt(ari_emu_std1**2 + ari_emu_std2**2)

            ari_cen1 = np.array([d["q1"] for d in Arinyo_central])
            ari_cen2 = np.array([d["q2"] for d in Arinyo_central])
            ari_cen = 0.5 * (ari_cen1 - ari_cen2)

            ari_cen1 = np.array([d["q1"] for d in Arinyo_seed])
            ari_cen2 = np.array([d["q2"] for d in Arinyo_seed])
            ari_seed = 0.5 * (ari_cen1 - ari_cen2)
        else:
            ari_emu = np.array([d[name_params[i]] for d in Arinyo_emu])
            ari_emu_std = np.array([d[name_params[i]] for d in Arinyo_emu_std])
            ari_cen = np.array([d[name_params[i]] for d in Arinyo_central])
            ari_seed = np.array([d[name_params[i]] for d in Arinyo_seed])

        print(name_params[i])
        # print(np.mean(np.abs(ari_emu)/np.abs(ari_cen)-1))
        # print(np.std(np.abs(ari_emu)/np.abs(ari_cen)-1))
        # if i != 6:
        ax1.plot(z_central, np.abs(ari_cen), "--", color=col, lw=2)
        ax1.plot(z_central, np.abs(ari_seed), "-.", color=col, lw=2)

        res = np.polyfit(z_central[:-1], np.log10(np.abs(ari_emu))[:-1], deg=2)
        print(res)
        p = 10 ** (res[0] * z_central**2 + res[1] * z_central + res[2])
        # p = 10 ** (res[0] * z_central + res[1])
        ax1.plot(z_central, p, color="k")
        ax2.plot(z_central, p / np.abs(ari_emu) - 1, color="k")

        ax1.plot(
            z_central,
            np.abs(ari_emu),
            color=col,
            ls="-",
        )
        ax1.fill_between(
            z_central,
            np.abs(ari_emu) - 0.5 * ari_emu_std,
            np.abs(ari_emu) + 0.5 * ari_emu_std,
            color=col,
            alpha=0.2,
        )

        ax2.plot(
            z_central,
            np.abs(ari_cen) / np.abs(ari_emu) - 1,
            "--",
            color=col,
            lw=2,
            alpha=0.8,
        )
        ax2.fill_between(
            z_central,
            np.abs(ari_cen) / (np.abs(ari_emu) - 0.5 * ari_emu_std) - 1,
            np.abs(ari_cen) / (np.abs(ari_emu) + 0.5 * ari_emu_std) - 1,
            color=col,
            alpha=0.2,
        )

        ax2.plot(
            z_central,
            np.abs(ari_seed) / np.abs(ari_emu) - 1,
            "-.",
            color=col,
            lw=2,
            alpha=0.8,
        )
        ax2.fill_between(
            z_central,
            np.abs(ari_seed) / (np.abs(ari_emu) - 0.5 * ari_emu_std) - 1,
            np.abs(ari_seed) / (np.abs(ari_emu) + 0.5 * ari_emu_std) - 1,
            color=col,
            alpha=0.2,
        )

        # ax2.plot(z_central, np.abs(ari_cen)
        # / np.abs(ari_emu)
        # - 1, color=colors[i], ls="-")

    for ii in range(0, 4, 2):
        ax[ii].set_ylabel("Parameter value", fontsize=ftsize)
        ax[ii].set_yscale("log")
        ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)

    for ii in range(1, 5, 2):
        ax[ii].set_ylabel("Residual", fontsize=ftsize)
        ax[ii].tick_params(axis="both", which="major", labelsize=ftsize)
        ax[ii].axhline(y=0, color="k", ls=":", lw=2)

    ax[0].set_ylim(8e-2, 25)
    ax[1].set_ylim(-0.1, 0.1)
    ax[2].set_ylim(1e-3, 3)
    ax[3].set_ylim(-1.0, 1.0)

    ax[-1].set_xlabel("$z$", fontsize=ftsize)

    hand = []
    for i in range(4):
        col = "C" + str(i)
        hand.append(mpatches.Patch(color=col, label=name2label[name_params[i]]))
    legend1 = ax[0].legend(
        fontsize=ftsize - 2, loc="lower right", handles=hand, ncols=2
    )

    # line1 = Line2D(
    #     [0], [0], label="Best fit to data", color="k", ls="", marker="o"
    # )
    line1 = Line2D([0], [0], label="ForestFlow", color="k", ls="-", linewidth=2)
    line2 = Line2D([0], [0], label="Central", color="k", ls="--", linewidth=2)
    line3 = Line2D([0], [0], label="Seed", color="k", ls="-.", linewidth=2)
    hand = [line1, line2, line3]
    ax[0].legend(fontsize=ftsize - 2, loc="upper left", handles=hand, ncols=1)
    ax[0].add_artist(legend1)

    hand = []
    for i in range(4, len(name_params)):
        col = "C" + str(i)
        hand.append(mpatches.Patch(color=col, label=name2label[name_params[i]]))
    legend1 = ax[2].legend(
        fontsize=ftsize - 2, loc="lower right", handles=hand, ncols=4
    )

    # plt.gca().add_artist(legend1)
    # Adjust layout
    plt.tight_layout()
    if folder_fig is not None:
        plt.savefig(folder_fig + "arinyo_z.png")
        plt.savefig(folder_fig + "arinyo_z.pdf")
