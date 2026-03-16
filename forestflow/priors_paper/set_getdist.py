import numpy as np
from getdist import MCSamples


def set_getdist_samples(BAO, P1D):

    names = [
        "b_delta_sigma8",
        "b_eta_f_sigma8",
        "bias_delta",
        "bias_eta",
        "beta",
        "bias_hcd",
    ]
    labels = [
        r"b_\delta \sigma_8",
        r"b_{\eta} f \sigma_8",
        r"b_\delta",
        r"b_\eta",
        r"\beta",
        r"b_\mathrm{HCD}",
    ]
    label_sample = {
        "dr1": "BAO DR1 low SNR",
        "dr2": "BAO DR2 low SNR",
        "dr1_hsnr": "BAO DR1",
        "dr2_hsnr": "BAO DR2",
    }
    all_samples = {}
    for key in BAO.keys():
        vstack = np.vstack(
            [
                BAO[key]["bias_delta_sig_8_z"],
                BAO[key]["bias_eta_f_sig_8_z"],
                BAO[key]["bias_delta"],
                BAO[key]["bias_eta"],
                BAO[key]["beta"],
                BAO[key]["bias_hcd"],
            ]
        ).T

        if "weights" in BAO[key]:
            all_samples[key] = MCSamples(
                samples=vstack.copy(),
                names=names,
                labels=labels,
                weights=BAO[key]["weights"].copy(),
                label=label_sample[key],
            )
        else:
            all_samples[key] = MCSamples(
                samples=vstack.copy(),
                names=names,
                labels=labels,
                label=label_sample[key],
            )

    p1d = np.vstack(
        [
            P1D["bias_delta_sig_8_z"],
            P1D["bias_eta_f_sig_8_z"],
            P1D["sig_8"],
            P1D["sig_8_z0"],
            P1D["fsig_8"],
            P1D["bias_delta"],
            P1D["beta"],
            P1D["q1"],
            P1D["kvav"],
            P1D["av"],
            P1D["bv"],
            P1D["kp"],
            P1D["q2"],
            P1D["bias_eta"],
            P1D["Delta2star"],
            P1D["nstar"],
        ]
    ).T
    names = [
        "b_delta_sigma8",
        "b_eta_f_sigma8",
        "sigma8",
        "sigma8_z0",
        "fsigma8",
        "bias_delta",
        "beta",
        "q1",
        "kvav",
        "av",
        "bv",
        "kp",
        "q2",
        "bias_eta",
        "Delta2star",
        "nstar",
    ]
    labels = [
        r"b_\delta \sigma_8",
        r"b_\eta f \sigma_8",
        r"\sigma_8(z=2.33)",
        r"\sigma_8(z=0)",
        r"f \sigma_8",
        r"b_\delta",
        r"\beta",
        r"q_1",
        r"k_\mathrm{vav}",
        r"a_\mathrm{v}",
        r"b_\mathrm{v}",
        r"k_\mathrm{p}",
        r"q_2",
        r"b_\eta",
        r"\Delta^2_\star",
        r"n_\star",
    ]

    label_sample = {"p1d": "P1D"}

    for key in ["p1d"]:
        all_samples[key] = MCSamples(
            samples=p1d.copy(),
            names=names,
            labels=labels,
            label=label_sample[key],
        )

    return all_samples
