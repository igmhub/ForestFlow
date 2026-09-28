# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: lace
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Response of the ForestFlow P1D emulator to its input parameters
#
# This notebook varies one ForestFlow input parameter at a time about the
# MP-Gadget central simulation at $z=3$. It predicts Arinyo parameters with
# ForestFlow and projects the associated 3D model to $P_\mathrm{1D}$.
# No simulation archive is loaded.

# %%
# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt
import numpy as np

from forestflow.P3D_cINN import P3DEmulator
from forestflow.model_p3d_arinyo import ArinyoModel
from lace.cosmo.cosmology import Cosmology

# %% [markdown]
# ## Load ForestFlow and define the central point
#
# We start from the Planck18 cosmology and use LaCE to compute its compressed
# linear-power parameters at $z=3$ and $k_p=0.7\,\mathrm{Mpc}^{-1}$. The four
# IGM inputs are fixed to `mpg_central` at the same redshift. These IGM values
# are stored here so the notebook does not need to load a simulation archive.

# %%
emulator_label = "forest_mpg"
z = 3.0
kp_Mpc = 0.7
fiducial_cosmology = Cosmology(cosmo_label="Planck18")
fiducial_linear_parameters = fiducial_cosmology.get_linP_Mpc_params(
    z=z, kp_Mpc=kp_Mpc
)
fiducial_parameters = {
    "Delta2_p": fiducial_linear_parameters["Delta2_p"],
    "n_p": fiducial_linear_parameters["n_p"],
    "mF": 0.6604100706377194,
    "sigT_Mpc": 0.12817463664956008,
    "gamma": 1.512170923999183,
    "kF_Mpc": 10.6348381789184,
}

# A fixed seed makes all one-at-a-time comparisons reproducible.
n_realizations = 500
emulator = P3DEmulator(key=emulator_label, Nrealizations=n_realizations)

# %% [markdown]
# ## Define the wavenumbers and parameter variations
#
# We evaluate 100 logarithmically spaced comoving wavenumbers from
# $0.05$ to $5\,\mathrm{Mpc}^{-1}$. For the cosmological responses we vary
# $A_s$ or $n_s$ and recompute both compressed parameters with LaCE. For the
# IGM responses, every other input and the Planck18 cosmology remain fixed.

# %%
len_max = 100
k_Mpc = np.logspace(np.log10(0.05), np.log10(5.0), len_max)

cosmology_steps = {
    "As_fraction": 0.05,
    "ns": 0.05,
}
igm_parameter_steps = {
    "mF": 0.05,
    "gamma": 0.10,
    "sigT_Mpc": 0.02,
    "kF_Mpc": 2.0,
}


def predict_power(
    parameters: dict[str, float], target_cosmology: Cosmology
) -> dict[str, np.ndarray]:
    """Predict P1D and P3D at two orientations from one emulator call."""
    arinyo_parameters = emulator.evaluate(
        emu_params=parameters,
        Nrealizations=n_realizations,
        seed=0,
    )
    model = ArinyoModel(target_cosmology)
    linear_theory = model.linear_theory(z)
    return {
        "p1d": np.asarray(
            model.P1D_Mpc(linear_theory, z, k_Mpc, arinyo_parameters)
        ).squeeze(),
        "p3d_mu0": np.asarray(
            model.P3D_Mpc_k_mu(
                linear_theory, z, k_Mpc, np.zeros_like(k_Mpc), arinyo_parameters
            )
        ).squeeze(),
        "p3d_mu1": np.asarray(
            model.P3D_Mpc_k_mu(
                linear_theory, z, k_Mpc, np.ones_like(k_Mpc), arinyo_parameters
            )
        ).squeeze(),
    }


# %% [markdown]
# ## Compute one-at-a-time P1D and P3D responses
#
# For every perturbation, ForestFlow is evaluated once and the resulting Arinyo
# model is used for P1D and P3D. We show P3D at $\mu=0$ (purely transverse)
# and $\mu=1$ (line of sight), which separate the angular response of the
# velocity and thermal terms. Each response is relative to the same central
# prediction. The fixed seed ensures that differences are due to the input
# variation rather than different latent samples.

# %%
fiducial_power = predict_power(fiducial_parameters, fiducial_cosmology)
responses = {}


def add_response(parameter_name, values, parameter_sets, cosmologies):
    """Store P1D and P3D fractional responses for one varied input."""
    relative_differences = {name: [] for name in fiducial_power}
    for parameters, target_cosmology in zip(parameter_sets, cosmologies, strict=True):
        prediction = predict_power(parameters, target_cosmology)
        for name, fiducial_value in fiducial_power.items():
            relative_differences[name].append(prediction[name] / fiducial_value - 1.0)
    responses[parameter_name] = {
        "values": np.asarray(values),
        "relative_differences": {
            name: np.asarray(value) for name, value in relative_differences.items()
        },
    }


# Vary As. This changes Delta2_p while preserving a consistent linear spectrum.
fiducial_cosmology_parameters = fiducial_cosmology.input_cosmo_params_dict.copy()
As_values = fiducial_cosmology_parameters["As"] * np.array(
    [1.0 - cosmology_steps["As_fraction"], 1.0 + cosmology_steps["As_fraction"]]
)
Delta2_p_values, parameter_sets, cosmologies = [], [], []
for As_value in As_values:
    cosmology_parameters = fiducial_cosmology_parameters.copy()
    cosmology_parameters["As"] = As_value
    varied_cosmology = Cosmology(cosmo_params_dict=cosmology_parameters)
    linear_parameters = varied_cosmology.get_linP_Mpc_params(z=z, kp_Mpc=kp_Mpc)
    varied_parameters = fiducial_parameters.copy()
    varied_parameters.update(
        {name: linear_parameters[name] for name in ("Delta2_p", "n_p")}
    )
    Delta2_p_values.append(linear_parameters["Delta2_p"])
    parameter_sets.append(varied_parameters)
    cosmologies.append(varied_cosmology)
add_response("Delta2_p", Delta2_p_values, parameter_sets, cosmologies)

# Vary ns, recomputing both compressed linear-power inputs.
ns_values = fiducial_cosmology_parameters["ns"] + np.array(
    [-cosmology_steps["ns"], cosmology_steps["ns"]]
)
n_p_values, parameter_sets, cosmologies = [], [], []
for ns_value in ns_values:
    cosmology_parameters = fiducial_cosmology_parameters.copy()
    cosmology_parameters["ns"] = ns_value
    varied_cosmology = Cosmology(cosmo_params_dict=cosmology_parameters)
    linear_parameters = varied_cosmology.get_linP_Mpc_params(z=z, kp_Mpc=kp_Mpc)
    varied_parameters = fiducial_parameters.copy()
    varied_parameters.update(
        {name: linear_parameters[name] for name in ("Delta2_p", "n_p")}
    )
    n_p_values.append(linear_parameters["n_p"])
    parameter_sets.append(varied_parameters)
    cosmologies.append(varied_cosmology)
add_response("n_p", n_p_values, parameter_sets, cosmologies)

# Vary each IGM input with Planck18 held fixed.
for parameter_name, step in igm_parameter_steps.items():
    varied_values = np.array(
        [
            fiducial_parameters[parameter_name] - step,
            fiducial_parameters[parameter_name] + step,
        ]
    )
    parameter_sets = []
    for varied_value in varied_values:
        varied_parameters = fiducial_parameters.copy()
        varied_parameters[parameter_name] = varied_value
        parameter_sets.append(varied_parameters)
    add_response(
        parameter_name,
        varied_values,
        parameter_sets,
        [fiducial_cosmology] * len(varied_values),
    )

# %% [markdown]
# ## Plot the parameter responses
#
# Each panel shows the P1D response when only its titled parameter is changed.
# Labels show the absolute values used for the lower and upper variations.

# %%
parameter_labels = {
    "Delta2_p": r"$\Delta_p^2$",
    "n_p": r"$n_p$",
    "mF": r"$\bar{F}$",
    "gamma": r"$\gamma$",
    "sigT_Mpc": r"$\sigma_T\,[\mathrm{Mpc}]$",
    "kF_Mpc": r"$k_F\,[\mathrm{Mpc}^{-1}]$",
}


def plot_responses(prediction_name, ylabel, title):
    """Plot all one-at-a-time responses for one power-spectrum observable."""
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for axis, (parameter_name, response) in zip(
        axes.flat, responses.items(), strict=True
    ):
        for varied_value, relative_difference in zip(
            response["values"],
            response["relative_differences"][prediction_name],
            strict=True,
        ):
            axis.plot(
                k_Mpc,
                relative_difference,
                label=f"{parameter_labels[parameter_name]} = {varied_value:.4g}",
            )
        axis.axhline(0.0, color="black", linestyle=":")
        axis.set_xscale("log")
        axis.set_title(parameter_labels[parameter_name])
        axis.legend(fontsize=9)
    for axis in axes[-1]:
        axis.set_xlabel(r"$k\,[\mathrm{Mpc}^{-1}]$")
    for axis in axes[:, 0]:
        axis.set_ylabel(ylabel)
    figure.suptitle(title)
    figure.tight_layout()
    return figure


p1d_figure = plot_responses(
    "p1d",
    r"$P_\mathrm{1D}/P_\mathrm{1D}^{\mathrm{central}}-1$",
    "ForestFlow P1D parameter responses at z=3",
)

# %% [markdown]
# ## P3D response at $\mu=0$
#
# At $\mu=0$ the mode is transverse to the line of sight. Comparing this plot
# to the next one isolates the angular dependence of each parameter response.

# %%
p3d_mu0_figure = plot_responses(
    "p3d_mu0",
    r"$P_\mathrm{3D}(k,\mu=0)/P_\mathrm{3D}^{\mathrm{central}}-1$",
    r"ForestFlow P3D parameter responses at z=3 ($\mu=0$)",
)

# %% [markdown]
# ## P3D response at $\mu=1$
#
# At $\mu=1$ the mode is parallel to the line of sight, where redshift-space
# distortions and thermal effects can differ substantially from $\mu=0$.

# %%
p3d_mu1_figure = plot_responses(
    "p3d_mu1",
    r"$P_\mathrm{3D}(k,\mu=1)/P_\mathrm{3D}^{\mathrm{central}}-1$",
    r"ForestFlow P3D parameter responses at z=3 ($\mu=1$)",
)

# %%
