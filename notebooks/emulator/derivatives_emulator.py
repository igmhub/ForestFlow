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
# %matplotlib inline
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


def predict_p1d(
    parameters: dict[str, float], target_cosmology: Cosmology
) -> np.ndarray:
    """Predict P1D using one cosmology consistently in ForestFlow and P1D."""
    arinyo_parameters = emulator.evaluate(
        emu_params=parameters,
        Nrealizations=n_realizations,
        seed=0,
    )
    model = ArinyoModel(target_cosmology)
    linear_theory = model.linear_theory(z)
    return np.asarray(
        model.P1D_Mpc(linear_theory, z, k_Mpc, arinyo_parameters)
    ).squeeze()


# %% [markdown]
# ## Compute one-at-a-time P1D responses
#
# The stored quantity is
# $P_\mathrm{1D}/P_\mathrm{1D}^{\mathrm{central}}-1$. The fixed emulator seed
# ensures that differences are caused by the input variation rather than by
# different Monte-Carlo latent samples. For the first two panels, changing
# $A_s$ or $n_s$ consistently changes both the emulator inputs and the linear
# spectrum used by the P1D projection.

# %%
fiducial_p1d = predict_p1d(fiducial_parameters, fiducial_cosmology)

responses = {}

# Vary As. This primarily changes Delta2_p while preserving a physically
# consistent linear spectrum in the P1D projection.
fiducial_cosmology_parameters = fiducial_cosmology.input_cosmo_params_dict.copy()
As_values = fiducial_cosmology_parameters["As"] * np.array(
    [1.0 - cosmology_steps["As_fraction"], 1.0 + cosmology_steps["As_fraction"]]
)
Delta2_p_values = []
Delta2_p_differences = []
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
    Delta2_p_differences.append(
        predict_p1d(varied_parameters, varied_cosmology) / fiducial_p1d - 1.0
    )
responses["Delta2_p"] = {
    "values": np.asarray(Delta2_p_values),
    "relative_difference": np.asarray(Delta2_p_differences),
}

# Vary ns. The resulting Delta2_p and n_p are both recomputed and passed to
# ForestFlow, while the same varied cosmology supplies the projected P1D.
ns_values = fiducial_cosmology_parameters["ns"] + np.array(
    [-cosmology_steps["ns"], cosmology_steps["ns"]]
)
n_p_values = []
n_p_differences = []
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
    n_p_differences.append(
        predict_p1d(varied_parameters, varied_cosmology) / fiducial_p1d - 1.0
    )
responses["n_p"] = {
    "values": np.asarray(n_p_values),
    "relative_difference": np.asarray(n_p_differences),
}

# Vary each IGM input with the fiducial Planck18 cosmology held fixed.
for parameter_name, step in igm_parameter_steps.items():
    varied_values = np.array(
        [
            fiducial_parameters[parameter_name] - step,
            fiducial_parameters[parameter_name] + step,
        ]
    )
    relative_differences = []
    for varied_value in varied_values:
        varied_parameters = fiducial_parameters.copy()
        varied_parameters[parameter_name] = varied_value
        relative_differences.append(
            predict_p1d(varied_parameters, fiducial_cosmology)
            / fiducial_p1d
            - 1.0
        )
    responses[parameter_name] = {
        "values": varied_values,
        "relative_difference": np.asarray(relative_differences),
    }

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

figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
for axis, (parameter_name, response) in zip(
    axes.flat, responses.items(), strict=True
):
    for varied_value, relative_difference in zip(
        response["values"], response["relative_difference"], strict=True
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
    axis.set_xlabel(r"$k_\parallel\,[\mathrm{Mpc}^{-1}]$")
for axis in axes[:, 0]:
    axis.set_ylabel(r"$P_\mathrm{1D}/P_\mathrm{1D}^{\mathrm{central}}-1$")

figure.suptitle("ForestFlow P1D parameter responses at z=3")
figure.tight_layout()
