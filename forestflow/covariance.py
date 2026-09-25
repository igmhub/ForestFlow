"""Leave-one-out covariance calculations for ForestFlow emulators."""

import numpy as np

from forestflow.P3D_cINN import P3DEmulator
from forestflow.model_p3d_arinyo import ArinyoModel
from lace.cosmo import cosmology


def data_for_l10_forest(
    archive,
    emulator_label="forest_mpg",
    *,
    fit_label="Arinyo_min",
    kmax_Mpc=4.0,
):
    """Evaluate ForestFlow L1O emulators against their omitted simulations."""
    nsam = len(archive.list_sim_cube)
    suite = archive.list_sim_cube[0].split("_")[0]
    reference = [
        sample
        for sample in archive.training_data
        if sample["sim_label"] == f"{suite}_2"
        and sample.get("val_scaling", 1.0) == 1.0
    ]
    if not reference:
        raise ValueError(f"No reference snapshots found for {suite}_2.")

    zz = np.unique([sample["z"] for sample in reference])
    input_k_Mpc = reference[0]["k_Mpc"]
    select_k = (input_k_Mpc > 0) & (input_k_Mpc < kmax_Mpc)
    k_Mpc = input_k_Mpc[select_k]

    shape = (nsam, len(zz), len(k_Mpc))
    p1d_Mpc_orig = np.zeros(shape)
    p1d_Mpc_sm = np.zeros(shape)
    p1d_Mpc_emu = np.zeros(shape)
    mask = np.ones(shape[:2], dtype=bool)

    for isim, sim_label in enumerate(archive.list_sim_cube):
        print(sim_label)
        testing_data = [
            sample
            for sample in archive.training_data
            if sample["sim_label"] == sim_label
            and sample.get("val_scaling", 1.0) == 1.0
        ]
        if not testing_data:
            mask[isim] = False
            continue

        emulator = P3DEmulator(key=f"l1O/{emulator_label}_l1O_{isim}")
        fiducial_cosmology = cosmology.Cosmology(
            cosmo_params_dict=testing_data[0]["cosmo_params"]
        )
        model = ArinyoModel(fiducial_cosmology)
        linear = model.linear_theory(zz)

        for sample in testing_data:
            iz = np.argmin(np.abs(zz - sample["z"]))
            if abs(zz[iz] - sample["z"]) > 0.01:
                continue
            if not np.allclose(sample["k_Mpc"][select_k], k_Mpc):
                raise ValueError(f"k_Mpc differs for {sim_label}.")
            if fit_label not in sample:
                mask[isim, iz] = False
                continue

            p1d_Mpc_orig[isim, iz] = sample["p1d_Mpc"][select_k]
            p1d_Mpc_sm[isim, iz] = np.squeeze(
                model.P1D_Mpc(linear, sample["z"], k_Mpc, sample[fit_label])
            )
            emulator_input = {
                parameter: sample[parameter] for parameter in emulator.input_labels
            }
            arinyo_parameters = emulator.evaluate(emulator_input)
            p1d_Mpc_emu[isim, iz] = np.squeeze(
                model.P1D_Mpc(linear, sample["z"], k_Mpc, arinyo_parameters)
            )

    return zz, k_Mpc, p1d_Mpc_orig, p1d_Mpc_sm, p1d_Mpc_emu, mask
