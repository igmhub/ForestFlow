import numpy as np

from forestflow.model_fits import ArinyoFitter, FitData


def test_supported_fitting_import_and_parameter_contract():
    assert ArinyoFitter.PARAM_NAMES == (
        "bias",
        "bias_eta",
        "q1",
        "q2",
        "kvav",
        "av",
        "bv",
        "kp",
    )
    assert FitData.__module__ == "forestflow.model_fits.data"


def test_parameter_dict_round_trip_uses_public_order():
    fitter = ArinyoFitter()
    parameters = dict(zip(ArinyoFitter.PARAM_NAMES, range(8)))
    assert fitter.params_to_dict(fitter.params_from_dict(parameters)) == parameters


def test_standard_postprocessing_initialization_is_used_for_missing_fit():
    fitter = ArinyoFitter()
    standard_params = {name: index + 10 for index, name in enumerate(fitter.PARAM_NAMES)}
    simulation = {
        "sim_label": "mpg_4", "ind_snap": 7, "ind_phase": "average",
        "ind_axis": 2, "ind_rescaling": 3, "z": 3.0,
    }
    standard_simulations = [
        {**simulation, "Arinyo_min": standard_params},
        {**simulation, "ind_rescaling": 4,
         "Arinyo_min": {name: -1 for name in fitter.PARAM_NAMES}},
    ]

    assert fitter._initial_parameters_from_simulation(
        simulation, standard_simulations=standard_simulations
    ) == standard_params


def test_initialization_requires_local_or_standard_postprocessing_fit():
    fitter = ArinyoFitter()
    local_params = {name: index for index, name in enumerate(fitter.PARAM_NAMES)}

    assert fitter._initial_parameters_from_simulation(
        {"z": 3.0, "Arinyo_min": local_params}
    ) == local_params

    import pytest

    with pytest.raises(KeyError, match="standard"):
        fitter._initial_parameters_from_simulation({"z": 3.0})


def test_p3d_model_bins_use_the_same_centre_cut_as_simulation_data():
    """Scale cuts select model P3D bins by centre, not by their edges."""
    fitter = ArinyoFitter(n_k_bins=40, kmin_3d=0.7, kmax_3d=4.5)

    assert fitter._p3d_shape == (13, 16)


def test_training_label_selection_reuses_loaded_archive_cache():
    from forestflow.archive.gadget_archive import GadgetArchive3D

    archive = object.__new__(GadgetArchive3D)
    archive.emu_params = ["mF"]
    archive.list_sim_cube = ["mpg_0", "mpg_1"]
    archive.training_data = [
        {"sim_label": "mpg_0", "Arinyo_min": {"bias": -0.1}},
        {"sim_label": "mpg_1", "Arinyo_min": {"bias": -0.2}},
    ]

    selected = archive.get_training_data("mpg_1")
    assert selected == [archive.training_data[1]]
    assert "Arinyo_min" in selected[0]


def test_default_fitter_redshift_grid_matches_mpg_archive():
    from lace.archive.gadget_archive import MPG_SIM_REDSHIFTS

    fitter = ArinyoFitter()
    np.testing.assert_allclose(fitter.zlist, MPG_SIM_REDSHIFTS)
    assert fitter.zlist is not MPG_SIM_REDSHIFTS


def test_save_results_writes_portable_archive_mapping(tmp_path):
    fitter = ArinyoFitter()
    snapshots = [{"z": 3.0, "ind_snap": 1, "ind_phase": "average", "ind_axis": 0,
                  "ind_rescaling": 2}]
    filename = fitter.save_results(
        tmp_path / "fits.npy",
        snapshots=snapshots,
        initial_chi2=[3.0], chi2=[2.0], success=[True], message=["ok"],
        arinyo={name: [index] for index, name in enumerate(fitter.PARAM_NAMES)},
        simulation_label="mpg_0", postproc="Cabayol23",
    )
    saved = np.load(filename, allow_pickle=True).item()
    assert saved["schema_version"] == 2
    assert saved["ind_rescaling"].tolist() == [2]
    assert saved["Arinyo"]["bias"].tolist() == [0.0]


def test_final_bound_warning_names_parameter_and_bound(capsys):
    fitter = ArinyoFitter()
    fitter.best_params = np.array([-0.999, -0.2, 1.0, 0.0, 1.0, 1.0, 2.0, 10.0])
    fitter._warn_if_final_parameters_near_bounds()
    output = capsys.readouterr().out
    assert "bias=-0.999" in output
    assert "lower bound (-1)" in output
