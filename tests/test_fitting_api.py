from forestflow.fitting import ArinyoFitter, FitData


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
    assert FitData.__module__ == "forestflow.new_fit.ArinyoFitter"


def test_parameter_dict_round_trip_uses_public_order():
    fitter = ArinyoFitter()
    parameters = dict(zip(ArinyoFitter.PARAM_NAMES, range(8)))
    assert fitter.params_to_dict(fitter.params_from_dict(parameters)) == parameters


def test_prepare_simulation_uses_matching_central_initialization():
    """Missing snapshot fits fall back to mpg-central at the same redshift."""
    fitter = ArinyoFitter()
    central_params = {name: index for index, name in enumerate(fitter.PARAM_NAMES)}
    params = fitter._initial_parameters_from_simulation(
        {"z": 3.0},
        central_simulations=[{"z": 2.75, "Arinyo_min": {}}, {"z": 3.0, "Arinyo_min": central_params}],
    )

    assert params == central_params
    assert params is not central_params


def test_prepare_simulation_prefers_local_initialization_and_can_disable_fallback():
    fitter = ArinyoFitter()
    local_params = {name: index for index, name in enumerate(fitter.PARAM_NAMES)}

    assert fitter._initial_parameters_from_simulation(
        {"z": 3.0, "Arinyo_min": local_params}
    ) == local_params

    import pytest

    with pytest.raises(KeyError, match="fallback_to_central"):
        fitter._initial_parameters_from_simulation(
            {"z": 3.0}, fallback_to_central=False
        )
