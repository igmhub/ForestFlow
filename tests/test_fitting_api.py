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
