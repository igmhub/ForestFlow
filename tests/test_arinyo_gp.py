import numpy as np

from forestflow.emulator.arinyo_gp import (
    ArinyoGPConfig,
    prepare_arinyo_gp_training_data,
    select_groups,
    split_groups,
    train_arinyo_gp,
    transformed_metrics,
)


class AffineTransform:
    """Small invertible stand-in for the stored ForestFlow transform."""

    def transf_stand(self, values, *, type_stand, direct):
        offset, scale = (10.0, 2.0) if type_stand == "input" else (3.0, 4.0)
        if direct:
            return {name: (np.asarray(value) - offset) / scale for name, value in values.items()}
        return {name: np.asarray(value) * scale + offset for name, value in values.items()}


def _records():
    result = []
    for group_index, group in enumerate(("sim_a", "sim_b", "sim_c", "sim_d")):
        for row in range(3):
            x = 10.0 + group_index + row / 10
            result.append({
                "x": x, "z": 2.0 + row, "sim": group,
                "fits": {"a": 3.0 + 2 * x, "b": 4.0 - x},
            })
    return result


def test_group_split_and_transformed_gp_prediction_round_trip(tmp_path):
    records = _records()
    transform = AffineTransform()
    data = prepare_arinyo_gp_training_data(
        records, transform, ("x",), ("a", "b"), fit_key="fits", group_key="sim"
    )
    train_groups, validation_groups, test_groups = split_groups(
        data.group_ids, validation_groups=1, test_groups=1, seed=9
    )
    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    train = select_groups(data, train_groups)
    emulator = train_arinyo_gp(
        train, transform, ArinyoGPConfig(alpha=1e-8, optimize_kernel=False)
    )
    transformed, std = emulator.predict_transformed({"x": 10.25}, return_std=True)
    assert set(transformed) == {"a", "b"}
    assert set(std) == {"a", "b"}
    physical = emulator.evaluate({"x": 10.25})
    assert np.isfinite(list(physical.values())).all()
    batch = emulator.evaluate([{"x": 10.25}, {"x": 10.5}])
    assert batch["a"].shape == (2,)
    path = tmp_path / "arinyo_gp.pkl"
    emulator.save(path)
    loaded = type(emulator).load(path)
    np.testing.assert_allclose(loaded.evaluate({"x": 10.25})["a"], physical["a"])


def test_gp_data_and_metric_validation_errors():
    transform = AffineTransform()
    records = _records()
    try:
        prepare_arinyo_gp_training_data(records, transform, ("missing",), ("a",), fit_key="fits")
    except KeyError:
        pass
    else:
        raise AssertionError("missing input label was accepted")
    metrics = transformed_metrics(np.array([[1.0, 2.0]]), np.array([[0.0, 0.0]]))
    np.testing.assert_allclose(metrics["rmse"], [1.0, 2.0])
