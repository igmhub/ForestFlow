import json

import pytest

from forestflow.model_manifest import ModelBundleError, load_manifest, write_manifest


def _bundle(tmp_path):
    model_path = tmp_path / "emulator.v1"
    weights = tmp_path / "emulator.v1.pt"
    metadata = tmp_path / "emulator.v1_metadata.npy"
    transformations = tmp_path / "emulator.v1_transf.npy"
    for path in (weights, metadata, transformations):
        path.write_bytes(path.name.encode())
    write_manifest(
        model_path,
        transformations,
        {"archive_class": "GadgetArchive3D", "z": 3.0},
    )
    return model_path, weights, metadata, transformations


def test_manifest_records_and_validates_complete_bundle(tmp_path):
    model_path, _, _, transformations = _bundle(tmp_path)
    manifest = load_manifest(model_path, transformations)
    assert manifest["training_provenance"]["archive_class"] == "GadgetArchive3D"
    assert set(manifest["artefacts"]) == {"weights", "metadata", "transformations"}


def test_manifest_rejects_changed_transformation_before_model_loading(tmp_path):
    model_path, _, _, transformations = _bundle(tmp_path)
    transformations.write_bytes(b"changed")
    with pytest.raises(ModelBundleError, match="transformations"):
        load_manifest(model_path, transformations)


def test_manifest_rejects_missing_weights(tmp_path):
    model_path, weights, _, transformations = _bundle(tmp_path)
    weights.unlink()
    with pytest.raises(ModelBundleError, match="missing"):
        load_manifest(model_path, transformations)


def test_manifest_rejects_incomplete_inventory(tmp_path):
    model_path, _, _, transformations = _bundle(tmp_path)
    path_manifest = tmp_path / "emulator.v1_manifest.json"
    contents = json.loads(path_manifest.read_text())
    del contents["artefacts"]["metadata"]
    path_manifest.write_text(json.dumps(contents))
    with pytest.raises(ModelBundleError, match="incomplete"):
        load_manifest(model_path, transformations)
