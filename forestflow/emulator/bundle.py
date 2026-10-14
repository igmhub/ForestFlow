"""
Self-describing, checksum-validated ForestFlow emulator bundles.
"""

from __future__ import annotations

import hashlib
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping


MODEL_BUNDLE_SCHEMA_VERSION = 1


class ModelBundleError(RuntimeError):
    """
    Raised when a model bundle is missing, corrupted, or incompatible.
    """


def manifest_path(model_path: str | Path) -> Path:
    """
    Return the manifest filename associated with a model prefix.

    Parameters
    ----------
    model_path : str or path-like
        Weights prefix without the generated manifest suffix.

    Returns
    -------
    pathlib.Path
        ``<model_path>_manifest.json``.
    """
    return Path(str(model_path) + "_manifest.json")


def sha256(path: str | Path) -> str:
    """
    Compute the SHA-256 digest of an emulator bundle artefact.

    Parameters
    ----------
    path : str or path-like
        Binary file to hash.

    Returns
    -------
    str
        Lowercase hexadecimal digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_versions() -> dict[str, str]:
    """
    Report versions of numerical dependencies used to train a model.

    Returns
    -------
    dict of str to str
        Python and relevant package versions; absent optional packages are
        recorded as ``"unavailable"``.
    """
    result = {"python": ".".join(map(str, sys.version_info[:3]))}
    for package in ("numpy", "scipy", "torch", "freia", "lace"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "unavailable"
    return result


def write_manifest(
    model_path: str | Path,
    transform_path: str | Path | None,
    training_provenance: Mapping[str, Any] | None = None,
) -> Path:
    """
    Write checksums and provenance for a complete emulator bundle.

    Parameters
    ----------
    model_path : str or path-like
        Weights prefix for ``.pt`` and ``_metadata.npy`` artefacts.
    transform_path : str or path-like, optional
        Transformation artefact included in the inventory.
    training_provenance : mapping, optional
        JSON-serializable training metadata.

    Returns
    -------
    pathlib.Path
        Written JSON manifest path.

    Raises
    ------
    FileNotFoundError
        If required bundle artefacts are absent.
    """
    import forestflow

    model_path = Path(model_path)
    artefacts = {
        "weights": Path(str(model_path) + ".pt"),
        "metadata": Path(str(model_path) + "_metadata.npy"),
    }
    if transform_path is not None:
        artefacts["transformations"] = Path(transform_path)

    missing = [str(path) for path in artefacts.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot write an emulator manifest; missing artefacts: " + ", ".join(missing)
        )

    manifest = {
        "schema_version": MODEL_BUNDLE_SCHEMA_VERSION,
        "forestflow_version": getattr(forestflow, "__version__", "unknown"),
        "dependencies": runtime_versions(),
        "training_provenance": dict(training_provenance or {}),
        "artefacts": {
            name: {"filename": path.name, "sha256": sha256(path)}
            for name, path in artefacts.items()
        },
    }
    target = manifest_path(model_path)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def load_manifest(
    model_path: str | Path, transform_path: str | Path | None
) -> dict[str, Any] | None:
    """
    Validate a bundle manifest before loading NumPy or Torch payloads.

    Parameters
    ----------
    model_path : str or path-like
        Weights prefix for the bundle.
    transform_path : str or path-like, optional
        Transformation file supplied to the loader.

    Returns
    -------
    dict or None
        Parsed validated manifest, or ``None`` for a legacy bundle without one.

    Raises
    ------
    ModelBundleError
        If a manifest is unreadable, incompatible, incomplete, or has missing
        or checksum-mismatched artefacts.
    """
    model_path = Path(model_path)
    path_manifest = manifest_path(model_path)
    if not path_manifest.is_file():
        return None
    try:
        manifest = json.loads(path_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelBundleError(f"Cannot read model manifest {path_manifest}: {error}") from error
    if manifest.get("schema_version") != MODEL_BUNDLE_SCHEMA_VERSION:
        raise ModelBundleError(f"Unsupported model manifest schema in {path_manifest}")

    artefacts = manifest.get("artefacts")
    if not isinstance(artefacts, dict) or {"weights", "metadata"} - artefacts.keys():
        raise ModelBundleError(f"Model manifest {path_manifest} has an incomplete artefact inventory")
    paths = {
        "weights": Path(str(model_path) + ".pt"),
        "metadata": Path(str(model_path) + "_metadata.npy"),
    }
    if "transformations" in artefacts:
        if transform_path is None:
            raise ModelBundleError(
                "This model bundle requires its transformation file; provide transf_file."
            )
        paths["transformations"] = Path(transform_path)

    for name, path in paths.items():
        expected = artefacts.get(name)
        if not isinstance(expected, dict) or not isinstance(expected.get("sha256"), str):
            raise ModelBundleError(f"Model manifest has no checksum for {name}")
        if not path.is_file():
            raise ModelBundleError(f"Required model artefact is missing: {path}")
        if sha256(path) != expected["sha256"]:
            raise ModelBundleError(
                f"Checksum mismatch for {name} ({path}); restore the complete trusted bundle."
            )
    return manifest
