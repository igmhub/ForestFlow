"""Sphinx configuration for the ForestFlow documentation."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / "_build/matplotlib"))

project = "ForestFlow"
author = "ForestFlow developers"
try:
    release = importlib.metadata.version("forestflow")
except importlib.metadata.PackageNotFoundError:
    release = "development"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
if importlib.util.find_spec("numpydoc") is not None:
    extensions.insert(0, "numpydoc")
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
if "numpydoc" in extensions:
    numpydoc_show_class_members = False
    numpydoc_validation_checks = {"GL01", "SS01", "SS02", "SS05"}
    numpydoc_validation_exclude = {r"\.undocumented_method$", r".__weakref__$"}

# Keep API discovery usable on Read the Docs without installing heavyweight
# scientific and project-specific dependencies.
autodoc_mock_imports = [
    "astropy",
    "corner",
    "emcee",
    "FrEIA",
    "getdist",
    "lace",
    "matplotlib",
    "psutil",
    "scipy",
    "torch",
    # Compatibility import paths retained by legacy ForestFlow modules.
    "forestflow.fit_p3d",
    "forestflow.likelihood",
    "forestflow.plot_routines",
]

html_theme = (
    "pydata_sphinx_theme"
    if importlib.util.find_spec("pydata_sphinx_theme") is not None
    else "alabaster"
)
html_title = f"ForestFlow {release}"
html_theme_options = (
    {"show_toc_level": 2, "navigation_with_keys": True}
    if html_theme == "pydata_sphinx_theme"
    else {}
)
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
intersphinx_mapping = {}
if os.environ.get("FORESTFLOW_DOCS_INTERSPHINX") == "1":
    intersphinx_mapping = {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable", None),
        "scipy": ("https://docs.scipy.org/doc/scipy", None),
    }
