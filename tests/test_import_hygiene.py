"""Importing ForestFlow must not need torch, import cup1d, or change rcParams."""

import subprocess
import sys

_BLOCK = """
import sys

class Block:
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in {blocked!r}:
            raise ImportError("blocked " + name)

sys.meta_path.insert(0, Block())
"""


def _run(code, blocked=()):
    script = _BLOCK.format(blocked=tuple(blocked)) + code
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr


def test_gp_path_and_analysis_modules_import_without_torch_or_cup1d():
    _run(
        """
import forestflow.emulator
from forestflow.emulator import P1DEmulator, load_emulator
import forestflow.fitting, forestflow.statistics, forestflow.archive
import forestflow.priors.loaders, forestflow.priors.processing
import forestflow.priors.importance, forestflow.priors.plots
""",
        blocked=("torch", "FrEIA", "cup1d", "lya_interface", "getdist"),
    )


def test_plotting_imports_leave_rcparams_unchanged_until_called():
    _run(
        """
import matplotlib
matplotlib.use("Agg")
from matplotlib import rcParams
before = dict(rcParams)
import forestflow.plots.test_sims, forestflow.plots.params_z, forestflow.priors.plots
assert dict(rcParams) == before
from forestflow.plots import apply_house_style
apply_house_style()
assert rcParams["mathtext.fontset"] == "stix"
assert rcParams["font.family"] == ["STIXGeneral"]
"""
    )
