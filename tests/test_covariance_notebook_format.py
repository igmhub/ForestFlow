"""Exercise only the notebook save cell with small synthetic arrays."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def test_covariance_notebook_saves_native_npy_mapping(tmp_path):
    path = Path(__file__).resolve().parents[1] / "notebooks/emulator/compute_cov.py"
    tree = ast.parse(path.read_text())
    save_cell = next(
        node for node in tree.body
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
        and node.test.id == "save_data"
    )
    zz = np.array([2., 3.])
    k = np.array([.1, .2, .3])
    powers = np.arange(4 * 2 * 3).reshape(4, 2, 3)
    covariance = np.diag(np.arange(1., 7.))
    scope = dict(
        np=np, Path=Path, save_data=True, emulator_label="forest_mpg_fix",
        forestflow=SimpleNamespace(__file__=str(tmp_path / "forestflow/__init__.py")),
        zz=zz, k_Mpc=k, cov_k=np.eye(3), cov_zk=covariance,
        p1d_Mpc_orig=powers, p1d_Mpc_sm=powers, p1d_Mpc_emu=powers,
        rel_diff=powers, mask=np.ones((4, 2), dtype=bool),
    )
    exec(compile(ast.Module(body=[save_cell], type_ignores=[]), str(path), "exec"), scope)
    output = tmp_path / "data/covariance/l1O_cov_forest_mpg_fix.npy"
    saved = np.load(output, allow_pickle=True).item()
    assert isinstance(saved, dict)
    np.testing.assert_array_equal(saved["zz_zk"], [2., 2., 2., 3., 3., 3.])
    np.testing.assert_array_equal(saved["k_Mpc_zk"], [.1, .2, .3, .1, .2, .3])
    np.testing.assert_array_equal(saved["cov_zk"], covariance)
    np.testing.assert_array_equal(saved["p1d_Mpc_emu"], powers)
    np.testing.assert_array_equal(saved["k_Mpc_k"], k)
    assert saved["cov_zk"].shape == (len(saved["zz_zk"]),) * 2
    assert not output.with_suffix(".npz").exists()
