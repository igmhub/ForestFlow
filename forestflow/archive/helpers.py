"""Archive measurement helpers."""
from typing import Any
import numpy as np

def get_sim_power(sim: Any, kmax_1d_Mpc: int | None=4, kmax_3d_Mpc: int | None=5) -> Any:

    """
    Return simulation power.

    Parameters
    ----------
    sim : object
        Sim used by the calculation.
    kmax_1d_Mpc : int, optional
        Kmax 1d mpc used by the calculation.
    kmax_3d_Mpc : int, optional
        Kmax 3d mpc used by the calculation.

    Returns
    -------
    object
        Result produced when the function is used to return simulation power.
    """
    data = {}

    mask_1d = (sim["k_Mpc"] <= kmax_1d_Mpc) & (sim["k_Mpc"] > 0)
    k1d_Mpc = sim["k_Mpc"][mask_1d]
    p1d_Mpc = sim["p1d_Mpc"][mask_1d]
    data["sim_k1d_Mpc"] = k1d_Mpc
    data["sim_p1d_Mpc"] = p1d_Mpc

    mask_3d = (sim["k3d_Mpc"] <= kmax_3d_Mpc) & np.isfinite(sim["p3d_Mpc"])
    k3d_Mpc = sim["k3d_Mpc"][mask_3d]
    p3d_Mpc = sim["p3d_Mpc"][mask_3d]
    mu3d = sim["mu3d"][mask_3d]
    data["sim_k3d_Mpc"] = k3d_Mpc
    data["sim_p3d_Mpc"] = p3d_Mpc
    data["sim_mu3d"] = mu3d

    return data
