"""Generic P3D bin averages for model callables.

These utilities describe continuous bin geometry.  They are distinct from
:mod:`forestflow.statistics.rebin_p3d`, whose routines reweight discrete
simulation measurements using finite-box Fourier mode counts.
"""

import numpy as np

from forestflow.statistics.binning import linear_bin_edges, logarithmic_bin_edges


__all__ = [
    "P3D_Mpc_k_mu_bin_averaged",
    "P3D_Mpc_k_mu_mode_averaged",
    "P3D_Mpc_k_mu_hybrid_averaged",
    "P3D_Mpc_kpar_kperp_bin_averaged",
]


def _validate_fine_factor(fine_factor):
    if not isinstance(fine_factor, int) or fine_factor < 1:
        raise ValueError("fine_factor must be a positive integer")


def _uniform_edges(centres, edges, name, minimum=None, maximum=None):
    centres = np.asarray(centres, dtype=float)
    if centres.ndim != 1 or centres.size == 0 or not np.all(np.isfinite(centres)):
        raise ValueError(f"{name} centres must be a non-empty finite 1D array")
    if minimum is not None and np.any(centres < minimum):
        raise ValueError(f"{name} centres must be >= {minimum}")
    if maximum is not None and np.any(centres > maximum):
        raise ValueError(f"{name} centres must be <= {maximum}")
    return linear_bin_edges(centres, edges, name=name)


def P3D_Mpc_k_mu_bin_averaged(
    linear,
    z,
    k_iMpc=None,
    mu=None,
    P3D_model=None,
    P3D_params=None,
    k_iMpc_edges=None,
    mu_edges=None,
    fine_factor=8,
    **kwargs,
):
    """
    Average a ``P3D_Mpc(k, mu)`` callable over continuous rectangular bins.

    ``P3D_model`` must accept ``(linear, z, k_iMpc, mu, parameters)``.  With
    explicit ``k_iMpc_edges`` and ``mu_edges``, ``k_iMpc`` and ``mu`` may be
    ``None``: geometric k and arithmetic mu centres are then derived only to
    define the returned array shape.  The numerical average always uses the
    bin edges, never the centre values.  Fine samples are weighted by
    ``k_iMpc**3``: this is the three-dimensional Fourier phase-space measure
    ``k**2 dk dmu`` expressed in ``dln(k) dmu`` coordinates.  Thus this
    function is the continuous, infinite-volume limit of mode averaging.
    """
    _validate_fine_factor(fine_factor)
    if P3D_model is None:
        raise ValueError("P3D_model is required")

    # Explicit edges completely determine the geometry.  In that case derive
    # centres for output indexing and deliberately ignore any supplied centres.
    if k_iMpc_edges is not None:
        k_edges = np.asarray(k_iMpc_edges, dtype=float)
        if (k_edges.ndim != 1 or k_edges.size < 2 or not np.all(np.isfinite(k_edges))
                or np.any(k_edges <= 0) or np.any(np.diff(k_edges) <= 0)):
            raise ValueError("k_iMpc_edges must be finite, positive, and increasing")
        k_iMpc = np.sqrt(k_edges[:-1] * k_edges[1:])
    else:
        if k_iMpc is None:
            raise ValueError("provide k_iMpc centres or explicit k_iMpc_edges")
        k_iMpc = np.asarray(k_iMpc, dtype=float)
        k_edges = logarithmic_bin_edges(k_iMpc, name="k_iMpc")

    inferred_mu_edges = mu_edges is None
    if mu_edges is not None:
        mu_edges = np.asarray(mu_edges, dtype=float)
        if (mu_edges.ndim != 1 or mu_edges.size < 2 or not np.all(np.isfinite(mu_edges))
                or np.any(np.diff(mu_edges) <= 0)):
            raise ValueError("mu_edges must be finite and increasing")
        if np.any(mu_edges < -1.0) or np.any(mu_edges > 1.0):
            raise ValueError("mu_edges must lie in [-1, 1]")
        mu = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    else:
        if mu is None:
            raise ValueError("provide mu centres or explicit mu_edges")
        mu = np.asarray(mu, dtype=float)
        mu_edges = _uniform_edges(mu, None, "mu", minimum=-1.0, maximum=1.0)
        # Archive grids conventionally use |mu| in [0, 1]; preserve that
        # physical boundary when all supplied centres are non-negative.
        lower_bound = 0.0 if np.all(mu >= 0.0) else -1.0
        mu_edges = np.clip(mu_edges, lower_bound, 1.0)

    fractions = (np.arange(fine_factor) + 0.5) / fine_factor
    fine_k = np.exp(
        np.log(k_edges[:-1, None]) + fractions * np.diff(np.log(k_edges))[:, None]
    )
    fine_mu = mu_edges[:-1, None] + fractions * np.diff(mu_edges)[:, None]
    shape = (len(k_iMpc), fine_factor, len(mu), fine_factor)
    k_grid = np.broadcast_to(fine_k[:, :, None, None], shape)
    mu_grid = np.broadcast_to(fine_mu[None, None, :, :], shape)
    values = np.asarray(
        P3D_model(
            linear, z, k_grid, mu_grid, {} if P3D_params is None else P3D_params,
            **kwargs,
        )
    )
    # In (ln k, mu), the isotropic three-dimensional phase-space measure is
    # k**3 dln(k) dmu.  The uniform fine grid therefore needs k**3 weights.
    phase_space_weights = k_grid**3
    numerator = np.sum(values * phase_space_weights, axis=(-3, -1))
    denominator = np.sum(phase_space_weights, axis=(-3, -1))
    return numerator / denominator

def P3D_Mpc_k_mu_mode_averaged(
    linear,
    z,
    P3D_model,
    P3D_params=None,
    *,
    k_mu_modes,
    k_iMpc=None,
    mu=None,
    k_iMpc_edges=None,
    mu_edges=None,
    **kwargs,
):
    """
    Average a P3D model over exact finite-volume Fourier modes.

    Provide exactly one geometry description: both ``k_iMpc`` and ``mu``
    centres, or both ``k_iMpc_edges`` and ``mu_edges``.  The model itself is
    evaluated only at coordinates stored in ``k_mu_modes``.  Edges are the
    preferred description because they are the simulation measurement bins.
    """
    if P3D_model is None:
        raise ValueError("P3D_model is required")
    has_centres = k_iMpc is not None or mu is not None
    has_edges = k_iMpc_edges is not None or mu_edges is not None
    if has_centres == has_edges:
        raise ValueError(
            "provide either both k_iMpc and mu centres or both k_iMpc_edges "
            "and mu_edges"
        )
    if has_edges:
        if k_iMpc_edges is None or mu_edges is None:
            raise ValueError("provide both k_iMpc_edges and mu_edges")
        k_edges = np.asarray(k_iMpc_edges, dtype=float)
        mu_edges = np.asarray(mu_edges, dtype=float)
        if (k_edges.ndim != 1 or k_edges.size < 2 or not np.all(np.isfinite(k_edges))
                or np.any(k_edges <= 0) or np.any(np.diff(k_edges) <= 0)):
            raise ValueError("k_iMpc_edges must be finite, positive, and increasing")
        if (mu_edges.ndim != 1 or mu_edges.size < 2 or not np.all(np.isfinite(mu_edges))
                or np.any(np.diff(mu_edges) <= 0) or np.any(mu_edges < -1.0)
                or np.any(mu_edges > 1.0)):
            raise ValueError("mu_edges must be finite, increasing, and lie in [-1, 1]")
        output_shape = (len(k_edges) - 1, len(mu_edges) - 1)
    else:
        if k_iMpc is None or mu is None:
            raise ValueError("provide both k_iMpc and mu centres")
        k_iMpc = np.asarray(k_iMpc, dtype=float)
        mu = np.asarray(mu, dtype=float)
        if k_iMpc.ndim != 2 or mu.shape != k_iMpc.shape:
            raise ValueError("k_iMpc and mu must be two-dimensional arrays of the same shape")
        output_shape = k_iMpc.shape

    output = None
    parameters = {} if P3D_params is None else P3D_params
    for k_index in range(output_shape[0]):
        for mu_index in range(output_shape[1]):
            key = f"{k_index}_{mu_index}"
            if key + "_k" not in k_mu_modes:
                continue
            values = np.asarray(
                P3D_model(
                    linear, z, k_mu_modes[key + "_k"], k_mu_modes[key + "_mu"],
                    parameters, **kwargs
                )
            )
            averaged = values.mean(axis=-1)
            if output is None:
                output = np.full(averaged.shape + output_shape, np.nan)
            output[(..., k_index, mu_index)] = averaged
    if output is None:
        return np.full(output_shape, np.nan)
    return output


def P3D_Mpc_k_mu_hybrid_averaged(
    linear,
    z,
    P3D_model,
    P3D_params=None,
    *,
    k_mu_modes,
    k_iMpc_edges,
    mu_edges,
    max_discrete_modes=256,
    fine_factor=4,
    **kwargs,
):
    """
    Predict finite-volume P3D bins with exact low-mode cells and fast high-mode cells.

    Bins containing at most ``max_discrete_modes`` Fourier modes are evaluated
    at their exact lattice coordinates.  Populated bins above that threshold
    use :func:`P3D_Mpc_k_mu_bin_averaged`, the phase-space continuous limit.
    This captures the sparse large-scale lattice while retaining the speed of
    continuous averaging where the mode density is high.
    """
    if P3D_model is None:
        raise ValueError("P3D_model is required")
    if not isinstance(max_discrete_modes, (int, np.integer)) or max_discrete_modes < 0:
        raise ValueError("max_discrete_modes must be a non-negative integer")

    continuous = P3D_Mpc_k_mu_bin_averaged(
        linear,
        z,
        P3D_model=P3D_model,
        P3D_params=P3D_params,
        k_iMpc_edges=k_iMpc_edges,
        mu_edges=mu_edges,
        fine_factor=fine_factor,
        **kwargs,
    )
    output = np.array(continuous, copy=True)
    n_k_bins, n_mu_bins = output.shape[-2:]
    parameters = {} if P3D_params is None else P3D_params

    selected_bins = []
    selected_k = []
    selected_mu = []
    for k_index in range(n_k_bins):
        for mu_index in range(n_mu_bins):
            key = f"{k_index}_{mu_index}"
            modes_k = k_mu_modes.get(key + "_k")
            if modes_k is None or len(modes_k) == 0:
                output[(..., k_index, mu_index)] = np.nan
                continue
            if len(modes_k) > max_discrete_modes:
                continue
            selected_bins.append((k_index, mu_index, len(modes_k)))
            selected_k.append(modes_k)
            selected_mu.append(k_mu_modes[key + "_mu"])

    # One batched model call avoids Python and emulator-call overhead for each
    # sparse bin.  The subsequent slices retain the exact per-bin mean.
    if selected_bins:
        values = np.asarray(
            P3D_model(
                linear,
                z,
                np.concatenate(selected_k),
                np.concatenate(selected_mu),
                parameters,
                **kwargs,
            )
        )
        start = 0
        for k_index, mu_index, count in selected_bins:
            output[(..., k_index, mu_index)] = values[..., start : start + count].mean(axis=-1)
            start += count
    return output

def P3D_Mpc_kpar_kperp_bin_averaged(
    linear, z, k_par_iMpc, k_perp_iMpc, p3d_kpar_kperp, P3D_params=None,
    k_par_edges=None, k_perp_edges=None, fine_factor=8, **kwargs
):
    """
    Average a Cartesian ``P3D_Mpc(k_parallel, k_perp)`` callable over bins.

    ``k_parallel`` bins are linearly averaged, so they may include zero or be
    signed.  ``k_perp`` bins are logarithmically averaged and must be positive.
    The callable must accept ``(linear, z, k_par_iMpc, k_perp_iMpc, parameters)``.
    """
    _validate_fine_factor(fine_factor)
    k_par_iMpc = np.asarray(k_par_iMpc, dtype=float)
    k_perp_iMpc = np.asarray(k_perp_iMpc, dtype=float)
    k_par_edges = linear_bin_edges(k_par_iMpc, k_par_edges, name="k_par_iMpc")
    k_perp_edges = logarithmic_bin_edges(
        k_perp_iMpc, k_perp_edges, name="k_perp_iMpc"
    )
    fractions = (np.arange(fine_factor) + 0.5) / fine_factor
    fine_k_par = k_par_edges[:-1, None] + fractions * np.diff(k_par_edges)[:, None]
    fine_k_perp = np.exp(
        np.log(k_perp_edges[:-1, None])
        + fractions * np.diff(np.log(k_perp_edges))[:, None]
    )
    shape = (len(k_par_iMpc), fine_factor, len(k_perp_iMpc), fine_factor)
    k_par_grid = np.broadcast_to(fine_k_par[:, :, None, None], shape)
    k_perp_grid = np.broadcast_to(fine_k_perp[None, None, :, :], shape)
    values = np.asarray(
        p3d_kpar_kperp(
            linear, z, k_par_grid, k_perp_grid,
            {} if P3D_params is None else P3D_params, **kwargs
        )
    )
    return values.mean(axis=(-3, -1))
