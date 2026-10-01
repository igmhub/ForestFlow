# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: lace
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ForestFlow inference accuracy and performance
#
# This notebook records the numerical checks used when changing the emulator
# inference path.  It deliberately separates a *scientific* accuracy check
# from wall-clock timing: timings depend on hardware, whereas relative changes
# in Arinyo coefficients and P1D are portable diagnostics.
#
# Run it after modifying cINN evaluation, latent-realization handling, or the
# P3D-to-P1D integration. The pretrained `forest_mpg_fix` bundle is required.

# %%
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch

from lace.cosmo import cosmology
from forestflow.emulator.p3d_cinn import P3DEmulator
from forestflow.statistics.p1d import P1DIntegrator
from forestflow.model.arinyo import ArinyoModel


torch.set_num_threads(1)

# %% [markdown]
# ## Common prediction point
#
# Use a central, physically representative emulator input.  Keeping the seed
# fixed makes every realization-count comparison deterministic.

# %%
input_params = {
    "Delta2_p": 0.18489945277410613,
    "n_p": -2.331713201486465,
    "mF": 0.23475637218289533,
    "sigT_Mpc": 0.10040737452608385,
    "gamma": 1.2115605945334802,
    "kF_Mpc": 14.191866950067904,
}
z = 3.0
kpar_Mpc = np.geomspace(0.02, 5.0, 80)

emulator = P3DEmulator(key="forest_mpg_fix", compile_model=True)
arinyo_model = ArinyoModel(cosmology.Cosmology())
linear = arinyo_model.linear.get_linear_theory(z)


def relative_difference(value, reference):
    """Maximum elementwise relative difference with a stable zero guard."""

    return np.max(np.abs(value - reference) / np.maximum(np.abs(reference), 1e-30))


def p1d_from_arinyo(arinyo_parameters, integrator):
    """Evaluate P1D with an explicit integration grid for convergence tests."""

    return integrator(
        linear,
        z,
        kpar_Mpc,
        arinyo_model.P3D_Mpc_k_mu,
        arinyo_parameters,
    )


# Warm up compilation.  Do not include this first call in timings.
_ = emulator.evaluate(input_params, Nrealizations=1000)

# %% [markdown]
# ## Latent-realization convergence
#
# The standard 1,000 realizations are the fiducial inference setting.  This quantifies the
# accuracy cost of a smaller Monte Carlo average before considering it for a
# production run.  It also times the complete emulator prediction, not model
# loading or PyTorch compilation.
#
# Nrealizations=1000 looks good

# %%
reference_nrealizations = 50000
reference_arinyo = emulator.evaluate(
    input_params, Nrealizations=reference_nrealizations
)
reference_p1d = p1d_from_arinyo(reference_arinyo, P1DIntegrator(n_k_perp=99))

realization_results = []
for ii, nrealizations in enumerate((30000, 3000, 1000, 500)):
    # A compiled PyTorch model specializes on first use of a new shape.
    # Warm this realization count so the timing is steady-state inference.
    _ = emulator.evaluate(input_params, Nrealizations=nrealizations)
    timings = []
    nn = 100
    p1ds = np.zeros((nn, reference_p1d.shape[1]))
    for _ in range(nn):
        start = perf_counter()
        arinyo = emulator.evaluate(
            input_params, Nrealizations=nrealizations, seed=_ * nrealizations
        )
        # p1ds[_, :] = p1d_from_arinyo(arinyo, P1DIntegrator(n_k_perp=99))
        # plt.plot(
        #     kpar_Mpc,
        #     p1ds[_, :] / reference_p1d[0] - 1,
        #     color="C"+str(ii),
        #     alpha=0.5
        # )
        timings.append(perf_counter() - start)
    seconds = float(np.median(timings))

    realization_results.append(
        {
            "Nrealizations": nrealizations,
            "seconds": seconds,
        }
    )

# plt.legend()

for result in realization_results:
    print(
        f"N={result['Nrealizations']:4d}  {result['seconds'] * 1e3:7.2f} ms  "
    )

# %% [markdown]
# ## P3D-to-P1D quadrature convergence
#
# This holds the Arinyo parameters fixed and isolates the numerical integral.
# The default 99-point grid remains the scientific reference.  Any lower-cost
# grid should be adopted only if its P1D error is negligible for the analysis.
#
# Simpson 48 looks good

# %%
reference_nrealizations = 50000
reference_arinyo = emulator.evaluate(
    input_params, Nrealizations=reference_nrealizations
)
reference_p1d = p1d_from_arinyo(reference_arinyo, P1DIntegrator(n_k_perp=999))


quadrature_results = []
quadrature_grids = {
    "simpson": (99, 64, 48, 32),
    "gauss_legendre": (99, 64),
}

for method, node_counts in quadrature_grids.items():
    for n_k_perp in node_counts:
        integrator = P1DIntegrator(n_k_perp=n_k_perp, method=method)
        # The second evaluation exercises the geometry cache.
        _ = p1d_from_arinyo(reference_arinyo, integrator)
        nn = 10
        start = perf_counter()
        for ii in range(nn):
            p1d = p1d_from_arinyo(reference_arinyo, integrator)
        seconds = perf_counter() - start

        plt.plot(
            kpar_Mpc,
            p1d[0] / reference_p1d[0] - 1,
            alpha=0.5,
            label=method + "_" + str(n_k_perp)
        )

        quadrature_results.append(
            {
                "method": method,
                "n_k_perp": n_k_perp,
                "seconds": seconds,
                "max_P1D_relative_difference": relative_difference(p1d, reference_p1d),
                "cached_geometries": len(integrator._geometry_cache),
            }
        )
plt.xscale("log")
plt.legend()

for result in quadrature_results:
    print(
        f"{result['method']:14s} n_k_perp={result['n_k_perp']:2d}  "
        f"{result['seconds'] * 1e3:7.2f} ms  "
        f"max |d P1D/P1D|={result['max_P1D_relative_difference']:.3e}  "
        f"cached geometries={result['cached_geometries']}"
    )

# %%
