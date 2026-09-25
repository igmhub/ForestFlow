"""Fit Arinyo parameters to every snapshot of one Gadget simulation.

This is the supported batch driver.  It uses ``forestflow.fitting`` and saves
the fit result in ordinary (non-LaTex) parameter names.  The other files in
this directory are retained as legacy scripts for historical analyses.
"""

import argparse
from pathlib import Path

import numpy as np

from forestflow.archive import GadgetArchive3D
from forestflow.fitting import ArinyoFitter


def _simulations_for_label(archive, sim_label):
    if sim_label in archive.list_sim_cube:
        return [sim for sim in archive.training_data if sim["sim_label"] == sim_label]
    return archive.get_testing_data(sim_label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_label", help="simulation label to fit")
    parser.add_argument("--output", type=Path, required=True, help="output .npz path")
    parser.add_argument("--kmax-3d", type=float, default=4.5)
    parser.add_argument("--kmax-1d", type=float, default=7.0)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    archive = GadgetArchive3D(average="both")
    snapshots = _simulations_for_label(archive, args.sim_label)
    if not snapshots:
        raise ValueError(f"No snapshots found for simulation {args.sim_label!r}")

    results = []
    for snapshot in snapshots:
        fitter = ArinyoFitter(kmax_3d=args.kmax_3d, kmax_1d=args.kmax_1d)
        fitter.prepare_simulation(snapshot)
        initial_chi2 = fitter.chi2(fitter.params_from_dict(fitter.data.ini_params))
        result = fitter.fit_iterative(niter=args.iterations)
        results.append(
            {
                "z": snapshot["z"],
                "ind_snap": snapshot.get("ind_snap"),
                "val_scaling": snapshot.get("val_scaling"),
                "initial_chi2": initial_chi2,
                "chi2": result.fun,
                "success": result.success,
                "message": result.message,
                "Arinyo": fitter.params_to_dict(fitter.best_params),
            }
        )
        print(f"z={snapshot['z']:.3f}: chi2 {initial_chi2:.4f} -> {result.fun:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, parameter_names=ArinyoFitter.PARAM_NAMES, results=results)
    print(f"Saved {len(results)} fits to {args.output}")


if __name__ == "__main__":
    main()
