"""Fit Arinyo parameters to every snapshot of one Gadget simulation.

This is the supported batch driver. It can fit either standard Cabayol23 or
corrected Cabayol23_fixp3d measurements, using the matching standard snapshot
as initialization when a corrected snapshot has no stored Arinyo fit. Results
use ordinary (non-LaTex) parameter names.
"""

import argparse
from pathlib import Path

import numpy as np

from forestflow.archive.gadget_archive import GadgetArchive3D
from forestflow.model_fits import ArinyoFitter


def _simulations_for_label(archive, sim_label):
    if sim_label in archive.list_sim_cube:
        return [sim for sim in archive.training_data if sim["sim_label"] == sim_label]
    return archive.get_testing_data(sim_label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sim_label", help="simulation label to fit")
    parser.add_argument("--output", type=Path, required=True, help="output .npy path")
    parser.add_argument(
        "--postproc",
        default="Cabayol23_fixp3d",
        choices=("Cabayol23", "Cabayol23_fixp3d"),
        help="post-processing to fit (default: corrected Cabayol23_fixp3d)",
    )
    parser.add_argument("--kmax-3d", type=float, default=4.5)
    parser.add_argument("--kmax-1d", type=float, default=6.0)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    archive = GadgetArchive3D(postproc=args.postproc, average="both")
    snapshots = _simulations_for_label(archive, args.sim_label)
    if not snapshots:
        raise ValueError(f"No snapshots found for simulation {args.sim_label!r}")

    standard_snapshots = None
    if args.postproc == "Cabayol23_fixp3d" and args.sim_label in archive.list_sim_cube:
        # The standard archive supplies the matching fitted Arinyo_min values.
        # Use its constructor-populated cache, rather than a fresh
        # get_training_data call that would omit those fits.
        standard_archive = GadgetArchive3D(postproc="Cabayol23", average="both")
        standard_snapshots = [
            snapshot
            for snapshot in standard_archive.training_data
            if snapshot["sim_label"] == args.sim_label
        ]

    fitter = ArinyoFitter(kmax_3d=args.kmax_3d, kmax_1d=args.kmax_1d)
    result_fits = {name: np.full(len(snapshots), np.nan) for name in fitter.PARAM_NAMES}
    initial_chi2_all = np.full(len(snapshots), np.nan)
    final_chi2_all = np.full(len(snapshots), np.nan)
    success = np.zeros(len(snapshots), dtype=bool)
    messages = np.empty(len(snapshots), dtype=object)

    for index, snapshot in enumerate(snapshots):
        print(index, len(snapshots))
        fitter.prepare_simulation(
            snapshot,
            standard_simulations=standard_snapshots,
            is_mpg=True,
        )
        initial_chi2_all[index] = fitter.chi2(
            fitter.params_from_dict(fitter.data.ini_params)
        )
        result = fitter.fit_iterative(niter=args.iterations)
        for name, value in fitter.params_to_dict(fitter.best_params).items():
            result_fits[name][index] = value
        final_chi2_all[index] = result.fun
        success[index] = result.success
        messages[index] = result.message
        print(
            f"z={snapshot['z']:.3f}: chi2 "
            f"{initial_chi2_all[index]:.4f} -> {result.fun:.4f}"
        )

    output = fitter.save_results(
        args.output,
        snapshots=snapshots,
        initial_chi2=initial_chi2_all,
        chi2=final_chi2_all,
        success=success,
        message=messages,
        arinyo=result_fits,
        simulation_label=args.sim_label,
        postproc=args.postproc,
    )
    print(f"Saved {len(snapshots)} fits to {output}")


if __name__ == "__main__":
    main()
