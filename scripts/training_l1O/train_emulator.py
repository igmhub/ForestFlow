#!/usr/bin/env python
"""Train ForestFlow full and leave-one-simulation-out (l1O) emulator bundles.

Each bundle is trained on the corrected MP-Gadget training archive with one
hypercube simulation excluded.  The output names follow the convention used by
``forestflow.emulator.covariance``::

    data/emulator_models/l1O/forest_mpg_fix_l1O_<index>.pt

For example, to train every l1O emulator with the documented production
settings, run from the ForestFlow repository root::

    python scripts/training_l1O/train_emulator.py

Use ``--simulations mpg_0 mpg_7`` to train or re-train selected bundles.
To train the full corrected emulator, including the central simulation, use
``--full``.  ``--simulations mpg_central`` instead trains the corresponding
leave-one-out emulator without that central simulation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import forestflow
from forestflow.archive.gadget_archive import GadgetArchive3D
from forestflow.emulator.p3d_cinn import P3DEmulator
from forestflow.emulator.training import Transf_data, get_training_data


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulations",
        nargs="+",
        metavar="LABEL",
        help="simulation labels to omit; defaults to every training simulation",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "train one emulator using every hypercube simulation and mpg_central, "
            "rather than the default leave-one-out suite"
        ),
    )
    parser.add_argument(
        "--full-name",
        default="forest_mpg_fix",
        help="bundle name for --full (default: forest_mpg_fix)",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--zmax", type=float, default=4.6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=30)
    parser.add_argument(
        "--validation-set",
        action="store_true",
        help="reserve the emulator's validation split (off in the notebook defaults)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing complete bundle instead of skipping it",
    )
    return parser.parse_args()


def _bundle_exists(model_path: Path, transform_path: Path) -> bool:
    """Return whether all files required to load a saved emulator exist."""
    return all(
        path.is_file()
        for path in (
            model_path.with_suffix(".pt"),
            model_path.with_name(model_path.name + "_metadata.npy"),
            model_path.with_name(model_path.name + "_manifest.json"),
            transform_path,
        )
    )


def _print_training_configuration(args: argparse.Namespace, description: str) -> None:
    """Print the scientific dataset and cINN settings for one planned run."""
    print("=" * 72)
    print(description)
    print("Post-processing: Cabayol23_fixp3d; fitted parameters: arinyo_fixp3d")
    print(
        "Network: "
        f"{args.layers} cINN layers, hidden dimension {args.hidden_dim}, "
        f"batch size {args.batch_size}"
    )
    print(
        f"Optimisation: {args.epochs} epochs, learning rate {args.learning_rate:g}, "
        f"validation set={'on' if args.validation_set else 'off'}"
    )
    print(f"Training redshifts: z <= {args.zmax:g}")
    print("Arinyo-fit cuts: P3D k < 4.5 iMpc; P1D k < 6.0 iMpc")


def train_emulator(
    archive: GadgetArchive3D,
    args: argparse.Namespace,
    name: str,
    output_directory: Path,
    description: str,
    excluded_simulation: str | None = None,
) -> Path | None:
    """Train one emulator bundle from the selected corrected archive snapshots."""
    output_directory.mkdir(parents=True, exist_ok=True)
    model_path = output_directory / name
    transform_path = output_directory / f"{name}_transf.npy"
    if _bundle_exists(model_path, transform_path) and not args.overwrite:
        print(f"Skipping {name}: complete bundle already exists at {model_path}")
        return None

    _print_training_configuration(args, description)
    print(f"Output bundle: {model_path}")
    emulator_data = get_training_data(
        archive.training_data,
        zmax=args.zmax,
        type_fit="arinyo_fixp3d",
        drop_sim=excluded_simulation,
    )
    print(
        "Training snapshots retained: "
        f"{len(next(iter(emulator_data['input_par'].values())))}"
    )
    transform = Transf_data(
        dict_all_params=emulator_data,
        save_file=transform_path,
        compute_fisher=False,
    )
    training_data = {
        "input_par": transform.transf_stand(
            emulator_data["input_par"], type_stand="input", direct=True
        ),
        "output_par": transform.transf_stand(
            emulator_data["output_par"], type_stand="output", direct=True
        ),
    }
    provenance = {
        "postproc": "Cabayol23_fixp3d",
        "type_fit": "arinyo_fixp3d",
        "zmax": args.zmax,
        "kmax_3d_iMpc": 4.5,
        "kmax_1d_iMpc": 6.0,
        "leave_one_out": excluded_simulation is not None,
        "includes_mpg_central": excluded_simulation != "mpg_central",
    }
    if excluded_simulation is not None:
        provenance["excluded_simulation"] = excluded_simulation

    P3DEmulator(
        training_data=training_data,
        train=True,
        nLayers_inn=args.layers,
        nepochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        dims_int=args.hidden_dim,
        use_val_set=args.validation_set,
        save_path=str(model_path),
        training_provenance=provenance,
        model_domain={
            "kp_iMpc": 0.7,
            "kmax_3d_iMpc": 4.5,
            "kmax_1d_iMpc": 6.0,
            "zmax": args.zmax,
            "list_sim_cube": archive.list_sim_cube,
        },
    )
    print(f"Saved {model_path}")
    return model_path


def main() -> None:
    """Run a full training or the selected leave-one-out trainings."""
    args = _parse_arguments()
    if args.full and args.simulations:
        raise ValueError("--full and --simulations cannot be used together")
    archive = GadgetArchive3D(postproc="Cabayol23_fixp3d", addcentral=True)
    repository = Path(forestflow.__path__[0]).parent
    if args.full:
        train_emulator(
            archive,
            args,
            name=args.full_name,
            output_directory=repository / "data" / "emulator_models",
            description=(
                f"Training full emulator {args.full_name} with all "
                f"{len(archive.list_sim_cube)} hypercube simulations plus mpg_central"
            ),
        )
        return

    training_labels = {snapshot["sim_label"] for snapshot in archive.training_data}
    labels = args.simulations or archive.list_sim_cube
    completed = []
    for label in labels:
        if label not in training_labels:
            raise ValueError(
                f"Unknown training simulation {label!r}; expected one of "
                f"{sorted(training_labels)}"
            )
        if label == "mpg_central":
            name = "forest_mpg_fix_l1O_mpg_central"
        else:
            name = f"forest_mpg_fix_l1O_{archive.list_sim_cube.index(label)}"
        result = train_emulator(
            archive,
            args,
            name=name,
            output_directory=repository / "data" / "emulator_models" / "l1O",
            description=f"Training l1O emulator {name}; excluding {label}",
            excluded_simulation=label,
        )
        if result is not None:
            completed.append(result)
    print(f"Finished {len(completed)} l1O emulator training run(s).")


if __name__ == "__main__":
    main()
