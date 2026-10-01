#!/usr/bin/env bash
# Train every ForestFlow leave-one-out emulator bundle.
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"

# The suite contains one emulator without each hypercube simulation and one
# additional emulator without mpg_central.
simulations=(mpg_{8..29} mpg_central)
for simulation in "${simulations[@]}"; do
    echo "================================================================"
    echo "Training leave-one-out emulator without ${simulation}."
    python scripts/training_emulators/train_emulator.py --simulations "$simulation"
done

echo "Finished all ${#simulations[@]} leave-one-out emulator trainings."
