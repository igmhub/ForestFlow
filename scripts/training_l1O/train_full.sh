#!/usr/bin/env bash
# Train the full ForestFlow emulator on the hypercube plus mpg_central.
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"

echo "Training the full emulator: hypercube simulations plus mpg_central."
python scripts/training_l1O/train_emulator.py --full "$@"
