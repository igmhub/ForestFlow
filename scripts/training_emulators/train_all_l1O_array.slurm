#!/usr/bin/env bash
# Submit with: sbatch scripts/training_emulators/train_all_l1O_array.slurm
#
# One array task trains one independent leave-one-out emulator. Tasks 0--29
# omit mpg_0--mpg_29; task 30 omits mpg_central.  At most eight tasks run at
# once, so the array does not request the entire suite's resources at once.
#SBATCH --job-name=forestflow-l1o
#SBATCH --account=desi
# premium is NERSC's priority QOS; it carries a higher charge multiplier.
#SBATCH --qos=premium
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --array=0-30%8
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"

# The cINN is small; a modest, explicitly bound thread count avoids
# oversubscription from PyTorch, OpenMP, and BLAS on a shared CPU allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

if (( SLURM_ARRAY_TASK_ID < 30 )); then
    simulation="mpg_${SLURM_ARRAY_TASK_ID}"
else
    simulation="mpg_central"
fi

echo "Array task ${SLURM_ARRAY_TASK_ID}: training l1O emulator without ${simulation}"
echo "Allocated CPU threads: ${SLURM_CPUS_PER_TASK}"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores \
    python scripts/training_emulators/train_emulator.py --simulations "$simulation"
