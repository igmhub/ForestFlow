#!/bin/bash

source /data/desi/scratch/jchavesm/mambaforge/bin/activate lace

echo "args: " $1

python /data/desi/scratch/jchavesm/lya_pk/lya_pk/scripts/fit_pflux.py $1

echo "JDONE!"

