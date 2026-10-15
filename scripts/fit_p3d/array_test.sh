#!/bin/bash

# Fit every testing simulation (mpg_central_seed is the combined
# central+seed measurement built from mpg_central and mpg_seed).
# for sim in central seed central_seed curved growth neutrinos reio running
for sim in curved growth neutrinos reio running
do
  python fit_pflux.py mpg_$sim --output Arinyo_fit_mpg_$sim.npy
done

echo "JDONE!"
