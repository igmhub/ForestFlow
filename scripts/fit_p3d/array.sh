#!/bin/bash

# Loop over the range from 0 to 29
for i in {1..29}
do
  python fit_pflux.py mpg_$i --output Arinyo_fit_mpg_$i.npy
done

echo "JDONE!"

