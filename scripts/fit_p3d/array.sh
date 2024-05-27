#!/bin/bash

# Loop over the range from 10 to 30
for i in {10..30}
do
  nohup python fit_pflux_z.py mpg_$i > nohup$i.out 2>&1 &
done

echo "JDONE!"

