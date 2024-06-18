#!/bin/bash

# Loop over the range from 0 to 29
for i in {0..10}
do
  nohup python fit_pflux.py mpg_$i > nohup$i.out 2>&1 &
done

echo "JDONE!"

