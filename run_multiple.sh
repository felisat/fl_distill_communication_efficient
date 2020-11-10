#!/bin/bash

max=$1
step=$2

head -n 50 exec.sh
echo "Running $max experiments..."

for (( c=0; c<$max; c+=$step ))
do
   sbatch exec.sh "--start $c --end $(($c+$step))"
done