#!/bin/bash
# Exercise 2 submission script - submit.sh
# Below, is the queue
#PBS -q normal
#PBS -j oe
#PBS -l select=3:ncpus=128:ngpus=2
#PBS -l walltime=09:00:00
#PBS -N sampler
module load miniforge3
conda activate myenv

python3 vll.py
