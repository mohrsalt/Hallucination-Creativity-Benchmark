#!/bin/bash
# Exercise 2 submission script - submit.sh
# Below, is the queue
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ncpus=16:ngpus=2
#PBS -l walltime=10:00:00
#PBS -N olstorylogs
module load miniforge3
conda activate cov
./bin/ollama serve &
sleep 5
#ollama rm QLlama
#ollama create FLlama -f ./finllama.modelfile
#ollama list
python3 -m pip install -r req.txt

python3 autfneo.py
