#!/bin/bash

#SBATCH -c 1
#SBATCH --mem=2048
#SBATCH -t 0-15:0:0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jake.robertson@queensu.ca
#SBATCH --qos=privileged

python3 algorithm.py
