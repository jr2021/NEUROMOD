#!/bin/bash

virtualenv --no-download ~/ENV
source ~/ENV/bin/activate
pip install --no-index --upgrade pip
pip install numpy --no-index
pip install pandas --no-index
pip install sklearn

for ((i=0;i<$1;i++))
do
	sbatch run.sh
done
