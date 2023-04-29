#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate deep-ensembles

for DATASET in mnist cifar caltech eurosat; do

python train_ensemble.py --dataset $DATASET --out $DATASET-ensembles --image-size 32

done
