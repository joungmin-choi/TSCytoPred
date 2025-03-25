#!/bin/bash

dataDir="./example"	# Dataset directory containing files
resDir="./results"	# Result directory to save results
num_seq_length=3 	# Number of timepoints per patient has
pred_range_days=15	# Number of total timepoints between the start and the end timepoint for each patient
train_epochs=2000	# Training epoch

python3 tsCytoPred.py $dataDir $resDir $num_seq_length $pred_range_days $train_epochs