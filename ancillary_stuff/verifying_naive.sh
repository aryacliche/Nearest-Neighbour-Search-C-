#!/bin/bash

dataset=$1
algo=$2
K=$3

if [ "$dataset" == "mirflickr" ]; then
    train_file="/hard-disk-2/users/aryavishe/VGGNET_features_MIRFLICKR_partitioned/train"
    val_file="/hard-disk-2/users/aryavishe/VGGNET_features_MIRFLICKR_partitioned/val"
elif [ "$dataset" == "imagenet" ]; then
    train_file="/home/aryavishe/Nearest-Neighbour-Search/VGGNET_features_IMAGENET/train"
    val_file="/home/aryavishe/Nearest-Neighbour-Search/VGGNET_features_IMAGENET/val"
else
    echo "Invalid dataset"
fi

temp_file="/home/aryavishe/Nearest-Neighbour-Search-C-/alt_temp/${dataset}/${algo}/naive_neighbours_returned.csv"

python3 verifying_naive.py $temp_file $train_file $val_file $K