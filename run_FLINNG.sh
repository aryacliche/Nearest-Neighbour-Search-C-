#!/bin/bash

dataset=$1
group_creation_algorithm=$2
R=${3:-3}
N_force=${4:--1}
temp_name=${5:-temp}
temp_dir="temp/${dataset}/${group_creation_algorithm}"

if [ ! -d "${temp_dir}" ]; then
    mkdir -p "${temp_dir}"
fi

if [ "${dataset}" == "imagenet" ]; then
    train_suffix="data/vggnet_imagenet_train"
    val_suffix="data/vggnet_imagenet_val"
elif [ "${dataset}" == "mirflickr" ]; then
    train_suffix="/hard-disk-2/users/aryavishe/data/vggnet_mirflickr_train"
    val_suffix="/hard-disk-2/users/aryavishe/data/vggnet_mirflickr_val"
else
    echo "Invalid dataset"
    exit 1
fi

echo "./a.out "${train_suffix}_features.bin" "${val_suffix}_features.bin" "${train_suffix}_labels.txt" "${val_suffix}_labels.txt" ${temp_dir} ${group_creation_algorithm} ${N_force}"
./a.out "${train_suffix}_features.bin" "${val_suffix}_features.bin" "${train_suffix}_labels.txt" "${val_suffix}_labels.txt" ${temp_dir} ${group_creation_algorithm} ${N_force}

source ~/.venv/bin/activate # This will just make sure the venv is active

echo "python3 plotter.py ${temp_dir} ${R}"
python3 plotter.py ${temp_dir} ${R}

# We also need to save the results for the sake of later analysis
current_time=$(date +"%Y%m%d_%H%M%S")
mv ${temp_dir}/results_${R}.csv history/results_${R}_${dataset}_${group_creation_algorithm}_${N_force}_${current_time}.csv
mv ${temp_dir}/metadata.json history/metadata_${R}_${dataset}_${group_creation_algorithm}_${N_force}_${current_time}.json