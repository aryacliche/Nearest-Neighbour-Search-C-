#!/bin/bash
mode=$1
dataset=$2
layer=$3
group_creation_algorithm=$4
R=${5:-3}
N_force=${6:--1}
temp_name=${7:-temp}
temp_dir="${temp_name}/${layer}/${dataset}/${group_creation_algorithm}"

if [ ! -d "${temp_dir}" ]; then
    mkdir -p "${temp_dir}"
fi

if [ "${dataset}" == "mirflickr" ]; then
    data_prefix="/hard-disk-2/users/aryavishe/data"
elif [ "${dataset}" == "imagenet" ]; then
    data_prefix="/home/aryavishe/Nearest-Neighbour-Search-C-/data"
else
    echo "Invalid dataset specified. Use 'mirflickr', 'nuswide', or 'coco'."
    exit 1
fi

if [ "${layer}" == "last" ]; then
    layer_prefix="vggnet_outputs"
elif [ "${layer}" == "secondlast" ]; then
    layer_prefix="vggnet_features"
else   
    echo "Invalid layer specified. Use 'last' or 'secondlast'."
    exit 1
fi

train_prefix="${data_prefix}/${layer_prefix}_${dataset}_train"
val_prefix="${data_prefix}/${layer_prefix}_${dataset}_val"

if [ "${group_creation_algorithm}" == "clustered" ]; then
    echo "./a.out "${train_prefix}_features.bin" "${val_prefix}_features.bin" "${train_prefix}_clustered_labels.txt" "${val_prefix}_labels.txt" ${temp_dir} labelled ${N_force}"
    ./a.out "${train_prefix}_features.bin" "${val_prefix}_features.bin" "${train_prefix}_clustered_labels.txt" "${val_prefix}_labels.txt" ${temp_dir} labelled ${N_force} > ${temp_dir}/log.out
else
    echo "./a.out "${train_prefix}_features.bin" "${val_prefix}_features.bin" "${train_prefix}_labels.txt" "${val_prefix}_labels.txt" ${temp_dir} ${group_creation_algorithm} ${N_force}"
    ./a.out "${train_prefix}_features.bin" "${val_prefix}_features.bin" "${train_prefix}_labels.txt" "${val_prefix}_labels.txt" ${temp_dir} ${group_creation_algorithm} ${N_force} > ${temp_dir}/log.out
fi

if [ "${mode}" == "visualise" ]; then
    echo "python3 ancillary_stuff/visualise_plots.py ${dataset} ${group_creation_algorithm} ${layer}"
    python3 ancillary_stuff/visualise_plots.py ${dataset} ${group_creation_algorithm} ${layer}
fi
