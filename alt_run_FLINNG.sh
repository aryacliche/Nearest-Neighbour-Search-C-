#!/bin/bash

dataset=$1
group_creation_algorithm=$2
R=${3:-3}
N_force=${4:--1}
temp_name=${5:-alt_temp}
temp_dir="${temp_name}/${dataset}/${group_creation_algorithm}"

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
    echo ${dataset}
    echo "Invalid dataset"
    exit 1
fi

echo "./alt.out "${train_suffix}_features.bin" "${val_suffix}_features.bin" "${train_suffix}_labels.txt" "${val_suffix}_labels.txt" ${temp_dir} ${group_creation_algorithm} ${N_force}"
./alt.out "${train_suffix}_features.bin" "${val_suffix}_features.bin" "${train_suffix}_labels.txt" "${val_suffix}_labels.txt" ${temp_dir} ${group_creation_algorithm} ${N_force} > ${temp_dir}/log.out

echo "python3 ancillary_stuff/visualise_plots.py ${dataset} ${group_creation_algorithm}"
python3 ancillary_stuff/visualise_plots.py ${dataset} ${group_creation_algorithm}