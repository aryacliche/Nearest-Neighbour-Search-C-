#!/bin/bash
mode=$1
dataset=$2
group_creation_algorithm=$3
R=${4:-3}
N_force=${5:--1}
temp_name=${6:-alt_temp}
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

echo "./alt.out "${train_suffix}_features.bin" "${val_suffix}_features.bin" "${train_suffix}_clustered_labels.txt" "${val_suffix}_labels.txt" ${temp_dir} ${group_creation_algorithm} ${N_force}"
./alt.out "${train_suffix}_features.bin" "${val_suffix}_features.bin" "${train_suffix}_labels.txt" "${val_suffix}_labels.txt" ${temp_dir} ${group_creation_algorithm} ${N_force} > ${temp_dir}/log.out

if [ "${mode}" == "visualise" ]; then
    echo "python3 ancillary_stuff/visualise_plots.py ${dataset} ${group_creation_algorithm}"
    python3 ancillary_stuff/visualise_plots.py ${dataset} ${group_creation_algorithm}
fi
