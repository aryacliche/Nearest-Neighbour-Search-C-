python3 extract_partition_convert_dataset.py /hard-disk-2/users/aryavishe/MIRFLICKR/images 0.2 secondlast vggnet_features_mirflickr
python3 extract_partition_convert_dataset.py /hard-disk-2/users/aryavishe/MIRFLICKR/images 0.2 last vggnet_outputs_mirflickr
./cluster.out /hard-disk-2/users/aryavishe/data/vggnet_outputs_mirflickr_train_features.bin /hard-disk-2/users/aryavishe/data/vggnet_outputs_mirflickr_val_features.bin 20
./cluster.out /hard-disk-2/users/aryavishe/data/vggnet_features_mirflickr_train_features.bin /hard-disk-2/users/aryavishe/data/vggnet_features_mirflickr_val_features.bin 20
