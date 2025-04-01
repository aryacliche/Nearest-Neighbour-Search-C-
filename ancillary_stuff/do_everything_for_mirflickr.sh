python3 extract_partition_convert_dataset.py /hard-disk-2/users/aryavishe/IMDB-Wiki/imdb/ 0.2 secondlast vggnet_features_imdb 
python3 extract_partition_convert_dataset.py /hard-disk-2/users/aryavishe/IMDB-Wiki/imdb 0.2 last vggnet_outputs_imdb
./cluster.out /hard-disk-2/users/aryavishe/data/vggnet_outputs_imdb_train_features.bin /hard-disk-2/users/aryavishe/data/vggnet_outputs_imdb_val_features.bin 1000
./cluster.out /hard-disk-2/users/aryavishe/data/vggnet_features_imdb_train_features.bin /hard-disk-2/users/aryavishe/data/vggnet_features_imdb_val_features.bin 4096
