#!/bin/bash

./cluster.out /hard-disk-2/users/aryavishe/data/lda_vggnet_features_imdb_train_features.bin /hard-disk-2/users/aryavishe/data/lda_vggnet_features_imdb_val_features.bin
./cluster.out /hard-disk-2/users/aryavishe/data/lda_vggnet_outputs_imdb_train_features.bin /hard-disk-2/users/aryavishe/data/lda_vggnet_outputs_imdb_val_features.bin

python3 extract_partition_convert_dataset.py yes inshallah mirflickr
python3 extract_partition_convert_dataset.py yes inshallah instacities

./cluster.out /hard-disk-2/users/aryavishe/data/lda_vggnet_features_mirflickr_train_features.bin /hard-disk-2/users/aryavishe/data/lda_vggnet_features_mirflickr_val_features.bin
./cluster.out /hard-disk-2/users/aryavishe/data/lda_vggnet_features_instacities_train_features.bin /hard-disk-2/users/aryavishe/data/lda_vggnet_features_instacities_val_features.bin

./cluster.out /hard-disk-2/users/aryavishe/data/lda_vggnet_outputs_mirflickr_train_features.bin /hard-disk-2/users/aryavishe/data/lda_vggnet_outputs_mirflickr_val_features.bin
./cluster.out /hard-disk-2/users/aryavishe/data/lda_vggnet_outputs_instacities_train_features.bin /hard-disk-2/users/aryavishe/data/lda_vggnet_outputs_instacities_val_features.bin