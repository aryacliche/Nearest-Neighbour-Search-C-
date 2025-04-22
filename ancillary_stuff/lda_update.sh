#!/bin/bash

cd /hard-disk-2/users/aryavishe/IMDB-Wiki
echo "I am in $PWD"
echo "I can see..."
ls
cd /home/aryavishe/Nearest-Neighbour-Search-C-/ancillary_stuff
echo "I am in $PWD"

python3 extract_partition_convert_dataset.py no mirflickr /hard-disk-2/users/aryavishe/MIRFLICKR/images 0.2
python3 extract_partition_convert_dataset.py no instacities /hard-disk-2/users/aryavishe/InstaCities1M/img_resized_1M/cities_instagram 0.2
#cd /hard-disk-2/users/aryavishe/
#rm -rf MIRFLICKR
#cd IMDB-Wiki
#./untar.sh 
#python3 extract_partition_convert_dataset.py yes /hard-disk-2/users/aryavishe/IMDB_Wiki/imdb imdb 0.2 no
#
./cluster.out /hard-disk-2/users/aryavishe/data/vggnet_features_mirflickr_train_features.bin /hard-disk-2/users/aryavishe/data/vggnet_features_mirflickr_val_features.bin 20
./cluster.out /hard-disk-2/users/aryavishe/data/vggnet_features_instacities_train_features.bin /hard-disk-2/users/aryavishe/data/vggnet_features_instacities_val_features.bin 20

./cluster.out /hard-disk-2/users/aryavishe/data/vggnet_outputs_mirflickr_train_features.bin /hard-disk-2/users/aryavishe/data/vggnet_outputs_mirflickr_val_features.bin 20
./cluster.out /hard-disk-2/users/aryavishe/data/vggnet_outputs_instacities_train_features.bin /hard-disk-2/users/aryavishe/data/vggnet_outputs_instacities_val_features.bin 20
