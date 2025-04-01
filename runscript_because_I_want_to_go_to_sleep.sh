#!/bin/bash

python3 automation.py nah secondlast imagenet
python3 automation.py nah last imagenet
g++ FLINNG.cpp -o a.out -O3 -DVISUALISE -fopenmp
python3 automation.py visualise secondlast instacities
