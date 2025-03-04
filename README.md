# Nearest-Neighbour-Search-C-
This repository deals with converting the code written in Python for my BTP - II project.

Towards FLINGG development:
1. A python script is needed which will pre-process a given dataset and then write it into a C++ friendly format : `convert_dataset.py`
2. We will first go through all the steps for IMAGENET where we are going to simply read the training and validation data and then time the naive implementation of nearest neighbour search : `naive-search.cpp`
Note that we have used a cosine-based distance measure. Projections on random hyperplanes results in an LSH function with probability of collision = 1 - Theta(x,y)/pi where Theta is the angle between them. Since cos Theta has the same monotonicity as this probability, we can state that the top-100 neighbours returned by the naive algorithm with cosine distances serve as golden reference.
3. With FLINGG, we will first write the code for making masks with the random group formation algorithm. Since we are dealing with really large datasets with not-so-big values of `L`, we should opt for a normal bit-map.

## FLINGG.cpp
This is going to be initially modelled the way we wrote the FLINGG python code. I am not sure what can be changed about it. Note that we are going allow Python to do all of the pre-processing of the dataset (since that is much easier anyways.) 
- We will ensure that the things that will be timed (i.e. creation of structures and querying) is implemented purely in C++.
