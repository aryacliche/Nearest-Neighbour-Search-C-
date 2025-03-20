import pandas as pd #type: ignore
import sys
import os
import torch
import numpy as np

def get_features_imagenette(folder_path, features, original_categories, num_features = -1):
    # List all the files and folders in the current folder
    categories = os.listdir(folder_path) # Each category is a folder

    for category in categories:
        category_path = os.path.join(folder_path, category)
        files = os.listdir(category_path)
        for file in files:
            file_path = os.path.join(category_path, file)
            feature = torch.load(file_path, weights_only = False)
            features.append(feature.detach().numpy().flatten())
            original_categories.append(category)
            if num_features != -1 and len(features) == num_features:
                return

def main():
    indices_file_name = sys.argv[1]
    train_dataset_name = sys.argv[2]
    val_dataset_name = sys.argv[3]
    K = int(sys.argv[4])

    print("Reading training data")
    train_features = []
    get_features_imagenette(train_dataset_name, train_features, [], -1)
    
    print("Reading validation data")
    val_features = []
    get_features_imagenette(val_dataset_name, val_features, [], -1)
    
    indices = pd.read_csv(indices_file_name)

    for row_num, index in indices.iterrows():
        query = val_features[int(index['query_index'])]
        distances = np.linalg.norm(train_features - query, axis=1)
        closest_indices = np.argsort(distances)[:K]
        
        naive_neighbours_reported = [int(x) for x in index.tolist()[1:K + 1]] # The first element is the query index

        if (set(closest_indices) == set(naive_neighbours_reported)):
            print("Perfect matches for ", index['query_index'])
        else:
            print("Mismatch for query ", index['query_index'])
            print("Number of mismatches = ", K - len(set(closest_indices).intersection(set(naive_neighbours_reported))))

if __name__ == "__main__":
    main()