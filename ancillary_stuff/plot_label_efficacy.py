import pandas as pd #type: ignore
import numpy as np
import matplotlib.pyplot as plt #type: ignore
import sys, os, torch

def get_features(folder_path, features, original_categories, num_features = -1):
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
            
def get_numeric_labels(labels):
    unique_labels = list(set(labels))
    feature_map = {feature: idx for idx, feature in enumerate(unique_labels)}
    numeric_labels = [feature_map[feature] for feature in labels]
    return numeric_labels
    

def main():
    dataset = sys.argv[1]
    if dataset == "mirflickr":
        folder_name = "/hard-disk-2/users/aryavishe/VGGNET_features_MIRFLICKR_partitioned"
    elif dataset == "imagenet":
        folder_name = "/home/aryavishe/Nearest-Neighbour-Search/VGGNET_features_IMAGENET"
    else:
        print("Invalid dataset")
        exit(1)
    
    distances_filepath = sys.argv[2]

    print("Reading the training repo")
    train_labels = []
    get_features(f"{folder_name}/train", [], train_labels)
    train_labels = get_numeric_labels(train_labels)

    print("Reading the validation repo")
    val_labels = []
    get_features(f"{folder_name}/val", [], val_labels)
    val_labels = get_numeric_labels(val_labels)

    print("Reading the distances csv")
    distances_df = pd.read_csv(distances_filepath)

    unique_queries = distances_df['Index1'].unique()

    for query_id in unique_queries:
        original_label = val_labels[query_id]
        query_distances = distances_df[distances_df['Index1'] == query_id]
        labels = [train_labels[x] for x in query_distances['Index2']]
        distances = query_distances['Distance']
        sorted_indices = np.argsort(labels)
        labels = np.array(labels)[sorted_indices]
        distances = np.array(distances)[sorted_indices]
        
        plt.scatter(np.arange(len(query_distances)), distances, c = labels, cmap = plt.get_cmap('tab20', 100))
        plt.xlabel("Training image index")
        plt.ylabel("Distance")
        plt.title(f"Distances from query {query_id} (label = {original_label}) to all training images")
        plt.savefig(f"label_efficacy_graphs/distances_{dataset}_{query_id}.png")
        plt.close()


if __name__=="__main__":
    main()
