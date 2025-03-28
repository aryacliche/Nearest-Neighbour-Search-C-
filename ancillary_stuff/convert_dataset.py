import torch
import sys
import struct
import os

def convert_features_to_bin(folder_path, output_name):
    # List all the files and folders in the current folder
    features = []
    labels = []
    categories = os.listdir(folder_path) # Each category is a folder

    for category in categories:
        category_path = os.path.join(folder_path, category)
        files = os.listdir(category_path)
        for file in files:
            file_path = os.path.join(category_path, file)
            feature = torch.load(file_path, weights_only = False)
            features.append(feature.detach().numpy().flatten())
            labels.append(category)
    
    num_points = len(features)
    feature_dim = 4096

    with open(f'{output_name}_features.bin', 'wb') as f:
        # Write header
        f.write(struct.pack('ii', num_points, feature_dim))
        
        # Write feature data
        for feature in features:
            f.write(struct.pack('f' * feature_dim, *feature))

    with open(f'{output_name}_labels.txt', 'w') as f:
        for label in labels:
            f.write(f'{label}\n')

def main():
    folder_path = sys.argv[1] # Should point to the folder containing the features
    output_name = sys.argv[2] # Name of the output file
    convert_features_to_bin(folder_path, output_name)

if __name__ == '__main__':
    main()