import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision import transforms
from PIL import Image
import sys
import os
import struct
import numpy as np

# Create a new model that outputs the second-to-last layer
class VGG16FeatureExtractor(nn.Module):
    def __init__(self, original_model, mode):
        super(VGG16FeatureExtractor, self).__init__()
        self.features = original_model.features
        self.avgpool = original_model.avgpool
        if mode == 'last':
            self.classifier = nn.Sequential(*list(original_model.classifier.children()))  # Include all layers
        else:
            self.classifier = nn.Sequential(*list(original_model.classifier.children())[:-1])  # Exclude the last layer

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Example input tensor of shape [1, 3, 224, 224]
# Define the image transformation
transform = transforms.Compose([
    transforms.RandomCrop(224),        # Random crop to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# List all the image files in the folder
# Function to process an image and return the output
def process_image(image_file, features, feature_extractor):
    try:
            # Load the image
            image = Image.open(image_file)

            # Apply the transformation
            input_tensor = transform(image).unsqueeze(0)

            # Forward pass to get the second-to-last layer output
            with torch.no_grad():
                output = feature_extractor(input_tensor)
            
            features.append(output.detach().numpy().flatten())
            
    except:
        print(f"Error processing image: {image_file}")

# Function to recursively go through a folder and process images
def process_folder(root_path, feature_extractor, training_features, val_features, training_labels, val_labels, testing_ratio):
    # List all the files and folders in the current folder
    folders = os.listdir(root_path)

    for folder in folders:
        folder_path = os.path.join(root_path, folder)
        files = os.listdir(folder_path)
        N = len(files)
        
        val_files = [files[x] for x in np.random.choice(N, int(N*testing_ratio), replace=False)]
        train_files = list(set(files) - set(val_files))
        
        for file in val_files:
            file_path = os.path.join(folder_path, file)
            process_image(file_path, val_features, feature_extractor)
        for file in train_files:
            file_path = os.path.join(folder_path, file)
            process_image(file_path, training_features, feature_extractor)
        
        val_labels.extend([folder] * len(val_files))
        training_labels.extend([folder] * len(train_files))

def convert_to_bin(features, labels, feature_dim, output_name):
    num_points = len(features)

    with open(f'{output_name}_features.bin', 'wb') as f:
        # Write header
        f.write(struct.pack('ii', num_points, feature_dim))
        
        # Write feature data
        for feature in features:
            f.write(struct.pack('f' * feature_dim, *feature))

    with open(f'{output_name}_labels.txt', 'w') as f:
        for label in labels:
            f.write(f'{label}\n')

# Process the images in the folder
def main():
    root_path = sys.argv[1]
    testing_ratio = float(sys.argv[2])
    mode = sys.argv[3]
    if mode == 'last':
        feature_dim = 1000
    elif mode == 'secondlast':
        feature_dim = 4096
    else:
        print("Invalid mode. Please use 'last' or 'secondlast'.")
        return
    phrase = sys.argv[4]

    data_root = "/hard-disk-2/users/aryavishe/data"

    if mode not in ['last', 'secondlast']:
        print("Invalid mode. Please use 'last' or 'secondlast'.")
        return

    # Load the pretrained VGG16 model
    model = vgg16(pretrained=True)
    model.eval()
    feature_extractor = VGG16FeatureExtractor(model, mode)

    training_features = []
    val_features = []
    training_labels = []
    val_labels = []
    print("Converting images to feature space...")
    process_folder(root_path, feature_extractor, training_features, val_features, training_labels, val_labels, testing_ratio)

    print("Number of training samples: ", len(training_features))
    print("Number of validation samples: ", len(val_features))

    print("Converting training data to binary format...")
    convert_to_bin(training_features, training_labels, feature_dim, f'{data_root}/{phrase}_train')

    print("Converting validation data to binary format...")
    convert_to_bin(val_features, val_labels, feature_dim, f'{data_root}/{phrase}_val')

if __name__=="__main__":
    main()
