import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision import transforms
from PIL import Image
import sys
import os
import struct
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

log_file = "extract_partition_convert_dataset.log"

# List all the image files in the folder
# Function to process an image and return the output
def process_image(image_file, last_features, secondlast_features, last_feature_extractor, second_last_feature_extractor, label_name, label_list):
    try:
        # Load the image
        image = Image.open(image_file)

        # Apply the transformation
        input_tensor = transform(image).unsqueeze(0)

        # Forward pass to get the second-to-last layer output
        with torch.no_grad():
            last_output = last_feature_extractor(input_tensor)
            secondlast_output = second_last_feature_extractor(input_tensor)
        
        last_features.append(last_output.detach().numpy().flatten())
        secondlast_features.append(secondlast_output.detach().numpy().flatten())
        label_list.append(label_name)
        return 1
    except:
        print(f"Error processing image: {image_file}")
        return 0

# Function to recursively go through a folder and process images
def process_folder(root_path, last_feature_extractor, secondlast_feature_extractor, training_last_features, training_secondlast_features, val_last_features, val_secondlast_features, training_labels, val_labels, testing_ratio):
    # List all the files and folders in the current folder
    folders = os.listdir(root_path)

    for folder in folders:
        folder_path = os.path.join(root_path, folder)
        files = os.listdir(folder_path)
        N = len(files)
        
        val_files = [files[x] for x in np.random.choice(N, int(N*testing_ratio), replace=False)]
        train_files = list(set(files) - set(val_files))
        
        num_valid_train_files = 0
        num_valid_val_files = 0
        for file in val_files:
            file_path = os.path.join(folder_path, file)
            num_valid_val_files += process_image(file_path, val_last_features, val_secondlast_features, last_feature_extractor, secondlast_feature_extractor, folder, val_labels)
        for file in train_files:
            file_path = os.path.join(folder_path, file)
            num_valid_train_files += process_image(file_path, training_last_features, training_secondlast_features, last_feature_extractor, secondlast_feature_extractor, folder, training_labels)

        with open(log_file, "a") as log:
            log.write(f"{folder} : num_valid_train_files = {num_valid_train_files}, num_valid_val_files = {num_valid_val_files}\n")

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

def read_from_bin(features, phrase):
    # Read the binary feature file
    print(f"Reading from {phrase}_features.bin...")
    with open(f'{phrase}_features.bin', 'rb') as f:
        # Read header
        num_points, feature_dim = struct.unpack('ii', f.read(8))
        
        # Read feature data
        for _ in range(num_points):
            feature = struct.unpack('f' * feature_dim, f.read(4 * feature_dim))
            features.append(list(feature))

    print(f"Reading from {phrase}_labels.txt...")
    # Read the label file
    with open(f'{phrase}_labels.txt', 'r') as f:
        labels = [line.strip() for line in f]

    return labels

# Process the images in the folder
def main():
    only_verify = (sys.argv[1].lower() == 'yes')
    dataset = sys.argv[2]
    data_root = "/hard-disk-2/users/aryavishe/data"
    
    global log_file
    log_file = log_file + f"_{dataset}"
    with open(log_file, "w") as log:
        log.write("Starting the process...\n")
        log.write(f"Dataset: {dataset}\n")
        log.write(f"Only verify: {only_verify}\n")

    if not only_verify:
        root_path = sys.argv[3]
        testing_ratio = float(sys.argv[4])
        
        # Load the pretrained VGG16 model
        model = vgg16(pretrained=True)
        model.eval()
        last_feature_extractor = VGG16FeatureExtractor(model, 'last')
        secondlast_feature_extractor = VGG16FeatureExtractor(model, 'secondlast')

    if not only_verify:
        print("Converting images to feature space...")
        training_last_features = []
        val_last_features = []
        training_secondlast_features = []
        val_secondlast_features = []
        training_labels = []
        val_labels = []
        process_folder(root_path, last_feature_extractor, secondlast_feature_extractor, training_last_features, training_secondlast_features, val_last_features, val_secondlast_features, training_labels, val_labels, testing_ratio)
        
        print("Number of training samples: ", len(training_last_features))
        print("Number of validation samples: ", len(val_last_features))

        d_last = 1000
        d_secondlast = 4096

        print("Converting training data to binary format...")
        convert_to_bin(training_last_features, training_labels, d_last, f'{data_root}/vggnet_outputs_{dataset}_train')
        convert_to_bin(training_secondlast_features, training_labels, d_secondlast, f'{data_root}/vggnet_features_{dataset}_train')

        print("Converting validation data to binary format...")
        convert_to_bin(val_last_features, val_labels, d_last, f'{data_root}/vggnet_outputs_{dataset}_val')
        convert_to_bin(val_secondlast_features, val_labels, d_secondlast, f'{data_root}/vggnet_features_{dataset}_val')

        del training_last_features
        del training_secondlast_features
        del val_last_features
        del val_secondlast_features
        del training_labels
        del val_labels
    
    # Before we finish, we will quickly verify if things are going as wanted them to go
    training_last_features = []
    val_last_features = []
    training_secondlast_features = []
    val_secondlast_features = []
    training_labels = []
    val_labels = []
    training_labels = read_from_bin(training_last_features, f'{data_root}/vggnet_outputs_{dataset}_train')
    print(f"Number of labels = {len(training_labels)}, number of features = {len(training_last_features)}")
    del training_last_features
    read_from_bin(training_secondlast_features, f'{data_root}/vggnet_features_{dataset}_train')
    print(f"Number of labels = {len(training_labels)}, number of features = {len(training_secondlast_features)}")
    del training_secondlast_features
    val_labels = read_from_bin(val_last_features, f'{data_root}/vggnet_outputs_{dataset}_val')
    print(f"Number of labels = {len(val_labels)}, number of features = {len(val_last_features)}")
    del val_last_features
    val_labels = read_from_bin(val_secondlast_features, f'{data_root}/vggnet_features_{dataset}_val')
    print(f"Number of labels = {len(val_labels)}, number of features = {len(val_secondlast_features)}")
    del val_secondlast_features


if __name__=="__main__":
    main()
