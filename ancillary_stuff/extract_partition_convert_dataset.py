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

# List all the image files in the folder
# Function to process an image and return the output
def process_image(image_file, last_features, secondlast_features, last_feature_extractor, second_last_feature_extractor):
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
            
    except:
        print(f"Error processing image: {image_file}")

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
        
        for file in val_files:
            file_path = os.path.join(folder_path, file)
            process_image(file_path, val_last_features, val_secondlast_features, last_feature_extractor, secondlast_feature_extractor)
        for file in train_files:
            file_path = os.path.join(folder_path, file)
            process_image(file_path, training_last_features, training_secondlast_features, last_feature_extractor, secondlast_feature_extractor)
        
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

def read_from_bin(features, labels, phrase):
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

    return features, labels

# Process the images in the folder
def main():
    read_bin = (sys.argv[1].lower() == 'yes')
    root_path = sys.argv[2]
    dataset = sys.argv[3]
    data_root = "/hard-disk-2/users/aryavishe/data"
    if read_bin:
        use_LDA = True
    else:
        testing_ratio = float(sys.argv[4])
        use_LDA = (sys.argv[5].lower() == 'yes')

        # Load the pretrained VGG16 model
        model = vgg16(pretrained=True)
        model.eval()
        last_feature_extractor = VGG16FeatureExtractor(model, 'last')
        secondlast_feature_extractor = VGG16FeatureExtractor(model, 'secondlast')

    training_last_features = []
    val_last_features = []
    training_secondlast_features = []
    val_secondlast_features = []
    training_labels = []
    val_labels = []
    print("Converting images to feature space...")
    if read_bin:
        read_from_bin(training_last_features, training_labels, f'{data_root}/vggnet_outputs_{dataset}_train')
        read_from_bin(training_secondlast_features, [], f'{data_root}/vggnet_features_{dataset}_train')
        read_from_bin(val_last_features, val_labels, f'{data_root}/vggnet_outputs_{dataset}_val')
        read_from_bin(val_secondlast_features, [], f'{data_root}/vggnet_features_{dataset}_val')
    else:
        process_folder(root_path, last_feature_extractor, secondlast_feature_extractor, training_last_features, training_secondlast_features, val_last_features, val_secondlast_features, training_labels, val_labels, testing_ratio)

    print("Number of training samples: ", len(training_last_features))
    print("Number of validation samples: ", len(val_last_features))

    d_last = 1000
    d_secondlast = 4096

    if use_LDA:
        lda = LinearDiscriminantAnalysis()
        training_last_features = lda.fit_transform(training_last_features, training_labels)
        val_last_features = lda.transform(val_last_features)
        d_last = np.shape(training_last_features)[1]

        training_secondlast_features = lda.fit_transform(training_secondlast_features, training_labels)
        val_secondlast_features = lda.transform(val_secondlast_features)
        d_secondlast = np.shape(training_secondlast_features)[1]

        print("Reduced feature dimensions using LDA:")
        print(f"Secondlast: {d_secondlast}")
        print(f"Last: {d_last}")

    print("Converting training data to binary format...")
    convert_to_bin(training_last_features, training_labels, d_last, f'{data_root}/lda_vggnet_outputs_{dataset}_train' if use_LDA else f'{data_root}/vggnet_outputs_{dataset}_train')
    convert_to_bin(training_secondlast_features, training_labels, d_secondlast, f'{data_root}/lda_vggnet_features_{dataset}_train' if use_LDA else f'{data_root}/vggnet_features_{dataset}_train')

    print("Converting validation data to binary format...")
    convert_to_bin(val_last_features, val_labels, d_last, f'{data_root}/lda_vggnet_outputs_{dataset}_val' if use_LDA else f'{data_root}/vggnet_outputs_{dataset}_val')
    convert_to_bin(val_secondlast_features, val_labels, d_secondlast, f'{data_root}/lda_vggnet_features_{dataset}_val' if use_LDA else f'{data_root}/vggnet_features_{dataset}_val')

if __name__=="__main__":
    main()
