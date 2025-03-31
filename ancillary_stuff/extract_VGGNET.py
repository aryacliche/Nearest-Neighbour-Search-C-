import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision import transforms
from PIL import Image
import sys
import os

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
def process_image(image_file, input_dir, output_dir, feature_extractor):
    try:
            # Load the image
            image = Image.open(image_file)

            # Apply the transformation
            input_tensor = transform(image).unsqueeze(0)

            # Forward pass to get the second-to-last layer output
            with torch.no_grad():
                output = feature_extractor(input_tensor)
            
            save_destination = image_file.replace(input_dir, output_dir).replace('.jpg', '_features.pt')    # For Imagenet : JPEG
            torch.save(output, save_destination)
    except:
        print(f"Error processing image: {image_file}")

# Function to recursively go through a folder and process images
def process_folder(folder_path, input_dir, output_dir, feature_extractor):
    # List all the files and folders in the current folder
    files = os.listdir(folder_path)

    for file in files:
        # Had to bring in this line because of the computer failing midway once
        if file in ['chicago', 'london', 'melbourne', 'newyork', 'sanfrancisco', 'sydney']:
            continue
        file_path = os.path.join(folder_path, file)
        if os.path.isdir(file_path):
            print(f"In {file} right now!")
            mkdir_path = file_path.replace(input_dir, output_dir)
            os.makedirs(mkdir_path, exist_ok=True)
            # Recursively process the subfolder
            process_folder(file_path, input_dir, output_dir, feature_extractor)
        elif file.endswith('.jpg'):                                                                         # For Imagenet : JPEG
            # Process the image and store the output
            process_image(file_path, input_dir, output_dir, feature_extractor)

# Process the images in the folder
def main():
    folder_path = sys.argv[1]
    input_dir = folder_path
    output_dir = sys.argv[2]
    mode = sys.argv[3]

    if mode not in ['last', 'secondlast']:
        print("Invalid mode. Please use 'last' or 'secondlast'.")
        return

    # Load the pretrained VGG16 model
    model = vgg16(pretrained=True)
    model.eval()
    feature_extractor = VGG16FeatureExtractor(model, mode)
    process_folder(folder_path, input_dir, output_dir, feature_extractor)

if __name__=="__main__":
    main()
