# Image classifier
import torch
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import argparse
from PIL import Image
import numpy as np
# Download the dataset
import requests
import tarfile
import os
import shutil


def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Working directory.')
    parser.add_argument('--save_dir', type=str, default='flower_classification_model.pth', help='save directory.')
    parser.add_argument('--arch', type=str, default='vgg11', help='model architecture.')
    parser.add_argument('--gpu', type=str, default='gpu', help='GPU or CPU.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate.')
    parser.add_argument('--hidden_units', type=int, default=512, help='hidden unit.')
    parser.add_argument('--epochs', type=int, default=20, help='epochs.')
    return parser.parse_args()

def check_command_line_arguments(in_arg):
    if in_arg is None:
        print("""* Doesn't Check the Command Line Arguments because 
        'get_input_args' hasn't been defined.""")
    else:
        # prints command line agrs
        print(f"""Command Line Arguments:\n     data_dir =, {in_arg.data_dir}, 
              \n     save_dir =, {in_arg.save_dir}, \n     arch = , {in_arg.arch}, 
              \n     learning rate = {in_arg.learning_rate}, \n     hidden units = {in_arg.hidden_units}
              \n     epochs = {in_arg.epochs}""")

# Define get_input_args function
in_arg = get_input_args()

# Function that checks command line arguments using in_arg  
check_command_line_arguments(in_arg)

# Define the URL and folder paths
url = "https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz"
folder_name = "flowers"
#data_dir = 'flowers'
file_name = "flower_data.tar.gz"
file_path = os.path.join(data_dir, file_name)

# Remove the folder or symbolic link if it already exists (equivalent to `rm -rf flowers`)
try:
    if os.path.islink(folder_name) or os.path.isfile(folder_name):
        os.remove(folder_name)  # Remove the symbolic link or file
    elif os.path.isdir(folder_name):
        shutil.rmtree(folder_name)  # Remove the directory
    print(f"Removed existing {folder_name} folder/file/soft link, if any.")
except FileNotFoundError:
    pass  # If the file or directory does not exist, do nothing

# Create the data directory 
if in_arg.data_dir:
    data_dir = in_arg.data_dir
else:
    path = getcwd()
    folder = os.makedirs(folder_name)
    data_dir = os.path.join(path, folder)
    

# Select device
if in_arg.gpu:
    device = 'gpu'
else:
    device = 'cpu'
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    


# Download the file
response = requests.get(url, stream=True)

# Save the file in the 'flowers' folder
with open(file_path, "wb") as file:
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            file.write(chunk)

print(f"Downloaded {file_name} to {folder_name}")

# Extract the file in the 'flowers' folder
if file_path.endswith("tar.gz"):
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=folder_name)
        print(f"Extracted {file_name} to {folder_name}")

# Clean up by removing the tar.gz file after extraction
os.remove(file_path)
print(f"Removed the downloaded tar.gz file: {file_path}")


# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomResizedCrop(size=(224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    "valid": transforms.Compose([
        transforms.Resize(256),  # Resize the shorter side to 256 pixels
        transforms.CenterCrop(224),  # Center crop to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}



model.class_to_idx = image_datasets['train'].class_to_idx

# TODO: Load the datasets with ImageFolder
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']} 

#image_datasets = map_labels(image_datasets, cat_to_name)

dataset_size = {x: len(image_datasets[x]) for x in ['train', 'valid']}

train_loader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True)
valid_loader = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32, shuffle=False)


# Load the pre-trained VGG11 model
arch = in_arg.arch
model = models.arch(pretrained=True)

# Freeze all the layers except for the classifier
for param in model.parameters():
    param.requires_grad = False

# Replace the classifier (fully connected layers) with a new one
model.classifier[6] = nn.Linear(4096, len(image_datasets['train'].classes))  # Assuming the number of classes matches the dataset

# Move the model to the GPU if available
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
lr = in_arg.learning_rate
optimizer = optim.SGD(model.classifier.parameters(), lr=lr, momentum=0.9)  # Only optimizing classifier parameters

# Training and validation data loaders
dataloaders = {
    "train": train_loader,
    "valid": valid_loader
}

# Dataset sizes
dataset_sizes = {
    "train": len(image_datasets['train']),
    "valid": len(image_datasets['valid'])
}

# Number of epochs
num_epochs = in_arg.epochs

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    print('-' * 10)
    
    # Each epoch has a training and validation phase
    for phase in ['train', 'valid']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Backward pass + optimize only in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # Calculate epoch loss and accuracy
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

print('Training complete')


# Assuming test dataset is in '/path/to/test_data' and similar transforms
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the test dataset
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_transforms)
test_dataset.classes = [cat_to_name[name] for name in test_dataset.classes]
# Create a DataLoader for the test dataset
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Calculate test dataset size
test_dataset_size = len(test_dataset)

# Function to validate the model on test data
def validate_model(model, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    running_corrects = 0

    # Disable gradient calculation
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    # Calculate loss and accuracy for the test dataset
    test_loss = running_loss / test_dataset_size
    test_acc = running_corrects.double() / test_dataset_size

    print(f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

# Example usage
criterion = nn.CrossEntropyLoss()  # Using the same loss function as training
validate_model(model, test_loader, criterion)


# Save the model
# Define the path where the model will be saved
if in_arg.save_dir:
    model_save_path = in_arg.save_dir
else:
    model_save_path = 'flower_classification_model.pth'

# Save the model's state dictionary
torch.save(model.state_dict(), model_save_path)

print(f"Model saved to {model_save_path}")

# Call to main function to run the program
if __name__ == "__main__":
    main()