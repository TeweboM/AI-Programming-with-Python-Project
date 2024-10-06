# Imports libraries
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import os
import json
from PIL import Image
import numpy as np
import torch.nn.functional as F
import argparse
import train

# image_dataset
# model_save_path
# image_path
# 
def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_image', type=str, help='Path to image.')
    parser.add_argument('--model', type=str, default= 'flower_classification_model.pth', help='Trained model.')
    parser.add_argument('--top_k', type=int, default=3, help='Likely flower names.')
    parser.add_argument('--device', type=str, default='gpu', help='gpu or cpu')
    return parser.parse_args()

args = get_input_args()

if args:
    device = args.device
else:
    device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved state dictionary into the model
model_save_path = args.model

model = models.vgg11(pretrained=True)  # Initialize the model structure
num_classes = len(train.image_datasets['train'].classes)

# Load the model's state dictionary
path_to_image = args.path_to_image

# Replace the classifier with the correct number of output classes
model.classifier[6] = nn.Linear(4096, num_classes)

model.load_state_dict(torch.load(model_save_path))

# Move the model to the correct device (CPU or GPU)
model = model.to(device)


def process_image(image_path):
    """Load an image from file and preprocess it for model input."""
    # Load the image with PIL
    image = Image.open(image_path)
    
    # Define the necessary transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),                   # Resize so the shortest side is 256 pixels
        transforms.CenterCrop(224),               # Crop the center 224x224 pixels
        transforms.ToTensor(),                    # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean/std
    ])
    
    # Apply the transformations to the image
    image_tensor = preprocess(image)
    
    return image_tensor  # This will be a tensor



topk = args.top_k
def predict(path_to_image, model, topk=topk):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    
    Parameters:
    - image_path (str): Path to the input image.
    - model (torch.nn.Module): Pre-trained PyTorch model.
    - topk (int): Number of top probabilities to return (default is 5).
    
    Returns:
    - probs (list): List of topk probabilities.
    - classes (list): List of topk predicted class labels.
    """
    # Process the image
    image = process_image(path_to_image)  # Use the function from earlier
    
    # Add a batch dimension (1, C, H, W) for PyTorch
    image = image.unsqueeze(0)
    
    # Move the model to the appropriate device (if using GPU)
    model = model.to(device)
    image = image.to(device)
    
    # Set the model to evaluation mode and turn off gradients
    model.eval()
    with torch.no_grad():
        # Forward pass: Get the model output
        output = model(image)
        
        # Apply softmax to get probabilities
        probs = F.softmax(output, dim=1)
        
        # Get the top-K probabilities and indices
        top_probs, top_idxs = probs.topk(topk, dim=1)
        
        # Convert to lists
        top_probs = top_probs.cpu().numpy().squeeze().tolist()
        top_idxs = top_idxs.cpu().numpy().squeeze().tolist()
    
    # Convert the indices back to the actual class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_idxs]
    
    return top_probs, top_classes

prediction = predict(path_to_image, model, topk)
for (a, b) in prediction:
    print(f"Prediction class : {b}, Predicted probability : {a}")
    

if __name__ == "__main__":
    main()