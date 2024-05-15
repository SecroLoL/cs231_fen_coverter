import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os 
import sys 

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import logging

from torchvision import models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Subset
import numpy as np
from occupancy_detection.baseline_cnn import CNN_100
from occupancy_detection.resnet import ResNetClassifier

# Assuming CNN_100 is already defined as per your provided code
EVAL_PATH = "/Users/alexshan/Desktop/chesscog/data/occupancy/test"

# Transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load your test dataset
test_dataset = datasets.ImageFolder(root=EVAL_PATH, transform=transform)

# Create DataLoader for the test dataset
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model
model = CNN_100()

# Load the saved model state
model.load_state_dict(torch.load('chess_square_classifier.pth'))

# Set the model to evaluation mode
model.eval()

# Function to test the model
def test_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():  # No need to compute gradients for evaluation
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')

# Test the model
test_model(model, test_loader)
