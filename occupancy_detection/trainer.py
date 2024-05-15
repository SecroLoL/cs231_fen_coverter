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

TRAIN_PATH = "/Users/alexshan/Desktop/chesscog/data/occupancy/train"
EVAL_PATH = "/Users/alexshan/Desktop/chesscog/data/occupancy/val"

# Assuming CNN_100 is already defined as per your provided code

# Transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor()
])

num_train_examples = 1000  # Adjust this number as needed
num_test_examples = 200    # Adjust this number as needed

train_dataset = datasets.ImageFolder(root=TRAIN_PATH, transform=transform)
test_dataset = datasets.ImageFolder(root=EVAL_PATH, transform=transform)

print(f"Training dataset size: {len(train_dataset)}. Test set {len(test_dataset)}")

train_indices = np.random.choice(len(train_dataset), num_train_examples, replace=False)
test_indices = np.random.choice(len(test_dataset), num_test_examples, replace=False)

train_subset = Subset(train_dataset, train_indices)
test_subset = Subset(test_dataset, test_indices)

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

print("Loaded datasets")

model = CNN_100()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10  

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Print statistics
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

print('Finished Training')

# Save the trained model
torch.save(model.state_dict(), 'chess_square_classifier.pth')




