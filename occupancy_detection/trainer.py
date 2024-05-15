import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os 
import sys 

# TODO: add this to PYTHONPATH so that this isn't an issue
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # sys path issues, this is a quick fix for now

import logging
from torchvision import models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from occupancy_detection.baseline_cnn import CNN_100
from occupancy_detection.resnet import ResNetClassifier
from typing import List, Mapping, Tuple, Any
from occupancy_detection.model_types import ModelType, load_model
from occupancy_detection.evaluate import evaluate_model
from occupancy_detection.utils import *

# Create and configure the logger
logger = logging.getLogger('chess_square_classifier')
logger.setLevel(logging.DEBUG)  # Set the logging level to DEBUG

# Create a file handler to log messages to a file
file_handler = logging.FileHandler('chess_square_classifier.log')
file_handler.setLevel(logging.DEBUG)  # Set the logging level for the file handler

# Create a console handler to log messages to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set the logging level for the console handler

# Create a formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def load_datasets(train_path: str, eval_path: str, batch_size: int = 32, train_size: int =  None, eval_size: int = None) -> Tuple[DataLoader, DataLoader]:
    
    """
    Generate DataLoader objects from train path and eval path. 

    Args:
        TODO

    Returns:
        TODO

    Raises:
        TODO
    """
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Convert datasets to ImageFolder types
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=eval_path, transform=transform)

    # Sample if required
    if train_size is not None:
        train_indices = np.random.choice(len(train_dataset), train_size, replace=False)
        train_dataset = Subset(train_dataset, train_indices)

    if eval_size is not None:
        test_indices = np.random.choice(len(test_dataset), eval_size, replace=False)
        test_dataset = Subset(test_dataset, test_indices)

    # Convert to dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train(num_epochs: int, model_type: ModelType, save_path: str, train_path: str, eval_path: str, batch_size: int = 32, lr: float = 0.001,
          train_size: int = None, eval_size: int = None) -> None:
    
    # For choosing models at each epoch
    best_acc = 0
    model_checkpoint_path = generate_checkpoint_path(save_path)
    print("using model checkpoint path", model_checkpoint_path)
    # Load datasets
    train_loader, test_loader = load_datasets(train_path, eval_path, batch_size, train_size, eval_size)
    
    # Init model
    model = load_model(model_type)  
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Train loop
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
        
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

        torch.save(model.state_dict(), model_checkpoint_path)
        acc = evaluate_model(model_type, model_checkpoint_path, test_loader)

        if acc > best_acc:
            torch.save(model.state_dict(), save_path)
            best_acc = acc
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}]: New best accuracy {acc}. Saved model checkpoint to {save_path}.")

    logger.info(f'Finished Training. Best val acc: {best_acc}.')
    
    if not os.path.exists(model_checkpoint_path):
        raise FileNotFoundError(f"Attempted to remove {model_checkpoint_path}, but could not find file.")
    os.remove(model_checkpoint_path)  # Delete checkpoint


def main():
    TRAIN_PATH = "/Users/alexshan/Desktop/chesscog/data/occupancy/train"
    EVAL_PATH = "/Users/alexshan/Desktop/chesscog/data/occupancy/val"
    SAVE_NAME = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models", "occupancy", "cnn_100.pth")

    NUM_EPOCHS = 10

    train(NUM_EPOCHS,
          ModelType.CNN_100,
          SAVE_NAME,
          TRAIN_PATH,
          EVAL_PATH,
          train_size=1500,
          eval_size=300)
    
if __name__ == "__main__":
    main()
