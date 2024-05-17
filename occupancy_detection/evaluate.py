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
from typing import List, Mapping, Tuple, Any
from occupancy_detection.model_types import ModelType, load_model

# Create and configure the logger
logger = logging.getLogger('chess_square_classifier_eval')
logger.setLevel(logging.DEBUG)  # Set the logging level to DEBUG

# Create a file handler to log messages to a file
file_handler = logging.FileHandler('chess_square_classifier_eval.log')
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


def evaluate_model(model_type: ModelType, model_save_path: str, test_loader: DataLoader) -> float:

    """
    Computes accuracy over the test set using a saved model
    """
    
    logger.info(f"Attempting to evaluate model {model_type}, path: {model_save_path}")
    model = load_model(model_type)  
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():  # No need to compute gradients for evaluation
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    logger.info(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')
    return accuracy

# TODO make a version of the eval function that evaluates using a save path instead of the dataloader object
# This will enable a CLI version to call this function, instead of current internal code use



def main():
    # evaluate_model()
    pass 


if __name__ == "__main__":
    pass
