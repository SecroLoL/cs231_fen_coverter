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
from piece_classifier.cnn import CNN_100
from piece_classifier.resnet import ResNetClassifier
from typing import List, Mapping, Tuple, Any
from piece_classifier.model_types import ModelType, load_model

# Create and configure the logger
logger = logging.getLogger('chess_piece_classifier_eval')
logger.setLevel(logging.DEBUG)  # Set the logging level to DEBUG

# Create a file handler to log messages to a file
file_handler = logging.FileHandler('chess_piece_classifier_eval.log')
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


def evaluate_model(model_type: ModelType, model_save_path: str, test_loader: DataLoader):
    """
    Computes accuracy and F1 scores over the test set using a saved model.
    """
    logger.info(f"Attempting to evaluate model {model_type}, path: {model_save_path}")
    
    # Model setup
    model = torch.load(model_save_path)
    model.eval()
    
    correct = 0
    total = 0
    class_tp = {}
    class_fp = {}
    class_fn = {}
    
    # Initialize counters for each class
    for _, labels in test_loader:
        num_classes = 12
        for class_index in range(num_classes):
            class_tp[class_index] = 0
            class_fp[class_index] = 0
            class_fn[class_index] = 0
        break
    
    with torch.no_grad():  
        for inputs, labels in test_loader:

            outputs = model(inputs)
            
            _, predicted = torch.max(outputs.data, 1)
            
            if predicted.shape != labels.shape:
                raise ValueError(f"Shape mismatch: labels have shape {labels.shape} while predictions have shape {predicted.shape}")

            for i in range(len(labels)):
                true_class = labels[i].item()
                pred_class = predicted[i].item()
                
                if true_class == pred_class:
                    class_tp[true_class] += 1
                else:
                    class_fp[pred_class] += 1
                    class_fn[true_class] += 1
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total

    precision = {class_index: class_tp[class_index] / (class_tp[class_index] + class_fp[class_index]) if (class_tp[class_index] + class_fp[class_index]) > 0 else 0 for class_index in class_tp}
    recall = {class_index: class_tp[class_index] / (class_tp[class_index] + class_fn[class_index]) if (class_tp[class_index] + class_fn[class_index]) > 0 else 0 for class_index in class_tp}
    f1 = {class_index: 2 * precision[class_index] * recall[class_index] / (precision[class_index] + recall[class_index]) if (precision[class_index] + recall[class_index]) > 0 else 0 for class_index in class_tp}

    macro_f1 = np.mean(list(f1.values()))
    weighted_f1 = sum((class_tp[class_index] + class_fn[class_index]) * f1[class_index] for class_index in class_tp) / total
    
    logger.info(f'Accuracy of the model on the test images: {accuracy:.2f}%')
    logger.info(f'Macro F1: {macro_f1}. Weighted F1: {weighted_f1}')
    
    return accuracy, macro_f1, weighted_f1

# TODO make a version of the eval function that evaluates using a save path instead of the dataloader object
# This will enable a CLI version to call this function, instead of current internal code use



def main():
    # evaluate_model()
    pass 


if __name__ == "__main__":
    pass
