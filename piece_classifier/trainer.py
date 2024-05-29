import argparse
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
from tqdm import tqdm
import numpy as np
from piece_classifier.cnn import CNN_100
from piece_classifier.resnet import ResNetClassifier
from typing import List, Mapping, Tuple, Any
from piece_classifier.model_types import ModelType, load_model, ARGPARSE_TO_TYPE
from piece_classifier.evaluate import evaluate_model
from occupancy_detection.utils import *

# Create and configure the logger
logger = logging.getLogger('chess_piece_classifier')
logger.setLevel(logging.DEBUG)  # Set the logging level to DEBUG

# Create a file handler to log messages to a file
file_handler = logging.FileHandler('chess_piece_classifier.log')
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


def load_datasets(train_path: str, eval_path: str, batch_size: int = 32, train_size: int =  None, eval_size: int = None,
                  model_type: ModelType = None) -> Tuple[DataLoader, DataLoader]:
    
    """
    Generate DataLoader objects from train path and eval path. 

    Args:
        TODO

    Returns:
        TODO

    Raises:
        TODO
    """

    if model_type is None:
        raise ValueError(f"ModelType {model_type} is invalid.")
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if model_type.value == ModelType.INCEPTION.value:
        transform = transforms.Compose([
        transforms.Resize((299, 299)),  
        transforms.ToTensor(),
        ])  # InceptionV3 requires images to be size 299 x 299
    
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
    device = default_device()
    # For choosing models at each epoch
    best_f1 = 0
    model_checkpoint_path = generate_checkpoint_path(save_path)
    # Load datasets
    train_loader, test_loader = load_datasets(train_path, eval_path, batch_size, train_size, eval_size, model_type)
    
    # Init model
    logger.info(f"Loading {model_type}")
    model = load_model(model_type).to(device)  
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Train loop
    for epoch in tqdm(range(num_epochs), desc="Beginning new epoch..."):
        model.train()
        running_loss = 0.0

        logger.info(f"Training? {model.training}")
        
        for inputs, labels in tqdm(train_loader, desc="Training batches..."):
            # Zero the parameter gradients
            optimizer.zero_grad()

            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            if model_type.value == ModelType.INCEPTION.value:   # the InceptionV3 model has two loss variants that need to be combined
                primary_outputs, aux_outputs = model(inputs)
                primary_loss = criterion(primary_outputs, labels)
                aux_loss = criterion(aux_outputs, labels)

                loss = primary_loss + 0.4 * aux_loss   # https://arxiv.org/abs/1512.00567 gives reason for choosing 0.4 for the aux weight
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
        
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

        torch.save(model, model_checkpoint_path)
        acc, macro_f1, weighted_f1 = evaluate_model(model_type, model_checkpoint_path, test_loader)

        if weighted_f1 > best_f1:
            torch.save(model, save_path)
            best_f1 = weighted_f1
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}]: New best weighted F1: {weighted_f1}. Saved model checkpoint to {save_path}.")

    logger.info(f'Finished Training. Best weighted F1 on the val set: {best_f1}.')
    
    if not os.path.exists(model_checkpoint_path):
        raise FileNotFoundError(f"Attempted to remove {model_checkpoint_path}, but could not find file.")
    os.remove(model_checkpoint_path)  # Delete checkpoint


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default="/Users/alexshan/Desktop/chesscog/data/pieces/train", help="Path to training data")
    parser.add_argument("--eval_path", type=str, default="/Users/alexshan/Desktop/chesscog/data/pieces/val", help="Path to dev set")
    parser.add_argument("--save_path", type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models", "pieces", "cnn.pt"),
                        help="Path to model save file")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--model_type", type=str, default="cnn", help="Model architecture: ['cnn', 'resnet', ...]")
    parser.add_argument("--train_size", type=int, default=None, help="Size of training subset if needed")
    parser.add_argument("--eval_size", type=int, default=None, help="Size of val subset if needed")
    parser.add_argument("--batch_size", type=int, default=32, help="Sizes of each batch used in training")

    args = parser.parse_args()

    NUM_EPOCHS = args.num_epochs
    MODEL_TYPE = ARGPARSE_TO_TYPE.get(args.model_type)
    SAVE_NAME = args.save_path
    TRAIN_PATH = args.train_path
    EVAL_PATH = args.eval_path
    LR = args.lr
    TRAIN_SIZE = args.train_size
    EVAL_SIZE = args.eval_size
    BATCH_SIZE = args.batch_size

    args = vars(args)
    logger.info("Using the following args for training: ")
    for arg, val in args.items():
        logger.info(f"{arg}: {val}")

    train(NUM_EPOCHS,
          MODEL_TYPE,
          SAVE_NAME,
          TRAIN_PATH,
          EVAL_PATH,
          train_size=TRAIN_SIZE,
          eval_size=EVAL_SIZE,
          lr=LR,
          batch_size=BATCH_SIZE)
    
if __name__ == "__main__":
    main()
