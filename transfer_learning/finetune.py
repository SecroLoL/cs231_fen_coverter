"""
Finetune a trained occupancy or piece classifier on a transfer learning dataset.

"""
import argparse
import torch 
import torch.nn as nn
import torch.optim as optim
import os 
import sys 
import logging
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from piece_classifier.utils import load_dataset
from piece_classifier.model_types import ModelType, load_model, ARGPARSE_TO_TYPE
from occupancy_detection.utils import *

# Create and configure the logger
logger = logging.getLogger('Chess transfer learning')
logger.setLevel(logging.DEBUG)  # Set the logging level to DEBUG

# Create a file handler to log messages to a file
file_handler = logging.FileHandler('chess_transfer_learning.log')
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

# Load the transfer dataset
def finetune_model(chkpt_path: str, num_epochs: int, model_type: ModelType, save_path: str, train_path: str, 
                   batch_size: int = 32, lr: float = 0.0001, train_size: int = None) -> None:
    """
    Finetune a pretrained model for either piece classification or occupancy detection
    """
    device = default_device()
    model_checkpoint_path = generate_checkpoint_path(save_path)
    # Load datasets
    loader = load_dataset(
        model_type=model_type,
        data_path=train_path,
        batch_size=batch_size,
        subset_size=train_size
    )
    
    # Init model
    if device == "cpu":
        model = torch.load(chkpt_path, map_location=torch.device('cpu'))
    else:
        model = torch.load(chkpt_path).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")
    
    # Train loop
    for epoch in tqdm(range(num_epochs), desc="Beginning new epoch..."):
        model.train()
        running_loss = 0.0

        logger.info(f"Training? {model.training}")
        
        for inputs, labels in tqdm(loader, desc="Training batches..."):
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
        
        avg_loss = running_loss/len(loader)
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(loader):.4f}')

        torch.save(model, model_checkpoint_path)

        if avg_loss < best_loss:
            torch.save(model, save_path)
            best_loss = avg_loss
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}]: New best loss: {avg_loss}. Saved model checkpoint to {save_path}.")

    logger.info(f'Finished Training. Best loss {best_loss}.')
    
    if not os.path.exists(model_checkpoint_path):
        raise FileNotFoundError(f"Attempted to remove {model_checkpoint_path}, but could not find file.")
    os.remove(model_checkpoint_path)  # Delete checkpoint


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", type=str, default="/Users/alexshan/Desktop/cs231_fen_coverter/saved_models/transfer_learning/occupancy/cnn.pt", help="Path to model that is being finetuned")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--model_type", type=str, default="cnn", help="Model architecture ['cnn', 'resnet', 'inception']")
    parser.add_argument("--save_path", type=str, default="/Users/alexshan/Desktop/cs231_fen_coverter/saved_models/transfer_learning/occupancy/cnn_finetuned.pt", help="Output file")
    parser.add_argument("--train_path", type=str, default="/Users/alexshan/Desktop/chesscog/data/transfer_learning/occupancy/train/", help="Path to training root dir")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    
    args = parser.parse_args()

    CHECKPOINT_PATH = args.checkpoint_path
    NUM_EPOCHS = args.num_epochs
    MODEL_TYPE = ARGPARSE_TO_TYPE.get(args.model_type)
    SAVE_PATH = args.save_path
    TRAIN_PATH = args.train_path
    BATCH_SIZE = args.batch_size

    args = vars(args)
    for k, v in args.items():
        logging.info(f"{k}: {v}")

    finetune_model(CHECKPOINT_PATH,
                   NUM_EPOCHS,
                   MODEL_TYPE,
                   SAVE_PATH,
                   TRAIN_PATH,
                   BATCH_SIZE)



if __name__ == "__main__":
    main()