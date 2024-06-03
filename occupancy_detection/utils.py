"""
General utility functions for all code in this dir
"""
import os 
import sys 
import torch

# TODO: add this to PYTHONPATH so that this isn't an issue
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # sys path issues, this is a quick fix for now

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from occupancy_detection.model_types import ModelType
import numpy as np

def generate_checkpoint_path(save_path: str):

    """
    Given a model path, create a checkpoint path by appending '_ckpt' to the filename.

    Args:
    model_path (str): The original path to the model file.

    Returns:
    str: The new path with '_ckpt' appended to the filename.
    """
    # Extract the directory and filename
    dir_name = os.path.dirname(save_path)
    base_name = os.path.basename(save_path)
    
    # Split the filename into name and extension
    name, ext = os.path.splitext(base_name)
    
    # Create the new filename with '_ckpt' appended
    new_base_name = f"{name}_ckpt{ext}"
    
    # Construct the new path
    new_path = os.path.join(dir_name, new_base_name)
    
    return new_path

def load_dataset(model_type: ModelType, data_path: str, batch_size: int = 32, subset_size: int = None) -> DataLoader:

    """
    Creates a DataLoader for a provided dataset, adjusting input sizes and shapes according to which model is 
    being trained.

    Args: 
        model_type (ModelType): The ModelType class of the model that this dataset is used for. This is included to support the INCEPTION class, which requires addtl preprocessing.
        data_path (str): The path to the data directory containing examples.
        batch_size (int): Size of each batch of examples
        subset_size (int, optional): If not using the entire dataset for the DataLoader, the number of examples to sample from the dataset. Defaults to all examples.

    Returns:
        A PyTorch DataLoader object that can be iterated to obtain the examples and labels.
    """

    if model_type is None:
        raise ValueError(f"ModelType {model_type} is invalid.")
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if model_type == ModelType.INCEPTION:
        transform = transforms.Compose([
        transforms.Resize((299, 299)),  
        transforms.ToTensor(),
        ])  # InceptionV3 requires images to be size 299 x 299

    dataset = datasets.ImageFolder(root=data_path, transform=transform)

    if subset_size is not None:
        indices = np.random.choice(len(dataset), subset_size, replace=False)
        final_dataset = Subset(dataset, indices)
        dataset = final_dataset
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def default_device():
    """
    Pick a default device based on what's available on this system
    """
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

