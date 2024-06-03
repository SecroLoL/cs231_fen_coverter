import os 
import sys 

sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Put the root dir as part of the PYTHONPATH

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from piece_classifier.model_types import ModelType
import numpy as np


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

    if model_type.value == ModelType.INCEPTION.value:
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