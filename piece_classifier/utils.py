import os 
import sys 

# TODO: add this to PYTHONPATH so that this isn't an issue
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # sys path issues, this is a quick fix for now

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from piece_classifier.model_types import ModelType
import numpy as np


def load_dataset(model_type: ModelType, data_path: str, batch_size: int = 32, subset_size: int = None):

    """
    TODO
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