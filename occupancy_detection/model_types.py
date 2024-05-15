from enum import Enum
from occupancy_detection.baseline_cnn import CNN_100
from occupancy_detection.resnet import ResNetClassifier

"""
Model types for cleaner handling in training and eval files
"""

class ModelType(Enum):
    CNN_100 = "CNN_100"
    RESNET = "RESNET"
    INCEPTION = "INCEPTION"
    LLM = "LLM"


def load_model(model_type: ModelType):
    """
    Loads in a base model given a specific ModelType
    """
    if model_type == ModelType.CNN_100:
        return CNN_100()
    if model_type == ModelType.RESNET:
        return ResNetClassifier()
    # TODO add Inception and LLM model version
    