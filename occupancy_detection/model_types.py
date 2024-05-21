from enum import Enum
from occupancy_detection.baseline_cnn import CNN_100
from occupancy_detection.resnet import ResNetClassifier
from occupancy_detection.inception_v3 import InceptionV3Classifier
from occupancy_detection.owlv2 import OwlV2OccupancyDetector

"""
Model types for cleaner handling in training and eval files
"""

class ModelType(Enum):
    CNN_100 = "CNN_100"
    RESNET = "RESNET"
    INCEPTION = "INCEPTION"
    LLM = "LLM"
    OWL = "OWL"


ARGPARSE_TO_TYPE = {
    "cnn": ModelType.CNN_100,
    "resnet": ModelType.RESNET,
    "inception": ModelType.INCEPTION,
    "owlv2": ModelType.OWL,
}


def load_model(model_type: ModelType):
    """
    Loads in a base model given a specific ModelType
    """
    if model_type == ModelType.CNN_100:
        return CNN_100()
    if model_type == ModelType.RESNET:
        return ResNetClassifier()
    if model_type == ModelType.INCEPTION:
        return InceptionV3Classifier()
    if model_type == ModelType.OWL:
        return OwlV2OccupancyDetector()
    