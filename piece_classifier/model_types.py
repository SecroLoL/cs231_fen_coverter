from enum import Enum
from piece_classifier.cnn import CNN_100
from piece_classifier.inception_v3 import InceptionV3Classifier
from piece_classifier.resnet import ResNetClassifier
from piece_classifier.owl_v2 import OwlV2Classifier
from piece_classifier.vit import ViTClassifier

class ModelType(Enum):
    CNN_100 = "CNN_100"
    RESNET = "RESNET"
    INCEPTION = "INCEPTION"
    LLM = "LLM"
    OWL = "OWL"
    VIT = "VIT"


ARGPARSE_TO_TYPE = {
    "cnn": ModelType.CNN_100,
    "resnet": ModelType.RESNET,
    "inception": ModelType.INCEPTION,
    "llm": ModelType.LLM,
    "owl": ModelType.OWL,
    "vit": ModelType.VIT,
}


def load_model(model_type: ModelType):
    if model_type == ModelType.CNN_100:
        return CNN_100()
    if model_type == ModelType.INCEPTION:
        return InceptionV3Classifier()
    if model_type == ModelType.RESNET:
        return ResNetClassifier()
    if model_type == ModelType.OWL:
        return OwlV2Classifier(12)  # 12 classes, one for each (color, piece) combo
    if model_type == ModelType.VIT:
        return ViTClassifier(12)
