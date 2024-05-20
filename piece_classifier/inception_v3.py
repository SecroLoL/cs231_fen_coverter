"""
Inception V3 model for occupancy detection
"""

import torch.nn.functional as F
import functools
import sys 
import os 

from torch import nn
from torchvision import models

class InceptionV3Classifier(nn.Module):
    """InceptionV3 Classifier for piece classification

    The final layer of InceptionV3's classification head is replaced with a fully-connected layer that has two
    output units to each of the following classes:
        White pawn, Black pawn, White knight, Black knight, ...
    """
    
    input_size = 299, 299
    pretrained = True

    def __init__(self):
        super().__init__()
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        # Replace auxillary network classification head 
        n = self.model.AuxLogits.fc.in_features
        self.model.AuxLogits.fc = nn.Linear(n, 12)  # 12 classes: one for each (color, piece) combination
        
        # Replace primary network classification head
        n = self.model.fc.in_features
        self.model.fc = nn.Linear(n, 12)
        self.params = {
            "head": list(self.model.AuxLogits.fc.parameters()) + list(self.model.fc.parameters())
        }

    def forward(self, x):
        return self.model(x)

