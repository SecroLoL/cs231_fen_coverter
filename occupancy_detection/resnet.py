"""
ResNet for occupancy detection
"""

import torch.nn.functional as F
import functools
import sys 
import os 

from torch import nn
from torchvision import models

class ResNetClassifier(nn.Module):
    """
    Baseline ResNet pretrained model for finetuning on occupancy classification
    """

    input_size = 100, 100
    pretrained = True

    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        n = self.model.fc.in_features
        self.model.fc = nn.Linear(n, 2)   
        self.params = {
            "head": list(self.model.fc.parameters())
        }

    def forward(self, x):
        return self.model(x)
    