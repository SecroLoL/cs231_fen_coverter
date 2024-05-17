"""
Multi-modal LLM for occupancy detection
"""

import torch.nn.functional as F
import functools
import sys 
import os 

from torch import nn
from torchvision import models