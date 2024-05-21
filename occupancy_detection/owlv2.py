import torch
import torch.nn as nn
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL import Image

class OwlV2OccupancyDetector(nn.Module):
    """OwlV2 Occupancy Detector for detecting if a chess square is occupied by a piece or empty."""

    input_size = 960, 960
    pretrained = True

    def __init__(self):
        super(OwlV2OccupancyDetector, self).__init__()
        
        # Load the pretrained OwlV2 model for object detection
        self.owlv2 = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")

        self.fc = nn.Linear(3600, 2)  # occupancy is binary classification

    def forward(self, images, texts):
        # Preprocess the input images and texts
        inputs = self.processor(images=images, text=texts, return_tensors="pt", padding=True, truncation=True)
        pixel_values = inputs['pixel_values']
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Forward pass through the OwlV2 model
        outputs = self.owlv2(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.fc(outputs.logits.squeeze())  # maybe we should be flattening instead of squeezing
        
        return outputs
