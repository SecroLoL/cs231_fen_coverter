import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Owlv2Processor, Owlv2Model

class OwlV2Classifier(nn.Module):
    def __init__(self, num_classes):
        super(OwlV2Classifier, self).__init__()
        
        # Load the pretrained OwlV2 model and processor
        self.owlv2 = Owlv2Model.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        
        # Use the projection dimension for the classification head
        hidden_size = self.owlv2.config.projection_dim
        
        # Define a custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, texts):
        # Preprocess the input images and texts
        inputs = self.processor(text=texts, images=images, return_tensors="pt")
        pixel_values = inputs['pixel_values']
        
        # Forward pass through the OwlV2 model
        outputs = self.owlv2(pixel_values=pixel_values)
        
        # Use the pooled output for classification
        pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0, :]
        
        # Forward pass through the classification head
        logits = self.classifier(pooled_output)
        
        return logits