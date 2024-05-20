import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTImageProcessor

class ViTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ViTClassifier, self).__init__()

        # Load the pretrained ViT model and feature extractor
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

        # Use the projection dimension for the classification head
        hidden_size = self.vit.config.hidden_size

        # Define a custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, images):
        # Preprocess the input images
        inputs = self.image_processor(images=images, return_tensors="pt")
        pixel_values = inputs['pixel_values']

        # Forward pass through the ViT model
        outputs = self.vit(pixel_values=pixel_values)

        # Use the pooled output for classification
        pooled_output = outputs.pooler_output

        # Forward pass through the classification head
        logits = self.classifier(pooled_output)

        return logits