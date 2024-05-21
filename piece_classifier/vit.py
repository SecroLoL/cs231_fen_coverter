import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTImageProcessor

class ViTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ViTClassifier, self).__init__()

        # Load the pretrained ViT model for image classification
        self.vit = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=num_classes,
                                                             ignore_mismatched_sizes=True)
        self.image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    def forward(self, images):
        # Preprocess the input images
        inputs = self.image_processor(images=images, return_tensors="pt")
        pixel_values = inputs['pixel_values']

        # Forward pass through the ViT model
        outputs = self.vit(pixel_values=pixel_values)

        # Return logits from the ViT model directly
        logits = outputs.logits

        return logits