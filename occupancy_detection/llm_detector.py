"""
Multi-modal LLM for occupancy detection
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torch import nn
from torchvision import models


#### Zero shot

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the class descriptions
class_descriptions = ["empty", "occupied"]  # 'empty' is index 0 , "occupied" is index 1.

# Preprocess text descriptions
text_inputs = processor(text=class_descriptions, return_tensors="pt", padding=True).to(device)

def predict(images, model, processor, text_inputs, device):
    # Preprocess the images
    image_inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    
    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**image_inputs, **text_inputs)
        logits_per_image = outputs.logits_per_image  # (batch_size, num_classes)
    
    # Compute probabilities
    probs = logits_per_image.softmax(dim=-1)
    
    # Get predicted class indices
    predicted_indices = probs.argmax(dim=-1)
    
    return predicted_indices


# Assuming test_dataloader is your DataLoader object for the test set
model.eval()

correct = 0
total = 0

test_dataloader = None  # TODO : Load this in

for batch in test_dataloader:
    images, labels = batch
    images = [image.to(device) for image in images]  # Ensure images are on the same device
    labels = labels.to(device)
    
    # Get predictions
    predicted_indices = predict(images, model, processor, text_inputs, device)
    
    # Update accuracy
    correct += (predicted_indices == labels).sum().item()
    total += labels.size(0)

accuracy = correct / total
print(f"Accuracy on test set: {accuracy:.4f}")



#### Finetune

class CLIPDetector(nn.Module):
    """
    CLIP Model for Occupancy Detection
    """
    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self, x): 
        inputs = self.processor(images=x, return_tensors="pt", padding=True)   # TODO: These need to be moved to device
        outputs = self.model(**inputs)
        logits_per_img = outputs.logits_per_image
        return logits_per_img