import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
classifier_path = class_path = os.path.join(project_dir, 'saved_models', 'pieces', 'stratified_data', 'resnet', '20k_resnet.pt')

# Load a pre-trained model
model = models.resnet18(pretrained=True)
model.eval()

# Define the transformation to match the model's input
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the image
img = Image.open("path/to/your/image.jpg")
img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension

# Make the input require gradients
img_tensor.requires_grad_()

# Forward pass
output = model(img_tensor)

# Assume we are interested in the score for a specific class (e.g., class index 243)
target_class = 243
score = output[0, target_class]

# Backward pass
score.backward()

# Get the gradients
gradients = img_tensor.grad.data

# Convert the gradients to a numpy array and take the maximum absolute value across the color channels
saliency, _ = torch.max(gradients.abs(), dim=1)
saliency = saliency.squeeze().cpu().numpy()

# Normalize the saliency map
saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

# Visualize the saliency map
plt.imshow(saliency, cmap='hot')
plt.axis('off')
plt.show()