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
image_folder = os.path.join(project_dir, 'data_visualization', 'sample_images')
classifier_path = os.path.join(project_dir, 'saved_models', 'pieces', 'stratified_data', 'cnn', '20k_cnn.pt')
output_folder = os.path.join(project_dir, 'data_visualization', 'saliency_maps', 'cnn')
resized_folder = os.path.join(output_folder, 'resized_images')

# Create the output directories if they don't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(resized_folder, exist_ok=True)

# Load a pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(classifier_path, map_location=device)
model.eval()

# Define the transformation to match the model's input
preprocess = transforms.Compose([
    transforms.Resize((200, 100)),
    transforms.ToTensor()
])

def generate_saliency_map(image_path):
    # Load the image
    img = Image.open(image_path).convert("RGB")
    
    # Resize the image and save it
    resized_img = img.resize((100, 200))
    resized_img_path = os.path.join(resized_folder, os.path.basename(image_path))
    resized_img.save(resized_img_path)
    
    # Preprocess the image for the model
    img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
    img_tensor = img_tensor.to(device)
    
    # Make the input require gradients
    img_tensor.requires_grad_()

    # Forward pass
    output = model(img_tensor)

    # Get the class with the highest score
    target_class = output.argmax().item()
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
    
    return saliency, resized_img_path

# Process all images in the sample_images folder
for filename in tqdm(os.listdir(image_folder)):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        input_path = os.path.join(image_folder, filename)
        output_path = os.path.join(output_folder, filename)
        output_path_saliency = os.path.join(output_folder, f'saliency_{filename}')
        
        saliency, resized_img_path = generate_saliency_map(input_path)
        
        # Save the saliency map as a heatmap
        plt.imsave(output_path_saliency, saliency, cmap='hot')
        
        print(f'Saliency map and resized image saved for {filename}')