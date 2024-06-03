import torch
import torch.optim as optim
from torchvision import models, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
classifier_path = class_path = os.path.join(project_dir, 'saved_models', 'pieces', 'stratified_data', 'resnet', '20k_resnet.pt')

# Load a pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(classifier_path, map_location=device)
model.eval()  # Set the model to evaluation mode
model.to(device)  # Move the model to the GPU

# Initialize a random image
img_tensor = torch.randn((1, 3, 224, 224), requires_grad=True, device=device)

# Define the target class
target_class = 4

# Set up the optimizer
optimizer = optim.Adam([img_tensor], lr=0.01)

def total_variation_loss(x):
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w

# Number of iterations to optimize
num_iterations = 400

for i in tqdm(range(num_iterations)):
    optimizer.zero_grad()
    output = model(img_tensor)
    loss = -output[0, target_class]
    tv_loss = total_variation_loss(img_tensor)
    total_loss = loss + 1e-5 * tv_loss  # Adjust the weight of the TV loss as needed
    total_loss.backward()
    optimizer.step()
    img_tensor.data = torch.clamp(img_tensor.data, 0, 1)
    if i % 10 == 0:
        print(f"Iteration {i}/{num_iterations}, Loss: {total_loss.item()}")

# Move the optimized image to CPU and convert to numpy array
img_np = img_tensor.detach().cpu().numpy()[0]
img_np = np.transpose(img_np, (1, 2, 0))  # Change dimensions to HWC

# Normalize the image
img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

# Visualize the image
plt.imshow(img_np)
plt.axis('off')
plt.show()