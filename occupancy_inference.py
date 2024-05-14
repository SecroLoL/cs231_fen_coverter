import torch
import os 

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch import nn, optim
from PIL import Image

# TODO clean up this code and make each function modular

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the trained model
model_ft = models.inception_v3(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 1)

SAVED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_models", 'square_classifier.pth')

model_ft.load_state_dict(torch.load(SAVED_MODEL_PATH))
model_ft = model_ft.to(device)
model_ft.eval()

# Data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}  # InceptionV3 expects size 299x299
# Use means [0.485, 0.456, 0.406] and STD [0.229, 0.224, 0.225] 
# because these are the values for the RGB in the imagenet dataset where inceptionv3 was trained.

# Inference function
def is_square_occupied(image_path):
    img = Image.open(image_path)
    img = data_transforms['validation'](img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model_ft(img)
        preds = torch.sigmoid(outputs) > 0.5
        return preds.item() == 1


def main():
    image_path = 'example.jpg'
    occupied = is_square_occupied(image_path)
    if occupied:
        print("The square is occupied")
    else:
        print("The square is empty")


if __name__ == "__main__":
    main()
