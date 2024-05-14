import torch
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch import nn, optim

# TODO: align code with this
# Code assumes that we have our data structured as such: 
# data/
#     train/
#         occupied/
#         empty/
#     validation/
#         occupied/
#         empty/


# TODO: Clean up the code a bit, make it much cleaner

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
 
data_dir = 'data'
image_datasets = {x: datasets.ImageFolder(f'{data_dir}/{x}', data_transforms[x])
                  for x in ['train', 'validation']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
               for x in ['train', 'validation']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
class_names = image_datasets['train'].classes

# ImageFolder class expects the dataset to be organized as 
# root/
#     class1/
#         img1.jpg
#         img2.jpg
#         ...
#     class2/
#         img3.jpg
#         img4.jpg
#         ...
#     ...

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model_ft = models.inception_v3(pretrained=True)
num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs, 1)
model_ft = model_ft.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model_ft.fc.parameters(), lr=0.001)

def train_model(model, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()  

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).float().view(-1, 1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.sigmoid(outputs) > 0.5  # arbitrarily choose 0.5 as the threshold that we want
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model

def main():

    SAVE_FILE_PATH = os.path.join(os.path.dirname(__file__), "saved_models", "sqaure_classififer.pth")
    model_ft = train_model(model_ft, criterion, optimizer, num_epochs=10)
    torch.save(model_ft.state_dict(), 'chess_square_classifier.pth')



if __name__ == "__main__":
    main()
