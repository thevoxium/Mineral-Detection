# -*- coding: utf-8 -*-
"""Mineral_Detection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1U53IhFU99mSkF08wjQ3OJOyJFC8W21JB
"""

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/mineral_project

import torch
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import random_split
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
from tqdm.notebook import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device Available: {}".format(device))

root_data_folder = "./data/"

target_labels = ['biotite', 'bornite', 'chrysocolla', 'malachite', 
                'muscovite', 'pyrite', 'quartz']

dataset = ImageFolder(root_data_folder,
                     transform=transforms.ToTensor())

print("Length of Dataset: {}".format(len(dataset)))
print("Total Number of Classes: {} and Labels : {}".format(len(dataset.classes),dataset.classes))

fig = plt.figure(figsize=(25, 4))

for i in range(20):
    image, label = dataset[i]
    ax = fig.add_subplot(2, 10, i+1)
    ax.imshow(image.permute(1,2,0))
    ax.set_title(target_labels[label], color='green')

height = []
width = []

for i in range(len(dataset)):
    image, _ = dataset[i]
    height.append(image.size(1))
    width.append(image.size(2))
    
min_h = np.min(height)
mean_h = np.mean(height)
max_h = np.max(height)

min_w = np.min(width)
mean_w = np.mean(width)
max_w = np.max(width)

print("Height : Min = {}, Max = {}, Mean = {}".format(min_h, max_h, mean_h))
print("Width : Min = {}, Max = {}, Mean = {}".format(min_w, max_w, mean_w))

data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.RandomRotation(50),
                                    transforms.RandomVerticalFlip(p=0.7),
                                    transforms.RandomHorizontalFlip(p=0.7),
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225]),
                                    ])

load_data = ImageFolder(root_data_folder, transform = data_transform)
#load_data = ConcatDataset([transformed_data, dataset])
print("Length of transformed data: {}".format(len(load_data)))

fig = plt.figure(figsize=(25, 4))

for i in range(20):
    image, label = load_data[i]
    ax = fig.add_subplot(2, 10, i+1)
    ax.imshow(image.permute(1,2,0))
    ax.set_title(target_labels[label], color='green')

batch_size = 32
random_seed = 42
validation_split = 0.1
shuffle_dataset = True

dataset_size = len(load_data)
indices = list(range(dataset_size))

split = int(np.floor(validation_split*dataset_size))

if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(load_data, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = DataLoader(load_data, batch_size=1,
                                                sampler=valid_sampler)

print("Train data length: {}".format(len(train_loader)))
print("Test data length: {}".format(len(validation_loader)))

for images, _ in train_loader:
    print('images.shape:', images.shape)
    plt.figure(figsize=(16,16))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    break

class Mineral_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 48, 11, stride=3, padding=0),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(3, 1), #out 70x70

            nn.Conv2d(48, 128, 5, stride=1, padding=0),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(3, 1),#out 64x64

            nn.Conv2d(128, 128, 4, stride=1, padding=0),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(4, 3),#out 20x20

            nn.Conv2d(128, 64, 3, stride=1, padding=0),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(3, 3),#out 20x20

            nn.Flatten(),
            nn.Linear(64*6*6, 512),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.3),
            nn.Linear(512, 7),
            nn.LogSoftmax(dim=1),
            )
        
    def forward(self, x):
        out = self.net(x)
        return out

model = Mineral_1().to(device)
print(model)

def fit(epochs, model, train_loader, val_loader, criterion, optimizer):
    
    train_losses = []
    test_losses = []
    
    start_time = time.time()
    
    

    for e in range(epochs):

        loop = tqdm(train_loader, leave=True)
        since = time.time()
        running_loss = 0
        train_acc = 0
        
        i = 1
        
        for idx, (image, label) in enumerate(loop):
            
            i+=1
            optimizer.zero_grad()
            image, label = image.to(device), label.to(device)
            
            output = model(image)
            
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            running_loss+=loss.item()
        print("Loss for epoch {}/{} is {} ".format(e+1, epochs, running_loss/i))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)
epoch = 100
fit(epoch, model, train_loader, validation_loader, criterion, optimizer)

PATH = "/content/drive/MyDrive/mineral_project/final.h5"
torch.save(model.state_dict(), PATH)

def predict_label(model, dataloader):
    prediction_list = []
    labels = []
    model.to(device)
    model.eval()
    for i, batch in enumerate(dataloader):
        image, label = batch
        image = image.to(device); label = label.to(device)
      
        out = model(image)
        ps = torch.exp(out)
        _, top_class = torch.max(ps , 1)
        preds = np.squeeze(top_class.cpu().numpy())
        #print(preds.shape)
        prediction_list.append(preds)
        labels.append(label.cpu().numpy())
    return np.squeeze(prediction_list), np.squeeze(labels)

y_predict, y_true = predict_label(model, validation_loader)

print(classification_report(y_true, y_predict))

y_true

y_predict

