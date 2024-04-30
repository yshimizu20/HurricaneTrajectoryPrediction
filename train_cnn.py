'''CPSC 452 Hurricane Trajectory Prediction
Written by Mike Zhang

Purpose: This program trains the CNN to predict windspeed from satellite images.
The idea is that we will pre-train embeddings of satellite images, which we 
will then concatenate with non-image data during training of the Neural ODE.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset

import torch.utils.data as data
import torchvision.transforms as tfs

from pathlib import Path
import matplotlib.pyplot as plt
import json

import netCDF4 as nc
import pandas as pd
import numpy as np
import os
import netCDF4
from PIL import Image

from model import CNN

'''Defining important functions and classes'''

# HURSAT class to prepare data for training
class HURSATDataset(Dataset):
    def __init__(self, root_dir, track_data, transform=None):
        self.root_dir = root_dir
        self.track_data = track_data
        self.transform = transform

        # Get list of file names
        self.files = os.listdir(root_dir)
        self.num_files = len(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.root_dir, file_name)
        
        # Load image data from NetCDF file
        raw_data = netCDF4.Dataset(file_path)
        image = raw_data.variables['IRWIN'][0]
        image_np = np.array(image)
        image = Image.fromarray(image_np)
        # image is 301 x 301
        
        # Extract storm name, date, and time from file name
        file_name = os.path.basename(file_path)
        file_name_parts = file_name.split('.')
        storm_name = file_name_parts[1]
        refined_name = storm_name[0] + storm_name[1:].lower()
        storm_year = int(file_name_parts[2])
        storm_month  = int(file_name_parts[3])
        storm_day = int(file_name_parts[4])
        time = int(file_name_parts[5])

        # Filter best track data to find matching row
        matching_track_data = self.track_data.loc[
            (self.track_data.name == refined_name) &
            (self.track_data.year == storm_year) &
            (self.track_data.month == storm_month) &
            (self.track_data.day == storm_day) &
            (self.track_data.hour*100 == time)
        ]

        # Get wind speed, pressure, long, and lat from matching row
        try:
            wind_speed = matching_track_data.wind.reset_index(drop=True)[0]
        except Exception:
            date = int(file_name_parts[2] + file_name_parts[3] + file_name_parts[4])
            print('\rCould not find label for image of ' + refined_name + ' at date ' + str(date) + ' and time ' + str(time))
            return None, None  # Return None for image and label if wind speed or pressure not found

        if self.transform:
            image = self.transform(image)

        return image, wind_speed

'''Pre-Training'''

# Assuming "HURDAT2_final.csv" is in your directory, read it
HURDAT2_data = pd.read_csv("HURDAT2_final.csv")

# Define path to image files
project_dir = Path('.').expanduser().absolute()
root_dir = project_dir / "HURSAT_Imagery"

# Define transforms
desired_image_size = (256, 256)

transform = tfs.Compose([
    tfs.Resize(desired_image_size),
    tfs.ToTensor()
])

# Initialize dataset
HURSAT_images = HURSATDataset(root_dir, HURDAT2_data, transform=transform)

# Filter out None values from the dataset
filtered_dataset = [sample for sample in HURSAT_images if sample[0] is not None]

# Convert list of tuples back to separate lists
images, labels = zip(*filtered_dataset)
images = np.array(images)
labels = np.array(labels)

'''DataLoader'''
# Define the ratio of the dataset to be used for training
train_ratio = 0.8

# Calculate the sizes of training and testing sets
train_size = int(train_ratio * len(filtered_dataset))
test_size = len(filtered_dataset) - train_size

# Split the dataset into training and testing sets
train_images, test_images = images[:train_size], images[train_size:]
train_labels, test_labels = labels[:train_size], labels[train_size:]

# Batch size
batch_size = 64

# Convert images and labels to PyTorch tensors
train_images_tensor = torch.tensor(train_images)
train_labels_tensor = torch.tensor(train_labels)
test_images_tensor = torch.tensor(test_images)
test_labels_tensor = torch.tensor(test_labels)

# Create TensorDataset objects for training and testing sets
train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DataLoader for training set
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,  
                          num_workers=2)  # Adjust based on available CPU cores

# DataLoader for testing set
test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=False,  
                         num_workers=2)  # Adjust based on available CPU cores

'''Training Step'''
# Define your neural network
model = CNN()
model.to(device)

# Definte loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=0.0001)

# Define number of epochs
num_epochs = 40

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to train mode
    running_train_loss = 0.0
    running_test_loss = 0.0
    
    # Training phase
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.squeeze()
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * inputs.size(0)
    
    # Calculate average training loss for the epoch
    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    
    # Evaluation phase (on test dataset)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels.float())
            running_test_loss += loss.item() * inputs.size(0)
    
    # Calculate average test loss for the epoch
    epoch_test_loss = running_test_loss / len(test_loader.dataset)
    
    # Print average losses for training and test datasets
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}")
    
# Create path to save model parameters
SAVE_DIR = Path('.').expanduser().absolute()
MODELS_DIR = SAVE_DIR / 'models'
PRE_CNN_DIR = MODELS_DIR / 'pretrain_CNN_params.pth'

# Save model
if not MODELS_DIR.is_dir():
    MODELS_DIR.mkdir(exist_ok=True)
    
torch.save(model.state_dict(), str(PRE_CNN_DIR))


