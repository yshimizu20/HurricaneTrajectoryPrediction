'''CPSC 452 Hurricane Trajectory Prediction
Written by Mike Zhang

Purpose: This model trains the CNN to predict windspeed from satellite images.
The idea is that we will pre-train embeddings of satellite images, which we 
will then concatenate with non-image data during training of the Neural ODE.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, data
import torchvision.transforms as tfs

from pathlib import Path

import netCDF4 as nc
import pandas as pd
import os

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
        with nc.Dataset(file_path, 'r') as f:
            image = f.variables['IRWIN'][0]
        
        # Extract storm name, date, and time from file name
        file_name = os.path.basename(file_path)
        file_name_parts = file_name.split('.')
        storm_name = file_name_parts[1]
        storm_year = int(file_name_parts[2])
        storm_month  = int(file_name_parts[3])
        storm_day = int(file_name_parts[4])
        time = int(file_name_parts[5])

        # Filter best track data to find matching row
        matching_track_data = self.track_data.loc[
            (self.track_data.name == storm_name) &
            (self.track_data.year == storm_year) &
            (self.track_data.month == storm_month) &
            (self.track_data.day == storm_day) &
            (self.track_data.hour == time)
        ]

        # Get wind speed from matching row
        try:
            wind_speed = matching_track_data.wind.reset_index(drop=True)[0]
        except Exception:
            date = int(file_name_parts[2] + file_name_parts[3] + file_name_parts[4])
            print('\rCould not find label for image of ' + storm_name + ' at date ' + str(date) + ' and time ' + str(time))
            return None, None  # Return None for image and label if wind speed not found

        if self.transform:
            image = self.transform(image)

        return image, wind_speed

'''Training Implementation'''

# Assuming "HURDAT2_final.csv" is in your directory, read it
HURDAT2_data = pd.read_csv("HURDAT2_final.csv")

# Define path to image files
project_dir = Path('.').expanduser().absolute()
root_dir = project_dir / "Satellite Imagery"

# Define transforms
desired_image_size = (256, 256)

transform = tfs.Compose([
    tfs.Resize(desired_image_size),
    tfs.RandomResizedCrop(desired_image_size,
                         scale=(0.6,1.6)),
    tfs.RandomHorizontalFlip(p=0.5),
    tfs.ToTensor(),
])

# Initialize dataset
HURSAT_images = HURSATDataset(root_dir, HURDAT2_data, transform=transform)

'''DataLoader'''
# Define the ratio of the dataset to be used for training
train_ratio = 0.8

# Calculate the sizes of training and testing sets
train_size = int(train_ratio * len(HURSAT_images))
test_size = len(HURSAT_images) - train_size

# Split the dataset into training and testing sets
train_dataset, test_dataset = data.random_split(HURSAT_images, [train_size, test_size])

# Batch size
batch_size = 64

# DataLoader for training set
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           num_workers=2,
                                           shuffle=True)

# DataLoader for testing set
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          num_workers=2,
                                          shuffle=False)

