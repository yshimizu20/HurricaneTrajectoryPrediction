'''CPSC 452 Hurricane Trajectory Prediction
Written by Mike Zhang

Purpose: This program takes the pre-trained form of the satellite images
in embedding form.
'''

import torch
import torchvision.transforms as tfs
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, Dataset

from pathlib import Path

from model import CNN 

import netCDF4 as nc
import pandas as pd
import numpy as np
import os
import netCDF4
from PIL import Image
from datetime import datetime

#-----------------------Define Class and Functions to Handle Vector Info---------------------
class HURSATVector(Dataset):
    def __init__(self, root_dir, track_data, transform=None):
        self.root_dir = root_dir
        self.track_data = track_data
        self.transform = transform

        # Get list of file names
        self.files = os.listdir(root_dir)
        self.num_files = len(self.files)

        # Create a mapping between storm names and numerical identifiers
        self.storm_name_to_id = {name: idx for idx, name in enumerate(sorted(set(self.track_data['name'])))}

    def __len__(self):
        return len(self.files)

    def get_reference_date(self, track_data, storm_name):
        # Filter track_data for the given storm_name and create a copy
        storm_data = self.track_data[self.track_data['name'] == storm_name].copy()

        # Concatenate year, month, day, and hour into a single numeric value
        storm_data.loc[:, 'timestamp'] = storm_data['year'] * 1000000 + storm_data['month'] * 10000 + storm_data['day'] * 100 + storm_data['hour']

        # Find the row with the minimum timestamp
        earliest_timestamp_row = storm_data.loc[storm_data['timestamp'].idxmin()]

        # Assemble the reference date from the row with the minimum timestamp
        reference_date = datetime(
            earliest_timestamp_row['year'],
            earliest_timestamp_row['month'],
            earliest_timestamp_row['day'],
            earliest_timestamp_row['hour']
        )

        return reference_date

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.root_dir, file_name)
        
        # Load image data from NetCDF file
        raw_data = nc.Dataset(file_path)
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
        time = int(file_name_parts[5])/100

        # Filter best track data to find matching row
        matching_track_data = self.track_data.loc[
            (self.track_data.name == refined_name) &
            (self.track_data.year == storm_year) &
            (self.track_data.month == storm_month) &
            (self.track_data.day == storm_day) &
            (self.track_data.hour == int(time))
        ]

        # Get wind speed, pressure, long, and lat from matching row
        try:
            wind_speed = matching_track_data.wind.reset_index(drop=True)[0]
            pressure = matching_track_data.pressure.reset_index(drop=True)[0]
            lat = matching_track_data.lat.reset_index(drop=True)[0]
            long = matching_track_data.long.reset_index(drop=True)[0]
        except Exception:
            date = int(file_name_parts[2] + file_name_parts[3] + file_name_parts[4])
            print('\rCould not find label for image of ' + refined_name + ' at date ' + str(date) + ' and time ' + str(time))
            return None, None  # Return None for image and label if wind speed or pressure not found

        # Convert to numerical values
        lat = int(lat)
        long = int(long)
        wind_speed = int(wind_speed)
        pressure = int(pressure)

        # Convert storm_name to its corresponding numerical identifier
        storm_id = self.storm_name_to_id[refined_name]

        # Get the reference date for normalization from track_data
        reference_date = self.get_reference_date(self.track_data, refined_name)

        # Convert date to a normalized timestamp in 6-hour intervals
        date_str = f"{storm_year}-{storm_month:02d}-{storm_day:02d}-{int(time):02d}"  # Format: YYYY-MM-DD-HH
        date_obj = datetime.strptime(date_str, "%Y-%m-%d-%H")
        time_diff = (date_obj - reference_date).total_seconds()#  / (6 * 3600)  # Calculate time difference in 6-hour intervals
        timestamp = int(time_diff)

        # Create label containing all relevant information
        label = np.array([storm_id, timestamp, long, lat, wind_speed, pressure], dtype=np.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

#-------------------------Define ncessary function and folders---------------------
# Set path to save CNN parameters and vectors
SAVE_DIR = Path('.').expanduser().absolute()
MODELS_DIR = SAVE_DIR / 'models'
PRE_CNN_DIR = MODELS_DIR / 'pretrain_CNN_params.pth'
VECTOR_REPS_DIR = SAVE_DIR / 'vectorized_representations.pth'

# Function to extract vectorized representations from the trained model
def extract_vectorized_representations(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    concatenated_data = []

    with torch.no_grad():
        for inputs, batch_labels in data_loader:
            inputs = inputs.to(device)
            inputs = inputs.unsqueeze(1)
            vector_rep = model(inputs, return_vector=True).cpu().numpy()
            
            # Concatenate vector representation with label values
            concatenated_sample = np.concatenate((batch_labels, vector_rep), axis=1)
            concatenated_data.append(concatenated_sample)

    # Convert concatenated data to array
    concatenated_data_array = np.vstack(concatenated_data)

    return concatenated_data_array

#-------------------------Conversion Step---------------------
# Set device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your dataset
HURDAT2_data = pd.read_csv("HURDAT2_final.csv")

# Define path to image files
root_dir = SAVE_DIR / "HURSAT_Imagery"

# Define transforms
desired_image_size = (256, 256)

transform = tfs.Compose([
    tfs.Resize(desired_image_size)
])

# Initialize dataset
HURSAT_vectors = HURSATVector(root_dir, HURDAT2_data, transform=transform)

# Set device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your pre-trained model
model = CNN()
model.load_state_dict(torch.load(PRE_CNN_DIR))
model.to(device)

# Filter out None values from the dataset
filtered_dataset = [sample for sample in HURSAT_vectors if sample[0] is not None]

# Convert list of tuples back to separate lists
images, labels = zip(*filtered_dataset)
images = np.array(images)
labels = np.array(labels)

# Create TensorDataset object for the filtered dataset
filtered_dataset = TensorDataset(torch.tensor(images), torch.tensor(labels))

# Define batch
batch_size = 16

# DataLoader for the filtered dataset
data_loader = DataLoader(filtered_dataset, batch_size=batch_size, shuffle=False)

# Extract vectorized representations from the dataset using the loaded model
vectorized_reps = extract_vectorized_representations(model, data_loader)

# Save the vectorized representations to a file
torch.save(vectorized_reps, VECTOR_REPS_DIR)
print("Vectorized representations saved successfully.")

