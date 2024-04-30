'''
CPSC 452 Hurricane Trajectory Prediction
Written by Mike Zhang

Purpose: implement and train Neural ODE
'''

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchdyn.core import NeuralODE
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset

from pathlib import Path

from model import FlowNet

# Define file paths to vectorized data
SAVE_DIR = Path('.').expanduser().absolute()
VECTOR_REPS_DIR = SAVE_DIR / 'vectorized_representations.pth'

# Describe Learner (training step) for NDE with Pytorch-Lightning
class Learner(pl.LightningModule):
    def __init__(self, 
                 model: nn.Module, 
                 train_loader: DataLoader):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
    
    def train_dataloader(self):
        return self.train_loader

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # x represents all features except long and lat
        # x: batch_size x 34
        # y represents long and lat
        # y: batch_size x 2
        # We deploy forced teaching by feeding the model the coorindates of the prior step

        # Forward pass through the neural ODE model with teacher forcing
        t_eval, y_hat = self.model(torch.cat((x, y), dim=1), t_span)
        y_hat = y_hat[-1]  # select the last point of the solution trajectory
        
        # Extract predicted longitude and latitude (second to last and last columns)
        predicted_longitude = y_hat[:, 34]
        predicted_latitude = y_hat[:, 35]
        
        # Compute Mean Squared Error (MSE) loss for longitude and latitude
        loss_longitude = F.mse_loss(predicted_longitude, y[:, 0])
        loss_latitude = F.mse_loss(predicted_latitude, y[:, 1])
        
        # Total loss is the sum of MSE losses for longitude and latitude
        loss = loss_longitude + loss_latitude
    
        # Log loss and return it
        self.log('loss', loss, prog_bar=True)
        return {'loss': loss}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        # x represents all features except long and lat
        # x: batch_size x 34
        # y represents long and lat
        # y: batch_size x 2
        
        # Forward pass through the neural ODE model
        t_eval, y_hat = model(torch.cat((x, y), dim=1), t_span)
        y_hat = y_hat[-1]  # select the last point of the solution trajectory
        
        # Extract predicted longitude and latitude
        predicted_longitude = y_hat[:, 34]
        predicted_latitude = y_hat[:, 35]
        
        # Compute Mean Squared Error (MSE) loss for longitude and latitude
        loss_longitude = F.mse_loss(predicted_longitude, y[:, 0])
        loss_latitude = F.mse_loss(predicted_latitude, y[:, 1])
        
        # Total loss is the sum of MSE losses for longitude and latitude
        loss = loss_longitude + loss_latitude

        # Log loss and return it
        self.log('test_loss', loss, prog_bar=True)
        return {'test_loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define NDE Model
'''
Input is size 36: long, lat, wind, pressure, 
and 32 vectorized image features
Hidden features size: 10
Output features sizel: 2 (long and lat)
''' 
model = NeuralODE(FlowNet(36), sensitivity='adjoint', solver='dopri5').to(device)

'''
Hurricanes last 10 days. That's 40 6-hour periods.
'''
t_span = torch.linspace(0, 1, 40)

# Load data and print first 5 elements
'''Format:
First Column: longitude
Second: Latitude
Third: Wind Speed
Fourth: Presure
Afterwards: Image Embeddings
'''
storm_data = torch.load(VECTOR_REPS_DIR)

# Investigate data and model
# print(storm_data[:5])
# print(model)

# Split storm_data into features (input) and labels (output)
features = storm_data[:, 2:]
labels = storm_data[:, :2]

# Convert features and labels to PyTorch tensors
features_tensor = torch.tensor(features, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.float32)

# Create Dataset objects for features and labels
storm_dataset = TensorDataset(features_tensor, labels_tensor)

# Calculate sizes of train and test sets based on the 80-20 split
total_samples = len(storm_dataset)
train_size = int(0.8 * total_samples)
test_size = total_samples - train_size

# Split the dataset into train and test sets
train_dataset, test_dataset = random_split(storm_dataset, [train_size, test_size])

# Define batch size for DataLoader
batch_size = 64

# Create DataLoaders for train and test sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create our PyLightning learner to handle training steps
learn = Learner(model, train_loader)
learn = learn.to(device)

# Create trainer
# Parameters subject to change (i.e, increase epochs)
trainer = pl.Trainer(min_epochs=10, 
                     gradient_clip_val=100,
                     gradient_clip_algorithm='value',
                     log_every_n_steps=5,
                     max_epochs=20)

# Train the model:
trainer.fit(learn)

# Test the model
# test_result = trainer.test(learn, dataloaders=test_loader)

# Print train result
print(trainer.logged_metrics)

# Print test result
# print(test_result)