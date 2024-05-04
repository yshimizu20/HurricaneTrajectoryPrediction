"""
CPSC 452 Hurricane Trajectory Prediction

Purpose: Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

"""Pre-trained CNN to produce embeddings of images"""


class CNN(nn.Module):
    """
    CNN model that takes in a x_dim * y_dim image and outputs a scalar.
    """

    def __init__(self, x_dim=256, y_dim=256):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x, return_vector=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Return the output of the second last fully connected layer if return_vector is True
        if return_vector:
            return self.fc3(x)

        # Otherwise, proceed
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


"""Dynamics to predict coordinates per timestamp"""


class FlowNet(nn.Module):
    def __init__(self, n_dim):
        super().__init__()
        self.func = nn.Sequential(
            nn.Linear(n_dim, 2 * n_dim),
            nn.ReLU(),
            nn.Linear(2 * n_dim, 2 * n_dim),
            nn.ReLU(),
            nn.Linear(2 * n_dim, n_dim),
        )

    def forward(self, x):
        return self.func(x)
