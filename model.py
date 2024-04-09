import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    CNN model that takes in a x_dim * y_dim image and outputs a 128-dimensional vector.
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

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(32 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 128)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 32 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
