"""
CPSC 452 Hurricane Trajectory Prediction
Written by Mike Zhang

Purpose: implement and train Neural ODE
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchdyn.core import NeuralODE
import json
from collections import defaultdict

from pathlib import Path

from model import FlowNet
from loader import CustomDataLoader

# Define file paths to vectorized data
SAVE_DIR = Path(".").expanduser().absolute()
VECTOR_REPS_DIR = SAVE_DIR / "vectorized_representations.pth"
JSON_DIR = SAVE_DIR / "HURDAT2_final.json"


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_data():
    """
    Function to prepare data for training and testing
    Returns: training and test data loaders
    """
    # Load data and print first 5 elements
    """Format:
    First Column: longitude
    Second: Latitude
    Third: Wind Speed
    Fourth: Presure
    Afterwards: Image Embeddings
    """
    storm_data = torch.load(VECTOR_REPS_DIR)

    # Load JSON file
    with open(JSON_DIR, "r") as f:
        storm_name_to_id = json.load(f)

    all_ids = list(storm_name_to_id.values())

    # Split data into training and test sets
    data_size = len(storm_name_to_id)
    training_size = int(0.8 * data_size)

    # randomly assign ids to training and test data
    perm = torch.randperm(data_size)
    training_id_idx = perm[:training_size]
    test_id_idx = perm[training_size:]

    training_ids = [all_ids[i] for i in training_id_idx]
    test_ids = [all_ids[i] for i in test_id_idx]

    training_data = defaultdict(lambda: {"X": [], "y": []})
    test_data = defaultdict(lambda: {"X": [], "y": []})

    # walk through the data and print the 1st element
    for vec in storm_data:
        storm_id = int(vec[0])
        if storm_id in training_ids:
            training_data[storm_id]["X"].append(torch.tensor(vec[4:]))
            training_data[storm_id]["y"].append(torch.tensor(vec[2:4]))
        elif storm_id in test_ids:
            test_data[storm_id]["X"].append(torch.tensor(vec[4:]))
            test_data[storm_id]["y"].append(torch.tensor(vec[2:4]))
        else:
            raise ZeroDivisionError("Invalid ID")

    # stack the tensors
    training_tensors = list(
        {"X": torch.stack(data["X"]), "y": torch.stack(data["y"])}
        for data in training_data.values()
    )
    test_tensors = list(
        {"X": torch.stack(data["X"]), "y": torch.stack(data["y"])}
        for data in test_data.values()
    )

    # make sure all tensors have dimension of 34 and 2
    for d in training_tensors:
        assert d["X"].shape[1] == 34
        assert d["y"].shape[1] == 2

    for d in test_tensors:
        assert d["X"].shape[1] == 34
        assert d["y"].shape[1] == 2

    train_loader = CustomDataLoader(training_tensors)
    test_loader = CustomDataLoader(test_tensors, shuffle=False)

    return train_loader, test_loader


def adjust_t_span(seq_len):
    return torch.linspace(0, 1, seq_len)


# Define the training function
def train(
    model,
    device,
    train_loader,
    test_loader,
    optimizer,
    num_epochs,
    log_path=None,
    loss_computation="one_run",
):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            t_span = adjust_t_span(x.shape[0])
            optimizer.zero_grad()
            t_eval, y_hat = model(torch.cat((x, y), dim=1), t_span)

            print(f"x.shape: {x.shape}")
            print(f"y.shape: {y.shape}")
            print(f"y_hat.shape: {y_hat.shape}")

            if loss_computation == "one_run":
                y_hat = y_hat[2]  # get the last output
                predicted_longitude = y_hat[:, 34]
                predicted_latitude = y_hat[:, 35]
                loss_longitude = nn.MSELoss()(predicted_longitude, y[:, 0])
                loss_latitude = nn.MSELoss()(predicted_latitude, y[:, 1])

                print(f"y[:, 0]: {y[:, 0]}")
                print(f"predicted_longitude: {predicted_longitude}")

                import sys
                sys.exit(0)

                loss = loss_longitude + loss_latitude

            elif loss_computation == "one_run_with_discount":
                y_hat = y_hat[-1]
                predicted_longitude = y_hat[:, 34]
                predicted_latitude = y_hat[:, 35]

                discount_factor = 0.99
                discount_matrix = torch.tensor([[discount_factor ** i for i in range(len(y_hat))]]).T
                discount_matrix = discount_matrix / discount_matrix.mean()

                # calculate loss and apply discount factor
                weighted_loss_longitude = nn.MSELoss(reduction="none")(predicted_longitude, y[:, 0]) * discount_matrix
                weighted_loss_latitude = (
                    nn.MSELoss(reduction="none")(predicted_latitude, y[:, 1])
                    * discount_matrix
                )

                loss_longitude = torch.mean(weighted_loss_longitude)
                loss_latitude = torch.mean(weighted_loss_latitude)

                loss = loss_longitude + loss_latitude

            elif loss_computation == "all":
                loss = 0
                for i in range(len(y_hat)):
                    predicted_longitude = y_hat[i][:, 34]
                    predicted_latitude = y_hat[i][:, 35]
                    loss_longitude = nn.MSELoss()(predicted_longitude, y[:, 0])
                    loss_latitude = nn.MSELoss()(predicted_latitude, y[:, 1])
                    loss += loss_longitude + loss_latitude
            else:
                raise NotImplementedError

            loss.backward(retain_graph=True)
            optimizer.step()
            total_loss += loss.item()

        if log_path is not None:
            with open(log_path, "a") as f:
                f.write(f"Epoch {epoch}, Average Loss {total_loss / len(train_loader)}\n")
        else:
            print(f"Epoch {epoch}, Average Loss {total_loss / len(train_loader)}")

        if epoch % 10 == 0:
            # run test and save model
            test(model, device, test_loader, log_path)
            # torch.save(model.state_dict(), f"saved_models/model_{epoch}.pth")


# Define the testing function
def test(model, device, test_loader, log_path=None):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            t_span = adjust_t_span(x.shape[0])
            t_eval, y_hat = model(torch.cat((x, y), dim=1), t_span)
            y_hat = y_hat[-1]  # get the last output
            predicted_longitude = y_hat[:, 34]
            predicted_latitude = y_hat[:, 35]
            loss_longitude = nn.MSELoss()(predicted_longitude, y[:, 0])
            loss_latitude = nn.MSELoss()(predicted_latitude, y[:, 1])
            loss = loss_longitude + loss_latitude
            total_loss += loss.item()
    
    if log_path is not None:
        with open(log_path, "a") as f:
            f.write(f"Test Loss: {total_loss / len(test_loader)}\n")
    else:
        print(f"Test Loss: {total_loss / len(test_loader)}")

if __name__ == "__main__":
    # Define NDE Model
    """
    Input is size 36: long, lat, wind, pressure, 
    and 32 vectorized image features
    Hidden features size: 10
    Output features sizel: 2 (long and lat)
    """
    model = NeuralODE(FlowNet(36), sensitivity="adjoint", solver="dopri5").to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Prepare data
    train_loader, test_loader = prepare_data()

    # Training and testing the model
    train(model, device, train_loader, test_loader, optimizer, num_epochs=2)
    test(model, device, test_loader)
