import torch
from NeuralODE import prepare_data, train, test
from torchdyn.core import NeuralODE
from model import FlowNet
import torch.optim as optim
import os

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare the data
training_loader, test_loader = prepare_data()

sensitivity_list = ["adjoint", "autograd"]
SOLVER_LIST = [
    "euler",
    "midpoint",
    "RungeKutta4",
    "DormandPrince45",
    "Tsitouras45",
    # "implicit_euler",
    # "AsynchronousLeapfrog",
]

for sensitivity in sensitivity_list:
    for solver in SOLVER_LIST:
        # if "log/" does not exist, create it
        if not os.path.exists("logs"):
            os.makedirs("logs")

        log_path = f"logs/{sensitivity}_{solver}_500.log"

        if os.path.exists(log_path):
            print(f"Skipping training with sensitivity: {sensitivity} and solver: {solver}")
            continue

        print(f"Training with sensitivity: {sensitivity} and solver: {solver}")

        # Define NDE Model
        model = NeuralODE(FlowNet(36), sensitivity=sensitivity, solver=solver).to(
            device
        )
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Train the model
        train(model, device, training_loader, test_loader, optimizer, num_epochs=500, log_path=log_path)

        # Test the model
        test(model, device, test_loader, log_path=log_path)