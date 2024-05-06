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

# if "log/" does not exist, create it
if not os.path.exists("logs"):
    os.makedirs("logs")
if not os.path.exists("logs/combined"):
    os.makedirs("logs/combined")

for sensitivity in sensitivity_list:
    for forward_solver in SOLVER_LIST:
        for backward_solver in SOLVER_LIST:
            if forward_solver == backward_solver:
                continue

            log_path = f"logs/combined/{sensitivity}_{forward_solver}_{backward_solver}.log"

            if os.path.exists(log_path):
                print(f"Skipping training with sensitivity: {sensitivity} and forward solver: {forward_solver} and backward solver: {backward_solver}")
                continue

            print(f"Training with sensitivity: {sensitivity} and forward solver: {forward_solver} and backward solver: {backward_solver}")

            # Define NDE Model
            # atol and rtol are the absolute and relative tolerances for the ODE solver
            model = NeuralODE(FlowNet(36), sensitivity=sensitivity, solver=forward_solver, solver_adjoint=backward_solver, atol=1e-3, rtol=1e-3).to(
                device
            )
            optimizer = optim.Adam(model.parameters(), lr=0.01)

            # Train the model
            train(model, device, training_loader, test_loader, optimizer, num_epochs=500, log_path=log_path)

            # Test the model
            test(model, device, test_loader, log_path=log_path)
