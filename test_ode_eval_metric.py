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

SENSITIVITY_LIST = ["adjoint", "autograd"]
SOLVER_LIST = [
    "DormandPrince45",
]
EVAL_METRIC_LIST = [
    "one_run",
    "one_run_with_discount",
    "all",
    "all_with_discount",
    "next_only"
]

# if "log/" does not exist, create it
if not os.path.exists("logs"):
    os.makedirs("logs")
if not os.path.exists("logs/eval_metric"):
    os.makedirs("logs/eval_metric")

for sensitivity in SENSITIVITY_LIST:
    for solver in SOLVER_LIST:
        for eval_metric in EVAL_METRIC_LIST:
            log_path = f"logs/eval_metric/{eval_metric}_{sensitivity}_{solver}_400.log"

            if os.path.exists(log_path):
                print(f"Skipping training with eval_metric: {eval_metric} and sensitivity: {sensitivity} and solver: {solver}")
                continue

            print(f"Training with eval_metric: {eval_metric} and sensitivity: {sensitivity} and solver: {solver}")

            # Define NDE Model
            model = NeuralODE(FlowNet(36), sensitivity=sensitivity, solver=solver).to(
                device
            )
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

            # Train the model
            train(model, device, training_loader, test_loader, optimizer, scheduler, num_epochs=400, log_path=log_path, eval_metric=eval_metric)

            # Test the model
            test(model, device, test_loader, log_path=log_path)
