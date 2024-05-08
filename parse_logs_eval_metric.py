import os
import pandas as pd


results = {}


for file in os.listdir("logs/eval_metric"):
    if not file.endswith(".log"):
        continue

    # parse file name
    file_wo_ext = file.split(".")[0]

    if "one_run_with_discount" in file_wo_ext:
        eval_metric = "one_run_with_discount"
    elif "all_with_discount" in file_wo_ext:
        eval_metric = "all_with_discount"
    elif "one_run" in file_wo_ext:
        eval_metric = "one_run"
    elif "all" in file_wo_ext:
        eval_metric = "all"
    elif "next_only" in file_wo_ext:
        eval_metric = "next_only"
    else:
        raise ValueError(f"Unknown file: {file}")

    if "adjoint" in file_wo_ext:
        sensitivity = "adjoint"
    elif "autograd" in file_wo_ext:
        sensitivity = "autograd"
    else:
        raise ValueError(f"Unknown file: {file}")

    training_losses = []
    test_losses_one_run = []
    test_losses_all = []
    test_losses_next_only = []

    with open(f"logs/eval_metric/{file}", "r") as f:
        for line in f:
            if "Test Loss one_run" in line:
                test_losses_one_run.append("{:0.3f}".format(float(line.split(": ")[1])))
            elif "Test Loss all" in line:
                test_losses_all.append("{:0.3f}".format(float(line.split(": ")[1])))
            elif "Test Loss next_only" in line:
                test_losses_next_only.append(
                    "{:0.3f}".format(float(line.split(": ")[1]))
                )
            elif "Average Loss" in line:
                training_losses.append(float(line.split("Average Loss ")[1]))
            else:
                raise ValueError(f"Unknown line: {line}")

    # find best test loss and its index for each eval metric
    best_test_loss_one_run = min(test_losses_one_run)
    best_test_loss_all = min(test_losses_all)
    best_test_loss_next_only = min(test_losses_next_only)
    best_training_loss = min(training_losses)

    best_test_loss_index_one_run = test_losses_one_run.index(best_test_loss_one_run) * 5
    best_test_loss_index_all = test_losses_all.index(best_test_loss_all) * 5
    best_test_loss_index_next_only = (
        test_losses_next_only.index(best_test_loss_next_only) * 5
    )
    best_training_loss_index = training_losses.index(best_training_loss) * 5

    # results[f"{eval_metric}, {sensitivity}"] = [(best_test_loss_one_run, best_test_loss_index_one_run), (best_test_loss_all, best_test_loss_index_all), (best_test_loss_next_only, best_test_loss_index_next_only)]
    results[f"{eval_metric}, {sensitivity}"] = {
        "one_run": (best_test_loss_one_run, best_test_loss_index_one_run),
        "all": (best_test_loss_all, best_test_loss_index_all),
        "next_only": (best_test_loss_next_only, best_test_loss_index_next_only),
        # "best_training_loss": (best_training_loss, best_training_loss_index),
    }

# order results by best test loss
results = dict(sorted(results.items(), key=lambda item: item[1]["one_run"][0]))

df = pd.DataFrame(results).transpose()
print(df)
