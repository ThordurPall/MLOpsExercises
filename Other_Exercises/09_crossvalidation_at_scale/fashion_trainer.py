"""
Credit to: https://www.kaggle.com/pankajj/fashion-mnist-with-pytorch-93-accuracy
"""
import logging
import sys

import optuna
import plotly.io as pio
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def output_label(label):
    output_mapping = {
        0: "T-shirt/Top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }
    input = label.item() if type(label) == torch.Tensor else label
    return output_mapping[input]


class FashionCNN(nn.Module):
    def __init__(self, n_out_sec_last=120, dropout_p=0.25, use_batch_norm=True):
        super(FashionCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.drop = nn.Dropout2d(dropout_p)
        self.fc2 = nn.Linear(in_features=600, out_features=n_out_sec_last)
        self.fc3 = nn.Linear(in_features=n_out_sec_last, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out


def objective(trial):
    # Suggest a set of hyperparameters
    lr = trial.suggest_loguniform("lr", 1e-6, 1.0)
    print(lr)
    n_out_sec_last = trial.suggest_int("n_out_sec_lasts", 100, 500)
    dropout_p = trial.suggest_uniform("dropout_p", 0.2, 0.8)
    batch_size = trial.suggest_int("batch_size", 32, 512)
    # use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])

    # Load the training and test sets
    train_set = FashionMNIST(
        "",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_set = FashionMNIST(
        "",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size)
    test_loader = DataLoader(train_set, batch_size=100)

    # Initialize the model
    model = FashionCNN(n_out_sec_last=n_out_sec_last, dropout_p=dropout_p)
    model.to(device)

    # Define a loss function and an optimizer
    error = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Lists for visualization of loss and accuracy
    loss_list = []
    iteration_list = []
    accuracy_list = []

    # Lists for knowing classwise accuracy
    predictions_list = []
    labels_list = []

    # Run the training and validation
    num_epochs = 10
    for epoch in range(num_epochs):
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Transfering images and labels to GPU if available
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = error(outputs, labels)

            # Initializing a gradient as 0 so there is no mixing of gradient among the batches
            optimizer.zero_grad()

            # Propagating the error backward
            loss.backward()

            # Optimizing the parameters
            optimizer.step()

            # Testing the model
            count = epoch * len(train_loader) + batch_idx
            if not (count % 50):  # It's same as "if count % 50 == 0"
                total = 0
                correct = 0

                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    labels_list.append(labels)

                    outputs = model(images)

                    predictions = torch.max(outputs, 1)[1].to(device)
                    predictions_list.append(predictions)
                    correct += (predictions == labels).sum()

                    total += len(labels)

                accuracy = correct * 100 / total
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)

            if not (count % 500):
                print(
                    "Iteration: {}, Loss: {}, Accuracy: {}%".format(
                        count, loss.data, accuracy
                    )
                )

        # To turn on the pruning feature, you need to call report()
        # and should_prune() after each step of the iterative training.
        # report() periodically monitors the intermediate objective values.
        # should_prune() decides termination of the trial that does
        # not meet a predefined condition.

        # Report intermediate objective value
        # Question: Should this be located here or outside after the whole epoch?
        # s.t. trial.report(accuracy, epoch)
        # What hyperparameter should one be especially careful about when using pruning?
        # Would epochs make sense as a hyperparameter as well?
        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            print("-----------------------PRUNING-----------------------")
            raise optuna.TrialPruned()

    print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))
    class_correct = [0.0 for _ in range(10)]
    total_correct = [0.0 for _ in range(10)]

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = torch.max(outputs, 1)[1]
            c = (predicted == labels).squeeze()

            for i in range(100):
                label = labels[i]
                class_correct[label] += c[i].item()
                total_correct[label] += 1

    for i in range(10):
        print(
            "Accuracy of {}: {:.2f}%".format(
                output_label(i), class_correct[i] * 100 / total_correct[i]
            )
        )
    return accuracy


if __name__ == "__main__":
    prune = True

    if prune:
        # Pruners automatically stop unpromising trials at the early
        # stages of the training (a.k.a., automated early-stopping)
        print("Using median pruning")

        # Add stream handler of stdout to show the messages
        optuna.logging.get_logger("optuna").addHandler(
            logging.StreamHandler(sys.stdout)
        )

        # Set up the median stopping rule as the pruning condition
        study = optuna.create_study(
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
            direction="maximize",
        )
    else:
        # call the optimizer
        study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=60)

    # Plot optimization history of all trials in a study
    fig = optuna.visualization.plot_optimization_history(study)
    pio.write_image(fig, "optuna_optimization_history.pdf")

    # Plot intermediate values of all trials in a study -
    # Visualize the learning curves of the trials
    fig = optuna.visualization.plot_intermediate_values(study)
    pio.write_image(fig, "optuna_accuracy_curve.pdf")

    # Plot the high-dimensional parameter relationships in a study
    fig = optuna.visualization.plot_parallel_coordinate(study)
    pio.write_image(fig, "optuna_high_dim_par_relationships.pdf")

    # Plot the parameter relationship as contour plot in a study
    fig = optuna.visualization.plot_contour(study)
    pio.write_image(fig, "optuna_contour_par_relationships.pdf")

    # Visualize individual hyperparameters as slice plot
    fig = optuna.visualization.plot_slice(study)
    pio.write_image(fig, "optuna_individual_pars.pdf")

    # Plot hyperparameter importances
    fig = optuna.visualization.plot_param_importances(study)
    pio.write_image(fig, "optuna_individual_par_importance.pdf")

    # Plot the objective value EDF (empirical distribution function) of a study
    fig = optuna.visualization.plot_edf(study)
    pio.write_image(fig, "optuna_edf.pdf")
    fig.show()
