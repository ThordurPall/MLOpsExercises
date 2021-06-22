# -*- coding: utf-8 -*-
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
import torch
from torch import nn, optim
from torchvision import datasets, transforms

from src.models.classifier import Classifier


def train_model(
    data_filepath,
    trained_model_filepath,
    training_statistics_filepath,
    training_figures_filepath,
    epochs,
    lr,
    dropout_p=0.25,
    batch_size=64,
    trial=None,
):
    """ Trains the neural network using MNIST training data and
        returns a dictionary with the training and validation 
        losses and accuracies """

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    logger.info("Training a neural network using MNIST training data")

    # Create the network and define the loss function and optimizer
    model = Classifier(dropout_p=dropout_p)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Define a transform to normalize the data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]
    )

    # Divide the training dataset into two parts:
    #  a training set and a validation set
    project_dir = Path(__file__).resolve().parents[2]
    train_set = datasets.MNIST(
        project_dir.joinpath(data_filepath),
        download=False,
        train=True,
        transform=transform,
    )
    train_n = int(0.7 * len(train_set))
    val_n = len(train_set) - train_n
    train_set, val_set = torch.utils.data.random_split(train_set, [train_n, val_n])
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True
    )

    # Implement the training loop
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    for e in range(epochs):
        train_loss = 0
        train_correct = 0

        for images, labels in train_loader:
            # Set model to training mode and zero
            #  gradients since they accumulated
            model.train()
            optimizer.zero_grad()

            # Make a forward pass through the network to get the logits
            log_ps = model(images)
            ps = torch.exp(log_ps)

            # Use the logits to calculate the loss
            loss = criterion(log_ps, labels)
            train_loss += loss.item()

            # Perform a backward pass through the network
            #  to calculate the gradients
            loss.backward()

            # Take a step with the optimizer to update the weights
            optimizer.step()

            # Keep track of how many are correctly classified
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            train_correct += equals.type(torch.FloatTensor).sum().item()
        else:
            # Compute validattion loss and accuracy
            val_loss = 0
            val_correct = 0

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                model.eval()  # Sets the model to evaluation mode
                for images, labels in val_loader:
                    # Forward pass and compute loss
                    log_ps = model(images)
                    ps = torch.exp(log_ps)
                    val_loss += criterion(log_ps, labels).item()

                    # Keep track of how many are correctly classified
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    val_correct += equals.type(torch.FloatTensor).sum().item()

            # Store and print losses and accuracies
            train_losses.append(train_loss / len(train_loader))
            train_accuracies.append(train_correct / len(train_set))
            val_losses.append(val_loss / len(val_loader))
            val_accuracies.append(val_correct / len(val_set))

            logger.info(
                str("Epoch: {}/{}.. ".format(e + 1, epochs))
                + str("Training Loss: {:.3f}.. ".format(train_losses[-1]))
                + str("Training Accuracy: {:.3f}.. ".format(train_accuracies[-1]))
                + str("Validation Loss: {:.3f}.. ".format(val_losses[-1]))
                + str("Validation Accuracy: {:.3f}.. ".format(val_accuracies[-1]))
            )

        if trial:
            # Report intermediate objective value
            trial.report(val_accuracies[-1], e)

            # Handle pruning based on the intermediate value
            if trial.should_prune():
                print("-----------------------PRUNING-----------------------")
                raise optuna.TrialPruned()

    # Save the trained network
    torch.save(model.state_dict(), project_dir.joinpath(trained_model_filepath))

    # Save the training and validation losses and accuracies as a dictionary
    train_val_dict = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
    }

    with open(
        project_dir.joinpath(training_statistics_filepath).joinpath(
            "train_val_dict.pickle"
        ),
        "wb",
    ) as f:
        # Pickle the 'train_val_dict' dictionary using
        #  the highest protocol available
        pickle.dump(train_val_dict, f, pickle.HIGHEST_PROTOCOL)

    # Plot the training loss curve
    f = plt.figure(figsize=(12, 8))
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch number")
    plt.ylabel("Loss")
    plt.legend()
    f.savefig(
        project_dir.joinpath(training_figures_filepath).joinpath("Training_Loss.pdf"),
        bbox_inches="tight",
    )

    # Plot the training accuracy curve
    f = plt.figure(figsize=(12, 8))
    plt.plot(train_accuracies, label="Training accuracy")
    plt.plot(val_accuracies, label="Validation accuracy")
    plt.xlabel("Epoch number")
    plt.ylabel("Accuracy")
    plt.legend()
    f.savefig(
        project_dir.joinpath(training_figures_filepath).joinpath(
            "Training_Accuracy.pdf"
        ),
        bbox_inches="tight",
    )

    return train_val_dict


def optuna_objective(
    trial,
    data_filepath,
    trained_model_filepath,
    training_statistics_filepath,
    training_figures_filepath,
    epochs,
):
    # Suggest a set of hyperparameters
    lr = trial.suggest_loguniform("lr", 1e-6, 1.0)
    print(lr)
    # n_out_sec_last = trial.suggest_int("n_out_sec_lasts", 100, 500)
    dropout_p = trial.suggest_uniform("dropout_p", 0.2, 0.8)
    batch_size = trial.suggest_int("batch_size", 32, 512)

    train_val_dict = train_model(
        data_filepath,
        trained_model_filepath,
        training_statistics_filepath,
        training_figures_filepath,
        epochs,
        lr=lr,
        dropout_p=dropout_p,
        batch_size=batch_size,
        trial=trial,
    )
    return train_val_dict["val_accuracies"][-1]

