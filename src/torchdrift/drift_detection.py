# -*- coding: utf-8 -*-
import copy
import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import Isomap
from torchvision import datasets, transforms

import torchdrift
from src.models.classifier import Classifier
from torchdrift.detectors.mmd import (ExpKernel, GaussianKernel,
                                      RationalQuadraticKernel)


@click.command()
@click.argument("data_filepath", type=click.Path(), default="data")
@click.argument(
    "trained_model_filepath", type=click.Path(), default="models/trained_model.pth"
)
def drift_detection(data_filepath, trained_model_filepath):
    """ Implements drift detection with the MNIST project """
    logger = logging.getLogger(__name__)
    logger.info("Drift detection with the MNIST data set")

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
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )

    # Plot example images
    N = 6
    images, labels = iter(train_loader).next()
    images_corrupt = corruption_function(images)
    plt.figure(figsize=(15, 5))
    for i in range(N):
        # Plot the original MNIST image
        plt.subplot(2, N, i + 1)
        plt.title(labels[i].item())
        plt.imshow(images[i][0], cmap="gray")
        # plt.imshow(images[i].permute(1, 2, 0))
        plt.xticks([])
        plt.yticks([])

        # Plot the MNIST image with Gaussian blur
        plt.subplot(2, N, i + 1 + N)
        plt.title(labels[i].item())
        plt.imshow(images_corrupt[i][0], cmap="gray")
        # plt.imshow(images[i].permute(1, 2, 0))
        plt.xticks([])
        plt.yticks([])
    plt.show()

    # Load the trained model
    model = Classifier()
    project_dir = Path(__file__).resolve().parents[2]
    state_dict = torch.load(project_dir.joinpath(trained_model_filepath))
    model.load_state_dict(state_dict)
    model.return_features = True
    # From: https://torchdrift.org/notebooks/drift_detection_on_images.html
    # feature_extractor = copy.deepcopy(model)
    # feature_extractor.classifier = torch.nn.Identity()

    # The drift detector - Using the Kernel MMD drift detector on
    # the features extracted by the pretrained model
    gaussian_kernel = torchdrift.detectors.KernelMMDDriftDetector(
        kernel=GaussianKernel()
    )
    exp_kernel = torchdrift.detectors.KernelMMDDriftDetector(kernel=ExpKernel())
    rational_quadratic_kernel = torchdrift.detectors.KernelMMDDriftDetector(
        kernel=RationalQuadraticKernel()
    )

    kernel_names = ["GaussianKernel", "ExpKernel", "RationalQuadraticKernel"]
    scores_real, p_vals_real, scores_corrupt, p_vals_corrupt = [], [], [], []
    for i, kernel in enumerate(
        [gaussian_kernel, exp_kernel, rational_quadratic_kernel]
    ):
        print(i)
        kernel_name = kernel_names[i]
        drift_detector = kernel

        # Fit the drift detector using training data
        torchdrift.utils.fit(train_loader, model, drift_detector, num_batches=20)

        # Test the output on actual training inputs
        features = model(images)
        score = drift_detector(features)
        p_val = drift_detector.compute_p_value(features)
        scores_real.append(score)
        p_vals_real.append(p_val)
        print(p_val)

        # Visualize the two distribution to detemine if the look close
        mapper = Isomap(n_components=2)
        base_embedded = mapper.fit_transform(drift_detector.base_outputs)
        features_embedded = mapper.transform(features.detach().numpy())
        f = plt.figure(figsize=(12, 8))
        plt.scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c="r")
        plt.scatter(features_embedded[:, 0], features_embedded[:, 1], s=4)
        plt.title(f"{kernel_name} real data, score {score:.2f} p-value {p_val:.2f}")
        # plt.show()
        f.savefig(kernel_name + "_Distributions_Real_Data.pdf", bbox_inches="tight")

        # Test the output on actual corrupt training inputs
        features = model(images_corrupt)
        score = drift_detector(features)
        p_val = drift_detector.compute_p_value(features)
        scores_corrupt.append(score)
        p_vals_corrupt.append(p_val)

        # Visualize the two distribution to detemine if the look close
        features_embedded = mapper.transform(features.detach().numpy())
        f = plt.figure(figsize=(12, 8))
        plt.scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c="r")
        plt.scatter(features_embedded[:, 0], features_embedded[:, 1], s=4)
        plt.title(f"{kernel_name} corrupt data, score {score:.2f} p-value {p_val:.2f}")
        # plt.show()
        f.savefig(kernel_name + "_Distributions_Corrupt_Data.pdf", bbox_inches="tight")
    print(scores_real)
    print(p_vals_real)
    print(scores_corrupt)
    print(p_vals_corrupt)


def corruption_function(x: torch.Tensor):
    """ Applies the Gsaussian blur to x """
    return torchdrift.data.functional.gaussian_blur(x, severity=5)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    drift_detection()

