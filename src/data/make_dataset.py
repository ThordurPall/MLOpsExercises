# -*- coding: utf-8 -*-
import logging
from pathlib import Path

from azureml.core import Run
from torchvision import datasets, transforms


def make_dataset(file_path, use_azure=False):
    """ Downloads and stores the MNIST training and test data into
        raw data (../file_path/MNIST/raw) and into cleaned data ready
        to be analyzed (../file_path/MNIST/processed). The function
        additionally returns the training and test datasets
    """
    
    if use_azure:
        # Get the experiment run context. That is, retrieve the experiment
        # run context when the script is run
        run = Run.get_context()

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)), ])

    logger = logging.getLogger(__name__)
    logger.info('Download and store the training and test data')
    project_dir = Path(__file__).resolve().parents[2]
    train = datasets.MNIST(project_dir.joinpath(file_path),
                           download=True, train=True,
                           transform=transform)
    test = datasets.MNIST(project_dir.joinpath(file_path),
                          download=True, train=False,
                          transform=transform)

    if use_azure:
        # Complete the run
        run.complete()
        print('Completed running the make dataset expriment')
    return train, test

