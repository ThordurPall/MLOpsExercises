# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
from torchvision import datasets, transforms


@click.command()
@click.argument('data_filepath',  type=click.Path())

def main(data_filepath):
    """ Downloads and stores the MNIST training and test data into
        raw data (../DATA_FILEPATH/MNIST/raw) and into cleaned data ready
        to be analyzed (../DATA_FILEPATH/MNIST/processed)
    """

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)), ])

    logger = logging.getLogger(__name__)
    logger.info('Download and store the training and test data')
    project_dir = Path(__file__).resolve().parents[2]
    _ = datasets.MNIST(project_dir.joinpath(data_filepath),
                       download=True, train=True,
                       transform=transform)
    _ = datasets.MNIST(project_dir.joinpath(data_filepath),
                       download=True, train=False,
                       transform=transform)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()

