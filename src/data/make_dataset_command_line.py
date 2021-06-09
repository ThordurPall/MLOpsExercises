# -*- coding: utf-8 -*-
import click
import logging
from src.data.make_dataset import make_dataset

@click.command()
@click.argument('data_filepath',  type=click.Path(), default='data')

def make_dataset_command_line(data_filepath):
    """ Downloads and stores the MNIST training and test data into
        raw data (../DATA_FILEPATH/MNIST/raw) and into cleaned data ready
        to be analyzed (../DATA_FILEPATH/MNIST/processed).
    """
    _, _ = make_dataset(data_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    make_dataset_command_line()
