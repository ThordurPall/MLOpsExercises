# -*- coding: utf-8 -*-
import click
import logging
from src.models.train_model_azure import train_model

@click.command()
@click.argument('data_filepath', type=click.Path(), default='data')
@click.argument('trained_model_filepath', type=click.Path(),
                default='models/trained_model.pth')
@click.argument('training_statistics_filepath', type=click.Path(),
                default='data/processed/')
@click.argument('training_figures_filepath', type=click.Path(),
                default='reports/figures/')
@click.option('-e', '--epochs', type=int, default=2,
              help='Number of training epochs (default=30)')
@click.option('-lr', '--learning_rate', type=float, default=0.001,
              help='Learning rate for the PyTorch optimizer (default=0.001)')

def train_model_command_line(data_filepath, trained_model_filepath,
                             training_statistics_filepath,
                             training_figures_filepath,
                             epochs, learning_rate):
    """ Trains the neural network using MNIST training data """
    _ = train_model(data_filepath, trained_model_filepath,
                    training_statistics_filepath,
                    training_figures_filepath, epochs, learning_rate)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    train_model_command_line()
