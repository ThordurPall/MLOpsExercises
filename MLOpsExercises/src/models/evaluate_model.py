# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import torch
from classifier import Classifier
from torchvision import datasets, transforms


@click.command()
@click.argument('data_filepath', type=click.Path(), default='data')
@click.argument('trained_model_filepath', type=click.Path(),
                default='models/trained_model.pth')

def main(data_filepath, trained_model_filepath):
    """ Evaluates the neural network using MNIST test data """
    logger = logging.getLogger(__name__)
    logger.info('Evaluating a neural network using MNIST test data')

    # Load the trained model
    model = Classifier()
    project_dir = Path(__file__).resolve().parents[2]
    state_dict = torch.load(project_dir.joinpath(trained_model_filepath))
    model.load_state_dict(state_dict)

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)), ])

    # Load the test data
    test_set = datasets.MNIST(project_dir.joinpath(data_filepath),
                              download=False, train=False,
                              transform=transform)
    batch_size = 64
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=True)

    # Evaluate test performance
    test_correct = 0

    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        model.eval()     # Sets the model to evaluation mode

        # Run through all the test points
        for images, labels in test_loader:
            # Forward pass
            log_ps = model(images)
            ps = torch.exp(log_ps)

            # Keep track of how many are correctly classified
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_correct += equals.type(torch.FloatTensor).sum().item()
        test_accuracy = test_correct/len(test_set)

    logger.info(str("Test Accuracy: {:.3f}".format(test_accuracy)))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()

