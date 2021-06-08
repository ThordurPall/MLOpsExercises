# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
# NOTE: Look into classifier
from classifier import Classifier
from sklearn.manifold import TSNE
from torchvision import datasets, transforms


@click.command()
@click.argument('trained_model_filepath',
                type=click.Path(), default='models/trained_model.pth')
@click.argument('data_folderpath',
                type=click.Path(), default='data')
@click.argument('figures_folderpath',
                type=click.Path(), default='reports/figures/')


def main(trained_model_filepath, data_folderpath, figures_folderpath):
    """ Extracts features just before the final classification layer of the network
        in TRAINED_MODEL_FILEPATH and does t-SNE embedding of the features for
        the MNIST test set located in DATA_FOLDERPATH. Additionally, plots data
        distribution of MNIST training set and stores in FIGURES_FOLDERPATH"""

    logger = logging.getLogger(__name__)
    logger.info('Creating predictions using a pre-trained neural network')

    # Load the trained model
    project_dir = Path(__file__).resolve().parents[2]
    model = Classifier()
    state_dict = torch.load(project_dir.joinpath(trained_model_filepath))
    model.load_state_dict(state_dict)
    model.eval()                     # Sets the model to evaluation mode

    # Load the test data
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = datasets.MNIST(project_dir.joinpath(data_folderpath),
                              download=False, train=False,
                              transform=transform)
    batch_size = 64
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=True)

    # Extracts features just before the final classification
    #  layer and do t-SNE embedding
    # Code from https://towardsdatascience.com/visualizing-feature-vectors
    # -embeddings-using-pca-and-t-sne-ef157cea3a42
    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        test_imgs = torch.zeros((0, 1, 28, 28), dtype=torch.float32)
        test_predictions = []
        test_targets = []
        test_embeddings = torch.zeros((0, 64), dtype=torch.float32)
        for x, y in test_loader:
            logits, embeddings = model(x)
            preds = torch.argmax(logits, dim=1)
            test_predictions.extend(preds.tolist())
            test_targets.extend(y.tolist())
            test_embeddings = torch.cat((test_embeddings, embeddings), 0)
            test_imgs = torch.cat((test_imgs, x), 0)
        test_imgs = np.array(test_imgs)
        test_embeddings = np.array(test_embeddings)
        test_targets = np.array(test_targets)
        test_predictions = np.array(test_predictions)

    # Create a two dimensional t-SNE projection of the embeddings
    tsne = TSNE(2, verbose=1)
    tsne_proj = tsne.fit_transform(test_embeddings)

    # Plot those points as a scatter plot and label
    # them based on the pred labels
    cmap = plt.cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8, 8))
    num_categories = 10
    for lab in range(num_categories):
        indices = test_predictions == lab
        ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1],
                   c=np.array(cmap(lab)).reshape(1, 4), label=lab, alpha=0.5)
    ax.legend(fontsize='large', markerscale=2)
    plt.show()

    # Plot the data distribution of the MNIST training set and
    #  compute moments for images
    train_set = datasets.MNIST(project_dir.joinpath(data_folderpath),
                               download=False, train=True,
                               transform=transform)
    mnist_data = train_set.data.float()

    f = plt.figure(figsize=(12, 8))
    plt.hist(train_set.targets.numpy(), density=False, bins=30)
    plt.ylabel('Count')
    plt.xlabel('Digit')
    plt.show()
    f.savefig(project_dir.joinpath(figures_folderpath).joinpath('MNIST_Training_Digit_Distribution.pdf'),
              bbox_inches='tight')
    logger.info(str("MNIST training images mean: {:.3f}.. ".format(mnist_data.mean().item()/255)) +
                str("and standard deviation: {:.3f}.. ".format(mnist_data.std().item()/255)))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()

