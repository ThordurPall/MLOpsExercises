# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import argparse
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from fashion_trainer import FashionCNN


def evaluate(batch_size=100):
    """ Evaluates the neural network using FashionMNIST test data """
    logger = logging.getLogger(__name__)
    logger.info('Evaluating a neural network using FashionMNIST test data')

    # Check if there is a GPU available
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the test data    
    test_set = FashionMNIST('', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test_loader = DataLoader(test_set, batch_size=batch_size)
    logger.info('Test data of size: ' + str(len(test_set)) + ', batch size: ' + str(batch_size))
    
    # Load the trained model and transfering model to GPU if available
    model = FashionCNN()
    state_dict = torch.load('trained_model.pth', map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    # Doing five repetitions to get empirical average and standard deviation
    res = []
    for _ in range(5):
        start = time.time()
        # Evaluate test performance
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            model.eval()     # Sets the model to evaluation mode
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predicted = torch.max(outputs, 1)[1]
                c = (predicted == labels).squeeze()
        end = time.time()
        res.append(end - start)
    res = np.array(res)
    print(res)
    return np.mean(res), np.std(res)
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-make_errorbar_plot', action='store_true')
    args = parser.parse_args()
    evaluate()

    if args.make_errorbar_plot:
        means, stds = [], []
        batch_sizes = list(range(10, 510, 10))
        print(batch_sizes)
        for batch_size in batch_sizes:
            t_mean, t_std = evaluate(batch_size)
            means.append(t_mean)
            stds.append(t_std)
        print(means)
        print(stds)
        f = plt.figure(figsize=(12, 8))
        plt.errorbar(batch_sizes, means, yerr=stds)
        plt.xlabel('Batch size')
        plt.ylabel('Time')
        plt.plot()
        f.savefig('./errorbar_plot_batch_size.pdf', bbox_inches='tight')
    

