"""
Adapted from
https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb

A simple implementation of Gaussian MLP Encoder and Decoder trained on MNIST
"""
import os
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Encoder, Decoder, Model
from torch.optim import Adam
import random
import numpy as np
import hydra
from omegaconf import OmegaConf
import logging


@hydra.main(config_path="conf", config_name="config")
def vae_mnist_hydra(config):
    log = logging.getLogger(__name__)
    log.info(f"Current configuration: \n {OmegaConf.to_yaml(config)}")
    print(config)
    hype = config.hyperparameters  # Current set of hyperparameters
    print(hype)
    cuda = False
    DEVICE = torch.device("cuda" if cuda else "cpu")

    # https://pytorch.org/docs/stable/notes/randomness.html
    # Use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA):
    seed = hype["seed"]
    torch.manual_seed(seed)

    # For custom operators, you might need to set python seed as well:
    random.seed(seed)

    # If you or any of the libraries you are using rely on NumPy, you can seed the global NumPy RNG with:
    np.random.seed(seed)

    # Disabling the benchmarking feature with torch.backends.cudnn.benchmark = False
    # causes cuDNN to deterministically select an algorithm, possibly at the cost of reduced performance.
    torch.backends.cudnn.benchmark = False

    # Avoiding nondeterministic algorithms
    torch.use_deterministic_algorithms(True)

    # Data loading
    mnist_transform = transforms.Compose([transforms.ToTensor()])

    breakpoint()
    dataset_path = hype["dataset_path"]
    train_dataset = MNIST(
        dataset_path, transform=mnist_transform, train=True, download=True
    )
    test_dataset = MNIST(
        dataset_path, transform=mnist_transform, train=False, download=True
    )

    # Question: What about the shuffle here for reproducibility?
    # Or is that also taken care of with the seed?
    # or do it like https://pytorch.org/docs/stable/notes/randomness.html ?
    batch_size = hype["batch_size"]
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    x_dim = hype["x_dim"]
    hidden_dim = hype["hidden_dim"]
    latent_dim = hype["latent_dim"]
    encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)

    model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)

    BCE_loss = nn.BCELoss()

    def loss_function(x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.binary_cross_entropy(
            x_hat, x, reduction="sum"
        )
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return reproduction_loss + KLD

    lr = hype["lr"]
    optimizer = Adam(model.parameters(), lr=lr)

    log.info("Start training VAE...")
    model.train()
    epochs = hype["epochs"]
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()
        log.info(
            f"Epoch {epoch+1} complete! Average Loss: {overall_loss / (batch_idx*batch_size)}"
        )
    log.info("Finish!!")

    # save weights
    torch.save(model, f"{os.getcwd()}/trained_model.pt")

    # Generate reconstructions
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)
            x_hat, _, _ = model(x)
            break

    save_image(x.view(batch_size, 1, 28, 28), "orig_data.png")
    save_image(x_hat.view(batch_size, 1, 28, 28), "reconstructions.png")

    # Generate samples
    with torch.no_grad():
        noise = torch.randn(batch_size, latent_dim).to(DEVICE)
        generated_images = decoder(noise)
    save_image(generated_images.view(batch_size, 1, 28, 28), "generated_sample.png")


if __name__ == "__main__":
    vae_mnist_hydra()
