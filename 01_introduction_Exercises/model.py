from torch import nn
import torch.nn.functional as F


class Classifier(nn.Module):
    """
    A class used build a neural network for classification of MNIST digits

    ...

    Methods
    -------
    forward()
        Forward pass through the network, returns the output logits
    """
    
    def __init__(self):
        super().__init__()
        # Define fully connected layers
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        """ Forward pass through the network, returns the output logits """

        # Flattening input tensor except for the minibatch dimension
        x = x.view(x.shape[0], -1)

        # Fully connected layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))

        # Output so no dropout here
        x = F.log_softmax(self.fc4(x), dim=1)
        return x
