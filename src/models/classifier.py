import torch.nn.functional as F
from torch import nn


class Classifier(nn.Module):
    """
    A class used build a neural network for classification of MNIST digits

    ...

    Methods
    -------
    forward()
        Forward pass through the network, returns the output logits
    """

    def __init__(self, dropout_p=0.25):
        super().__init__()
        # Define fully connected layers
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.return_features = False

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        """ Forward pass through the network, returns the output logits """

        # Chack that there are batch, channel, width and height dimensions
        if x.ndim != 4:
            raise ValueError("Expected input to be a 4D tensor")

        # Check that the number of channals is one and width=height=28
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError("Expected each sample to have the shape 1x28x28")

        # Flattening input tensor except for the minibatch dimension
        x = x.view(x.shape[0], -1)

        # Fully connected layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))

        # Store the abstract features at this point
        features = x
        if self.return_features:
            return features
        # Output so no dropout here
        x = F.log_softmax(self.fc4(x), dim=1)
        return x  # , features
