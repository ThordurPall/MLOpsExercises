# -*- coding: utf-8 -*-
from src.models.classifier import Classifier
import torch
import pytest


class TestModel:
    def test_classifier(self):
        """
        Test that given input with shape [X, 1, 28, 28] that the output of
        the model has shape [X, 10] and the the sum of the exponentials
        is one for each input in the batch
        """
        X = 64
        model = Classifier()
        x = model.forward(torch.rand(X, 1, 28, 28))

        # Check that the output from the forward has the correct shape
        assert x.shape == torch.Size([X, 10])

        # Test that the sum of the exponentials of the logits is one
        assert X == int(round(torch.exp(x).sum().item()))
    
    def test_classifier_exception_4D(self):
        model = Classifier()

        # Chack that there are batch, channel, width and height dimensions
        with pytest.raises(ValueError, match='Expected input to be a 4D tensor'):
            x = model.forward(torch.rand(1, 28, 28))

    @pytest.mark.parametrize("test_input", 
                             [torch.rand(64, 3, 28, 28),
                              torch.rand(64, 1, 27, 28),
                              torch.rand(64, 1, 28, 27)])                
    def test_classifier_exception_sample_shape(self, test_input):
        model = Classifier()
        
        # Check that the number of channels is one and width=height=28
        with pytest.raises(ValueError, match='Expected each sample to have the shape 1x28x28'):
            x = model.forward(test_input)

