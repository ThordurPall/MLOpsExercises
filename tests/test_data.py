# -*- coding: utf-8 -*-
from src.data.make_dataset import make_dataset
import torch


class TestData:
    def test_make_dataset(self):
        """
        Test that the training and test MNIST data
        has the correct dimensions
        """
        train, test = make_dataset('tests/tests_temp')
        assert len(train) == 60000
        assert len(test) == 10000
        assert train.data.shape == torch.Size([60000, 28, 28])
        assert test.data.shape == torch.Size([10000, 28, 28])
        assert train.targets.shape == torch.Size([60000])
        assert test.targets.shape == torch.Size([10000])
        assert (train.targets.min() == torch.tensor(0)).item()
        assert (train.targets.max() == torch.tensor(9)).item()
        assert (test.targets.min() == torch.tensor(0)).item()
        assert (test.targets.max() == torch.tensor(9)).item()

