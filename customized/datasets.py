"""
Contains all Dataset objects customized to the data.
"""
__author__ = 'ryanquinnnelson'

import torch
from torch.utils.data import Dataset


class TrainValDataset(Dataset):
    """
    Define a Dataset for training and validation data.
    """

    # load the dataset
    def __init__(self, x):
        self.X = x

    # get number of items/rows in dataset
    def __len__(self):
        return len(self.X)

    # get row item at some index
    def __getitem__(self, index):
        x = torch.FloatTensor(self.X[index])
        return x
