"""
All things related to datasets that require customized classes for Training, Validation, and Testing data.
"""
__author__ = 'ryanquinnnelson'

import logging

import numpy as np
import torchvision


class NumericalDatasetHandler:
    def __init__(self, data_dir, train_data, val_data, train_class, val_class):
        logging.info('Initializing numerical dataset handler...')

        self.data_dir = data_dir
        self.train_data = train_data
        self.val_data = val_data
        self.train_class = train_class
        self.val_class = val_class

    def get_train_dataset(self):
        """
        Load training data into memory and initialize the Dataset object.
        :return: Dataset
        """

        # load data
        # data = np.load(self.train_data, allow_pickle=True)
        # logging.info(f'Loaded {len(data)} training records.')

        # initialize dataset
        # dataset = self.train_class(data)

        # TODO: Replace after MNIST
        dataset = torchvision.datasets.MNIST(self.train_data,
                                             transform=torchvision.transforms.ToTensor(),
                                             download=False)

        return dataset

    def get_val_dataset(self):
        """
        Load validation data into memory and initialize the Dataset object.
        :return: Dataset
        """

        # load data
        # data = np.load(self.val_data, allow_pickle=True)
        # logging.info(f'Loaded {len(data)} validation records.')

        # initialize dataset
        # dataset = self.val_class(data)

        # TODO: Replace after MNIST
        dataset = torchvision.datasets.MNIST(self.val_data,
                                             transform=torchvision.transforms.ToTensor(),
                                             download=False)

        return dataset
