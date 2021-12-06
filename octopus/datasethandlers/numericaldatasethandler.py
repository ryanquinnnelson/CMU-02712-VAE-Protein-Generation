"""
All things related to datasets that require customized classes for Training and Validation data.
"""
__author__ = 'ryanquinnnelson'

import logging

import numpy as np


class NumericalDatasetHandler:
    def __init__(self, data_dir, train_data, val_data, train_class, val_class):
        """
        Initialize NumericalDatasetHandler.

        :param data_dir (str): fully qualified path to root directory inside which data subdirectories are placed
        :param train_data (str): fully qualified path to training data
        :param val_data (str): fully qualified path to validation data
        :param train_class (Dataset): torch Dataset class to use for training data
        :param val_class (Dataset): torch Dataset class to use for validation data
        """

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
        data = np.load(self.train_data, allow_pickle=True)
        logging.info(f'Loaded {len(data)} training records.')

        # initialize dataset
        dataset = self.train_class(data)

        return dataset

    def get_val_dataset(self):
        """
        Load validation data into memory and initialize the Dataset object.
        :return: Dataset
        """

        # load data
        data = np.load(self.val_data, allow_pickle=True)
        logging.info(f'Loaded {len(data)} validation records.')

        # initialize dataset
        dataset = self.val_class(data)

        return dataset
