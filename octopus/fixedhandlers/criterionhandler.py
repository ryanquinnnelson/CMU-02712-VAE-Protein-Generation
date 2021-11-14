"""
All things related to criterion.
"""
__author__ = 'ryanquinnnelson'

import logging

import torch.nn as nn

import customized.criterion as crit


class CriterionHandler:
    """
    Defines an object to handle criterion initialization.
    """

    def __init__(self, criterion_type):
        """
        Initialize CriterionHandler.

        Args:
            criterion_type (str): represents loss function to use
        """
        logging.info('Initializing criterion handler...')
        self.criterion_type = criterion_type

    def get_loss_function(self):
        """
        Obtain the desired loss function.

        Args:
            **kwargs: Any keyword arguments required by loss function

        Returns: class representing the loss function

        """
        criterion = None
        if self.criterion_type == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss()

        elif self.criterion_type == 'CustomLoss1':
            criterion = crit.CustomCriterion1()

        logging.info(f'Criterion is set:\n{criterion}')
        return criterion
