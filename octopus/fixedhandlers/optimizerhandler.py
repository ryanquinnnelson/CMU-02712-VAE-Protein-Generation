"""
All things related to optimizers.
"""
__author__ = 'ryanquinnnelson'

import logging

import torch.optim as optim


class OptimizerHandler:
    """
    Defines object to handle initializing optimizers.
    """

    def __init__(self, optimizer_type, optimizer_kwargs):
        """
        Initialize OptimizerHandler.
        Args:
            optimizer_type (str): represents the optimizer to construct
            optimizer_kwargs (Dict): dictionary of arguments for use in optimizer initialization
        """
        logging.info('Initializing optimizer handler...')
        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = optimizer_kwargs

    def get_optimizer(self, model):
        """
        Obtain the optimizer based on parameters.

        Args:
            model (nn.Module): model optimizer will manage

        Returns: nn.optim optimizer

        """
        opt = None
        if self.optimizer_type == 'Adam':
            opt = optim.Adam(model.parameters(), **self.optimizer_kwargs)
        elif self.optimizer_type == 'SGD':
            opt = optim.SGD(model.parameters(), **self.optimizer_kwargs)
        logging.info(f'Optimizer initialized:\n{opt}')
        logging.info(f'LR={opt.state_dict()["param_groups"][0]["lr"]}') # to ensure function works during training
        return opt
