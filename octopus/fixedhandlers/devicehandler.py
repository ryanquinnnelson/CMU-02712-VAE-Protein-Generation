"""
All things related to device.
"""
__author__ = 'ryanquinnnelson'

import logging

import torch


class DeviceHandler:
    """
    Defines an object that manages interactions with torch.device.
    """

    def __init__(self):
        """
        Initialize DeviceHandler.
        """
        logging.info('Initializing device handler...')
        self.device = None

    def setup(self):
        """
        Set up device handler. Set the device according to torch.device.
        Returns: None

        """
        logging.info('Setting up device...')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == 'cuda':
            logging.info(f'Gpu detected. device is set to {self.device}.')
        else:
            logging.info(f'No gpu detected. device is set to {self.device}.')

    def get_device(self):
        """
        Obtain the torch.device.
        Returns: torch.device

        """
        return self.device

    def move_model_to_device(self, model):
        """
        Move model to GPU if GPU is available.

        Avoids duplication issue with moving to device.

        "Note that calling my_tensor.to(device) returns a new copy of my_tensor on GPU. It does NOT overwrite my_tensor.
        Therefore, remember to manually overwrite tensors: my_tensor = my_tensor.to(torch.device('cuda'))."
        https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict

        Args:
            model (nn.Module): model to move

        Returns: nn.Module representing model after it has been moved

        """

        if self.device.type == 'cuda':
            model = model.to(device=torch.device('cuda'))

        return model

    def move_data_to_device(self, model, inputs, targets=None):
        """
        Move input and target to device if GPU is available and target is available, otherwise move input to device if
        GPU is available.

        Avoids duplication issue with moving to device.

        "Note that calling my_tensor.to(device) returns a new copy of my_tensor on GPU. It does NOT overwrite my_tensor.
        Therefore, remember to manually overwrite tensors: my_tensor = my_tensor.to(torch.device('cuda'))."
        https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict

        Args:
            model (nn.Module): model already moved to device
            inputs (Tensor): inputs data to move to device
            targets (Tensor): targets data to move to device, if any. Set as None if no targets data is available.

        Returns: Tuple (Tensor,Tensor) representing (inputs, targets).

        """

        # send input and targets to device
        if self.device.type == 'cuda':
            inputs = inputs.to(device=torch.device('cuda'))

            if targets is not None:
                targets = targets.to(device=torch.device('cuda'))

        # validate that model and input/targets are on the same device
        assert next(model.parameters()).device == inputs.device

        if targets is not None:
            assert inputs.device == targets.device

        return inputs, targets


