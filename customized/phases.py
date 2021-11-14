"""
All things related to model training phases.
"""
__author__ = 'ryanquinnnelson'

import logging
import warnings

import torch
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')


class Training:
    """
    Defines object to manage Training phase of training.
    """

    def __init__(self, train_loader, criterion, devicehandler):
        """
        Initialize Training object.

        Args:
            train_loader (DataLoader): DataLoader for training data
            criterion (class): loss function
            devicehandler (DeviceHandler):manages device on which training is being run
        """
        logging.info('Loading training phase...')
        self.train_loader = train_loader
        self.criterion = criterion
        self.devicehandler = devicehandler

    def train_model(self, epoch, num_epochs, model, optimizer):
        """
        Executes one epoch of training.

        Args:
            epoch (int): Epoch being trained
            num_epochs (int): Total number of epochs to be trained
            model (nn.Module): model being trained
            optimizer (nn.optim): optimizer for this model

        Returns: float representing average training loss

        """
        logging.info(f'Running epoch {epoch}/{num_epochs} of training...')

        train_loss = 0

        # Set model in 'Training mode'
        model.train()

        # process mini-batches
        # TODO: Remove targets when change to protein dataset
        for i, (inputs, targets) in enumerate(self.train_loader):
            # prep
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            inputs, targets = self.devicehandler.move_data_to_device(model, inputs, None)

            # compute forward pass
            out, mu, sigma = model.forward(inputs)

            # calculate loss
            loss = self.criterion.calculate_loss(inputs, out, mu, sigma)
            train_loss += loss.item()

            # delete mini-batch data from device
            del inputs

            # compute backward pass
            loss.backward()

            # update model weights
            optimizer.step()

        # calculate average loss across all mini-batches
        train_loss /= len(self.train_loader)

        return train_loss


class Evaluation:
    """
    Defines an object to manage the evaluation phase of training.
    """

    def __init__(self, val_loader, criterion, devicehandler):
        """
        Initialize Evaluation object.

        Args:
            val_loader (DataLoader): DataLoader for validation dataset
            criterion (class): loss function
            devicehandler (DeviceHandler): object to manage interaction of model/data and device
        """
        logging.info('Loading evaluation phase...')
        self.val_loader = val_loader
        self.criterion = criterion
        self.devicehandler = devicehandler

    def evaluate_model(self, epoch, num_epochs, model):
        """
        Perform evaluation phase of training.

        Args:
            epoch (int): Epoch being trained
            num_epochs (int): Total number of epochs to be trained
            model (nn.Module): model being trained

        Returns: Tuple (float,float) representing (val_loss, val_metric)

        """
        logging.info(f'Running epoch {epoch}/{num_epochs} of evaluation...')

        val_loss = 0
        num_hits = 0

        with torch.no_grad():  # deactivate autograd engine to improve efficiency

            # Set model in validation mode
            model.eval()

            # process mini-batches
            for i, (inputs, targets) in enumerate(self.val_loader):
                # prep
                inputs, targets = self.devicehandler.move_data_to_device(model, inputs, None)

                # compute forward pass
                out, mu, sigma = model.forward(inputs)

                # calculate loss
                loss = self.criterion.calculate_loss(inputs, out, mu, sigma)
                val_loss += loss.item()

                # calculate accuracy
                # TODO: fix once stop using MNIST
                num_hits += 1 #accuracy_score(inputs, out, normalize=False)

                # delete mini-batch from device
                del inputs

            # calculate evaluation metrics
            val_loss /= len(self.val_loader)  # average per mini-batch
            val_acc = num_hits / len(self.val_loader.dataset)

            return val_loss, val_acc
