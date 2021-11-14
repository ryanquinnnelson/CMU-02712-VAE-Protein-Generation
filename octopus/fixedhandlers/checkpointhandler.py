"""
All things related to model checkpoints.
"""
__author__ = 'ryanquinnnelson'

import os
import logging

import torch

from octopus import helper


class CheckpointHandler:
    """
    Defines an object to handle checkpoints.
    """

    def __init__(self, checkpoint_dir, delete_existing_checkpoints, run_name, load_from_checkpoint):
        """
        Initialize CheckpointHandler. Override given value of delete_existing_checkpoints if loading from a previous
        checkpoint is not None.

        Args:
            checkpoint_dir (str): fully-qualified path where checkpoints should be written
            delete_existing_checkpoints (Boolean): True if checkpoint directory should be deleted and recreated
            run_name (str): Name of the current run
            load_from_checkpoint (str): fully-qualified filename of checkpoint file to load
        """
        logging.info('Initializing checkpoint handler...')
        self.checkpoint_dir = checkpoint_dir
        self.delete_existing_checkpoints = delete_existing_checkpoints
        self.run_name = run_name
        self.load_from_checkpoint = load_from_checkpoint

        # override
        if self.load_from_checkpoint:
            logging.info('Overriding delete_existing_checkpoints value. ' +
                         'Existing checkpoints will not be deleted because checkpoint is being loaded for this run.')
            self.delete_existing_checkpoints = False

    def setup(self):
        """
        Set up handler. Delete and recreate checkpoint directory if delete_existing_checkpoints=True, otherwise
        create checkpoint directory.

        Returns: None

        """
        logging.info('Preparing checkpoint directory...')
        if self.delete_existing_checkpoints:
            helper.delete_directory(self.checkpoint_dir)

        helper.create_directory(self.checkpoint_dir)

    def save(self, model, optimizer, scheduler, next_epoch, stats):
        """
        Save current model environment to a checkpoint.

        Args:
            model (nn.Module): model to save
            optimizer (nn.optim): optimizer to save
            scheduler (nn.optim): scheduler to save
            next_epoch (int): next epoch to execute if this model is restored
            stats (Dict): dictionary of statistics for all epochs collected during model training to this point

        Returns: None

        """
        # build filename
        filename = os.path.join(self.checkpoint_dir, f'{self.run_name}.checkpoint.{next_epoch - 1}.pt')
        logging.info(f'Saving checkpoint to {filename}...')

        # build state dictionary
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'next_epoch': next_epoch,
            'stats': stats
        }

        torch.save(checkpoint, filename)

    def load(self, filename, device, model, optimizer, scheduler):
        """
        Load a previously saved model environment from a checkpoint file, mapping the load based on the device.

        Args:
            filename (str): fully-qualified filename of checkpoint file
            device (torch.device): device on which model was previously running
            model (nn.Module): model to update based on checkpoint
            optimizer (nn.optim): optimizer to update based on checkpoint
            scheduler (nn.optim): scheduler to update based on checkpoint

        Returns: checkpoint object

        """
        logging.info(f'Loading checkpoint from {filename}...')
        checkpoint = torch.load(filename, map_location=device)

        # reload saved states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint
