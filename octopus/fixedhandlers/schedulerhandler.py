"""
All things related to schedulers.
"""
__author__ = 'ryanquinnnelson'

import logging

import torch.optim as optim


class SchedulerHandler:
    """
    Defines object to initialize schedulers.
    """

    def __init__(self, scheduler_type, scheduler_kwargs, scheduler_plateau_metric):
        """
        Initialize SchedulerHandler.

        Args:
            scheduler_type (str): represents the scheduler to be used
            scheduler_kwargs (Dict): Dictionary of arguments needed to initialize the scheduler.
            scheduler_plateau_metric (str): Name of the metric used when updating scheduler, if any.
        """
        logging.info('Initializing scheduler handler...')
        self.scheduler_type = scheduler_type
        self.scheduler_kwargs = scheduler_kwargs
        self.scheduler_plateau_metric = scheduler_plateau_metric

    def get_scheduler(self, optimizer):
        """
        Obtain the scheduler based on parameters.

        Args:
            optimizer (nn.optim): Optimizer associated with the scheduler

        Returns: nn.optim Scheduler

        """
        scheduler = None

        if self.scheduler_type == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer, **self.scheduler_kwargs)

        elif self.scheduler_type == 'MultiStepLR':

            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **self.scheduler_kwargs)

        elif self.scheduler_type == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.scheduler_kwargs)

        logging.info(f'Scheduler initialized:\n{scheduler}\n{scheduler.state_dict()}')
        return scheduler

    def update_scheduler(self, scheduler, stats):
        """
        Perform a single scheduler step.

        Args:
            scheduler (nn.optim): scheduler to step
            stats (Dictionary): dictionary of run stats from which the latest scheduler_plateau_metric value should be
            extracted

        Returns: None

        """
        if self.scheduler_type == 'ReduceLROnPlateau':
            metric_val = stats[self.scheduler_plateau_metric][-1]
            scheduler.step(metric_val)
        else:
            scheduler.step()
