"""
All things related to managing training, evaluation, and testing phases.
"""
__author__ = 'ryanquinnnelson'

import logging
import time


class PhaseHandler:
    """
    Defines an object to manage training phases.
    """

    def __init__(self, num_epochs, outputhandler, devicehandler, statshandler, checkpointhandler,
                 schedulerhandler, wandbconnector, formatter, load_from_checkpoint, checkpoint_file=None):
        """
        Initialize PhaseHandler. Set the first epoch to 1.

        Args:
            num_epochs (int): Number of epochs to train
            outputhandler (OutputHandler): handler for writing to files
            devicehandler (DeviceHandler): handler for torch.device
            statshandler (StatsHandler): handler for stats
            checkpointhandler (CheckpointHandler): handler for checkpoints
            schedulerhandler (SchedulerHandler): handler for schedulers
            wandbconnector (WandbConnector): connector to wandb
            formatter (OutputFormatter): class defining how to format test output
            load_from_checkpoint (Boolean): True if model environment should be loaded from a previously saved checkpoint
            checkpoint_file (str): Fully-qualified filename of checkpoint file to be loaded, if any
        """
        logging.info('Initializing phase handler...')
        self.load_from_checkpoint = load_from_checkpoint
        self.checkpoint_file = checkpoint_file
        self.first_epoch = 1
        self.num_epochs = num_epochs

        # handlers
        self.outputhandler = outputhandler
        self.devicehandler = devicehandler
        self.statshandler = statshandler
        self.checkpointhandler = checkpointhandler
        self.schedulerhandler = schedulerhandler
        self.wandbconnector = wandbconnector

        # formatter for test output
        self.formatter = formatter

    def _load_checkpoint(self, model, optimizer, scheduler):
        """
        Load model environment from previous checkpoint. Replace stats dictionary with stats dictionary recovered
        from checkpoint and update first epoch to next epoch value recovered from checkpoint.

        Args:
            model (nn.Module): model to update based on checkpoint
            optimizer (nn.optim): optimizer to update based on checkpoint
            scheduler (nn.optim): scheduler to update based on checkpoint

        Returns: None

        """
        device = self.devicehandler.get_device()
        checkpoint = self.checkpointhandler.load(self.checkpoint_file, device, model, optimizer, scheduler)

        # restore stats
        self.statshandler.stats = checkpoint['stats']

        # set which epoch to start from
        self.first_epoch = checkpoint['next_epoch']

    def process_epochs(self, model, optimizer, scheduler, training, evaluation, generation):
        """
        Run training phases for all epochs. Load model from checkpoint first if necessary and submit all previous
        stats to wandb.

        """

        # load checkpoint if necessary
        if self.load_from_checkpoint:
            self._load_checkpoint(model, optimizer, scheduler)

            # submit old stats to wandb to align with other runs
            self.statshandler.report_previous_stats(self.wandbconnector)

        # run epochs
        for epoch in range(self.first_epoch, self.num_epochs + 1):
            # record start time
            start = time.time()

            # train
            train_loss = training.train_model(epoch, self.num_epochs, model, optimizer)

            # validate
            val_loss, val_metric = evaluation.evaluate_model(epoch, self.num_epochs, model)

            # generate
            out = generation.generate_proteins(epoch, self.num_epochs, model)
            out = self.formatter.format_output(out)
            self.outputhandler.save(out, epoch)

            # stats
            end = time.time()
            lr = optimizer.state_dict()["param_groups"][0]["lr"]
            self.statshandler.collect_stats(epoch, lr, train_loss, val_loss, val_metric, start, end)
            self.statshandler.report_stats(self.wandbconnector)

            # scheduler
            self.schedulerhandler.update_scheduler(scheduler, self.statshandler.stats)

            # save model checkpoint
            if epoch % 5 == 0:
                self.checkpointhandler.save(model, optimizer, scheduler, epoch + 1, self.statshandler.stats)

            # check if early stopping criteria is met
            if self.statshandler.stopping_criteria_is_met(epoch, self.wandbconnector):
                logging.info('Early stopping criteria is met. Stopping the training process...')
                break  # stop running epochs
