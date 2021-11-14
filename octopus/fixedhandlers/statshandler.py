"""
All things related to stats collection.
"""
__author__ = 'ryanquinnnelson'

import logging


class StatsHandler:
    """
    Define object to manage stats collection and use.
    """

    def __init__(self, val_metric_name, comparison_metric, comparison_best_is_max, comparison_patience):
        """
        Initialize StatsHandler. Set the num_epochs_worse_than_best_model equal to 0.

        Args:
            val_metric_name (str): Name of the second metric returned from Evaluation.evaluate_model() for clarity in
            stats collection.
            comparison_metric (str): Name of the metric to be used for comparison between models for the purposes of
            early stopping.
            comparison_best_is_max (Boolean): True if higher values of the comparison metric mean better performance.
            comparison_patience (int): Number of epochs current model can perform worse than a previous run before early stopping.
        """
        logging.info('Initializing stats handler...')
        self.stats = {'epoch': [],
                      'lr': [],
                      'runtime': [],
                      'train_loss': [],
                      'val_loss': [],
                      val_metric_name: []}
        self.val_metric_name = val_metric_name
        self.comparison_metric = comparison_metric
        self.best_is_max = comparison_best_is_max
        self.comparison_patience = comparison_patience
        self.num_epochs_worse_than_best_model = 0

    def _model_is_worse_by_comparison_metric(self, epoch, wandbconnector):
        """
        Determine whether this model is performing better or worse than any previous run for the given epoch and
        predefined comparison metric.

        Args:
            epoch (int): epoch to compare
            wandbconnector (WandbConnector): connection to wandb

        Returns: True if model is worse, False otherwise.

        """

        # check whether the metric we are using for comparison against other runs
        # is better than other runs for this epoch
        logging.info('Checking whether comparison metric for current model is worse than the best model so far...')
        best_name, best_val = wandbconnector.get_best_value(self.comparison_metric, epoch, self.best_is_max)
        model_name = wandbconnector.run_name
        model_val = self.stats[self.comparison_metric][-1]
        logging.info(f'best:\t{best_name}\t{best_val}\nmodel:\t{model_name}\t{model_val}')

        if best_val is not None and self.best_is_max:
            # compare values for this epoch
            if model_val < best_val:
                model_is_worse = True
                logging.info('A previous model has the best value for the comparison metric for this epoch.')
            else:
                logging.info('Current model has the best value for the comparison metric for this epoch.')
                model_is_worse = False
        elif best_val is not None and not self.best_is_max:  # best is min
            # compare values for this epoch
            if model_val > best_val:
                model_is_worse = True
                logging.info('A previous model has the best value for the comparison metric for this epoch.')
            else:
                logging.info('Current model has the best value for the comparison metric for this epoch.')
                model_is_worse = False
        else:
            # no model to compare against for this epoch
            model_is_worse = False

        return model_is_worse

    def stopping_criteria_is_met(self, epoch, wandbconnector):
        """
        Determine whether stopping criteria is met. Criteria is currently:
        1. Does the current model perform worse than any previous run for longer than is allowed according to
        comparison_patience?

        Args:
            epoch (int): epoch to compare
            wandbconnector (WandbConnector): connection to wandb

        Returns: True if stopping criteria is met, False otherwise.

        """
        logging.info('Checking early stopping criteria...')

        # criteria 1 - metric comparison
        model_is_worse = self._model_is_worse_by_comparison_metric(epoch, wandbconnector)
        if model_is_worse:
            self.num_epochs_worse_than_best_model += 1
            logging.info('Number of epochs in a row this model is worse than best model' +
                         f':{self.num_epochs_worse_than_best_model}')
            logging.info('Number of epochs in a row this model can be worse than best model ' +
                         f'before stopping:{self.comparison_patience}')
        else:
            self.num_epochs_worse_than_best_model = 0  # reset value

        # check all criteria to determine if we need to stop learning
        if self.num_epochs_worse_than_best_model > self.comparison_patience:
            stopping_criteria_met = True
            logging.info('Early stopping criteria is met.')
        else:
            stopping_criteria_met = False
            logging.info('Early stopping criteria is not met.')

        return stopping_criteria_met

    def report_stats(self, wandbconnector):
        """
        Report latest stats for all captured metrics to wandb.

        Args:
            wandbconnector (WandbConnector): connection to wandb

        Returns: None

        """

        # save epoch stats to wandb
        epoch_stats_dict = dict()
        for key in self.stats.keys():
            epoch_stats_dict[key] = self.stats[key][-1]  # latest value
        wandbconnector.log_stats(epoch_stats_dict)
        logging.info(f'stats:{epoch_stats_dict}')

    def collect_stats(self, epoch, lr, train_loss, val_loss, val_metric, start, end):
        """
        Collect and store stats for a given epoch of model training.
        Args:
            lr (float): learning rate
            epoch (int): number of the epoch
            train_loss (float): average training loss
            val_loss (float): average validation loss
            val_metric (float): second validation metric calculated during validation
            start (float): starting time of the epoch
            end (float): ending time of the epoch

        Returns: None

        """

        logging.info(f'Collecting stats for epoch {epoch}...')

        # calculate runtime
        runtime = end - start

        # update stats
        self.stats['epoch'].append(epoch)
        self.stats['lr'].append(lr)
        self.stats['runtime'].append(runtime)
        self.stats['train_loss'].append(train_loss)
        self.stats['val_loss'].append(val_loss)
        self.stats[self.val_metric_name].append(val_metric)

    def report_previous_stats(self, wandbconnector):
        """
        For each epoch stored in the stats dictionary, send all metrics for that epoch to wandb.

        Args:
            wandbconnector (WandbConnector): connection to wandb

        Returns: None

        """
        logging.info('Reporting previous stats...')
        n_stats = len(self.stats[list(self.stats.keys())[0]])  # calculate how many epochs of stats were collected
        for i in range(0, n_stats):
            epoch_stats_dict = dict()
            for key in self.stats.keys():
                epoch_stats_dict[key] = self.stats[key][i]
            wandbconnector.log_stats(epoch_stats_dict)
