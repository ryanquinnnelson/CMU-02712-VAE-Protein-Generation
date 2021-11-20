"""
Performs environment setup for deep learning and runs a deep learning pipeline.
"""
__author__ = 'ryanquinnnelson'

import logging
import os
import sys

# execute before loading torch
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # better error tracking from gpu

# reusable local modules
from octopus.helper import _to_string_list, _to_float_dict, _to_int_list, _to_mixed_dict
from octopus.connectors.wandbconnector import WandbConnector
from octopus.fixedhandlers.checkpointhandler import CheckpointHandler
from octopus.fixedhandlers.devicehandler import DeviceHandler
from octopus.fixedhandlers.criterionhandler import CriterionHandler
from octopus.fixedhandlers.optimizerhandler import OptimizerHandler
from octopus.fixedhandlers.schedulerhandler import SchedulerHandler
from octopus.fixedhandlers.statshandler import StatsHandler
from octopus.fixedhandlers.phasehandler import PhaseHandler
from octopus.fixedhandlers.dataloaderhandler import DataLoaderHandler
from octopus.fixedhandlers.outputhandler import OutputHandler
from octopus.fixedhandlers.piphandler import PipHandler
from octopus.datasethandlers.numericaldatasethandler import NumericalDatasetHandler
from octopus.modelhandlers.vaehandler import VaeHandler

# customized to this data
from customized.phases import Training, Evaluation, Generation
from customized.formatters import OutputFormatter
from customized.datasets import TrainValDataset
import customized.datasets as customized_datasets


class Octopus:
    """
    Class that manages the building and training of deep learning models.
    """

    def __init__(self, config, config_file):
        """
        Initialize octopus object. Set up logging, initialize all connectors and handlers.
        Args:
            config (ConfigParser): contains the configuration for this run
            config_file (str): fully qualified path to the configuration file
        """

        # save configuration
        self.config = config

        # logging
        _setup_logging(config['debug']['debug_path'], config['DEFAULT']['run_name'])
        _draw_logo()
        logging.info('Initializing octopus...')
        logging.info(f'Parsing configuration from {config_file}...')

        # connectors
        self.wandbconnector = initialize_connectors(config)

        # fixed handlers
        fixedhandlers = initialize_fixed_handlers(config, self.wandbconnector)
        self.piphandler = fixedhandlers[0]
        self.checkpointhandler = fixedhandlers[1]
        self.criterionhandler = fixedhandlers[2]
        self.dataloaderhandler = fixedhandlers[3]
        self.devicehandler = fixedhandlers[4]
        self.optimizerhandler = fixedhandlers[5]
        self.outputhandler = fixedhandlers[6]
        self.schedulerhandler = fixedhandlers[7]
        self.statshandler = fixedhandlers[8]
        self.phasehandler = fixedhandlers[9]

        # variable handlers
        self.inputhandler, self.modelhandler = initialize_variable_handlers(config)
        logging.info('octopus initialization is complete.')

    def install_packages(self):
        """
        Perform installation of all packages via pip.
        :return:
        """
        logging.info('octopus is installing necessary packages...')
        self.piphandler.install_packages()
        logging.info('Package installation is complete.')

    def setup_environment(self):
        """
        Perform all tasks necessary to prepare the environment for training. Set up and log into wandb, set up
        kaggle (if necessary), delete previous version of the checkpoint directory if necessary and create checkpoint
        directory, create output directory, determine whether device is cpu or cuda (gpu), and set DataLoader
        arguments based on device.

        Returns:None
        """

        logging.info('octopus is setting up the environment...')
        # wandb
        self.wandbconnector.setup()

        # checkpoint directory
        self.checkpointhandler.setup()

        # output directory
        self.outputhandler.setup()

        # device
        self.devicehandler.setup()

        # dataloaders
        self.dataloaderhandler.setup(self.devicehandler.device, self.inputhandler)
        logging.info('octopus has finished setting up the environment.')

    def initialize_pipeline_components(self):
        """
        Initialize all components of the deep learning pipeline, including model, loss function, optimizer, scheduler,
        dataloaders, and training phases.

        :return: None
        """

        logging.info('octopus is initializing pipeline components...')
        # initialize model
        self.model = self.modelhandler.get_model()
        self.model = self.devicehandler.move_model_to_device(self.model)  # move before initializing optimizer - Note 1
        self.wandbconnector.watch(self.model)

        # initialize model components
        self.loss_func = self.criterionhandler.get_loss_function()
        self.optimizer = self.optimizerhandler.get_optimizer(self.model)
        self.scheduler = self.schedulerhandler.get_scheduler(self.optimizer)

        # load data
        self.train_loader, self.val_loader = self.dataloaderhandler.load(self.inputhandler)

        # load phases
        self.training = Training(self.train_loader, self.loss_func, self.devicehandler)
        self.evaluation = Evaluation(self.val_loader, self.loss_func, self.devicehandler)
        self.generation = Generation(self.config['hyperparameters'].getint('num_proteins_to_generate'),
                                     self.devicehandler)

        logging.info('Pipeline components are initialized.')

    def run_pipeline(self):
        """
        Run training, validation, and test phases of training for all epochs.

        Note 1:
        Reason behind moving model to device first:
        https://stackoverflow.com/questions/66091226/runtimeerror-expected-all-tensors-to-be-on-the-same-device-but-found-at-least

        Returns: None

        """
        logging.info('octopus is running the pipeline...')

        # run epochs
        self.phasehandler.process_epochs(self.model, self.optimizer, self.scheduler, self.training, self.evaluation,
                                         self.generation)

        logging.info('octopus has finished running the pipeline.')

    def cleanup(self):
        """
        Perform any cleanup steps. Stop wandb logging for the current run to enable multiple runs within a single
        execution of run_octopus.py.
        Returns:None

        """
        self.wandbconnector.run.finish()  # finish logging for this run
        logging.info('octopus shutdown complete.')


def _setup_logging(debug_path, run_name):
    """
    Perform all tasks necessary to set up logging to stdout and to a file.
    Args:
        debug_path (str): fully qualified path where logs should be written
        run_name(str): name of the current run to be appended to the log filename

    Returns: None

    """

    # create directory if it doesn't exist
    if not os.path.isdir(debug_path):
        os.mkdir(debug_path)

    # generate filename
    debug_file = os.path.join(debug_path, 'debug.' + run_name + '.log')

    # delete older debug file if it exists
    if os.path.isfile(debug_file):
        os.remove(debug_file)

    # define basic logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    logger.handlers = []  # clear out previous handlers

    # write to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # write to debug file
    handler = logging.FileHandler(debug_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def _draw_logo():
    """
    Write octopus logo to the log.
    Returns:None

    """

    logging.info('              _---_')
    logging.info('            /       \\')
    logging.info('           |         |')
    logging.info('   _--_    |         |    _--_')
    logging.info('  /__  \\   \\  0   0  /   /  __\\')
    logging.info('     \\  \\   \\       /   /  /')
    logging.info('      \\  -__-       -__-  /')
    logging.info('  |\\   \\    __     __    /   /|')
    logging.info('  | \\___----         ----___/ |')
    logging.info('  \\                           /')
    logging.info('   --___--/    / \\    \\--___--')
    logging.info('         /    /   \\    \\')
    logging.info('   --___-    /     \\    -___--')
    logging.info('   \\_    __-         -__    _/')
    logging.info('     ----               ----')
    logging.info('')
    logging.info('       O  C  T  O  P  U  S')
    logging.info('')


def initialize_connectors(config):
    """
    Initialize classes that manage connections to external tools like kaggle and wandb.
    Args:
        config (ConfigParser): contains the configuration for this run

    Returns:Tuple (kaggleconnector, wandbconnector)

    """
    # wandb
    # get all hyperparameters from different parts of config so wandb can track things that we might want to change
    hyper_dict = dict(config['hyperparameters'])
    hyper_dict.update(dict(config['model']))
    hyper_dict.update(dict(config['dataloader']))
    wandbconnector = WandbConnector(config['wandb']['wandb_dir'],
                                    config['wandb']['entity'],
                                    config['DEFAULT']['run_name'],
                                    config['wandb']['project'],
                                    config['wandb']['notes'],
                                    _to_string_list(config['wandb']['tags']),
                                    hyper_dict)

    return wandbconnector


def initialize_fixed_handlers(config, wandbconnector):
    """
    Initialize all object handlers that remain the same regardless of changes to data.
    Args:
        config (ConfigParser): contains the configuration for this run
        wandbconnector (WandbConnector): object that manages connection to wandb

    Returns: Tuple (checkpointhandler, criterionhandler, dataloaderhandler, devicehandler, \
           optimizerhandler, outputhandler, schedulerhandler, statshandler, phasehandler)

    """

    # pip
    if config.has_option('pip', 'packages'):
        packages_list = _to_string_list(config['pip']['packages'])
    else:
        packages_list = None
    piphandler = PipHandler(packages_list)

    # checkpoints
    checkpointhandler = CheckpointHandler(config['checkpoint']['checkpoint_dir'],
                                          config['checkpoint'].getboolean('delete_existing_checkpoints'),
                                          config['DEFAULT']['run_name'],
                                          config['checkpoint'].getboolean('load_from_checkpoint'))

    # criterion
    criterionhandler = CriterionHandler(config['hyperparameters']['criterion_type'],
                                        _to_mixed_dict(config['hyperparameters']['criterion_kwargs']))

    # device
    devicehandler = DeviceHandler()

    # dataloader
    dataloaderhandler = DataLoaderHandler(config['dataloader'].getint('batch_size'),
                                          config['dataloader'].getint('num_workers'),
                                          config['dataloader'].getboolean('pin_memory'))

    # optimizer
    optimizerhandler = OptimizerHandler(config['hyperparameters']['optimizer_type'],
                                        _to_float_dict(config['hyperparameters']['optimizer_kwargs']))

    # output
    outputhandler = OutputHandler(config['DEFAULT']['run_name'],
                                  config['output']['output_dir'])

    # scheduler
    schedulerhandler = SchedulerHandler(config['hyperparameters']['scheduler_type'],
                                        _to_float_dict(config['hyperparameters']['scheduler_kwargs']),
                                        config['hyperparameters']['scheduler_plateau_metric'])

    # statshandler
    statshandler = StatsHandler(config['stats']['val_metric_name'],
                                config['stats']['comparison_metric'],
                                config['stats'].getboolean('comparison_best_is_max'),
                                config['stats'].getint('comparison_patience'))

    # phasehandler
    if config.has_option('checkpoint', 'checkpoint_file'):
        checkpoint_file = config['checkpoint']['checkpoint_file']
    else:
        checkpoint_file = None

    phasehandler = PhaseHandler(config['hyperparameters'].getint('num_epochs'),
                                outputhandler,
                                devicehandler,
                                statshandler,
                                checkpointhandler,
                                schedulerhandler,
                                wandbconnector,
                                OutputFormatter(),
                                config['checkpoint'].getboolean('load_from_checkpoint'),
                                checkpoint_file)

    return piphandler, checkpointhandler, criterionhandler, dataloaderhandler, devicehandler, optimizerhandler, \
           outputhandler, schedulerhandler, statshandler, phasehandler


def initialize_variable_handlers(config):
    """
    Initialize all object handlers that may change depending on the data and model configurations.

    Args:
        config  (ConfigParser): contains the configuration for this run

    Returns: Tuple (inputhandler, modelhandler, verification)

    """
    # input
    if config['data']['data_type'] == 'numerical':
        inputhandler = NumericalDatasetHandler(config['data']['data_dir'],
                                               config['data']['train_data'],
                                               config['data']['val_data'],
                                               TrainValDataset,
                                               TrainValDataset)
    else:
        inputhandler = None

    # model
    if 'VAE' in config['model']['model_type']:
        modelhandler = VaeHandler(config['model']['model_type'],
                                  config['model'].getint('input_size'),
                                  _to_int_list(config['model']['hidden_sizes']),
                                  config['model'].getint('latent_dim'),
                                  config['model'].getboolean('batch_normalization'),
                                  config['model'].getfloat('dropout'))

    else:
        modelhandler = None

    return inputhandler, modelhandler
