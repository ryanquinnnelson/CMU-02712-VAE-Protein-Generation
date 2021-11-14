"""
All things related to data output.
"""
__author__ = 'ryanquinnnelson'

import logging
import os
from datetime import datetime

import octopus.helper as utilities


class OutputHandler:
    """
    Defines object to handle saving model output.
    """
    def __init__(self, run_name, output_dir):
        """
        Initialize OutputHandler.
        Args:
            run_name (str): Name of the run
            output_dir (str): fully qualified path to the directory where output should be written
        """
        logging.info('Initializing output handler...')
        self.run_name = run_name
        self.output_dir = output_dir

    def setup(self):
        """
        Perform all setup for output handler. Create output directory.

        Returns: None

        """
        logging.info('Preparing output directory...')
        utilities.create_directory(self.output_dir)

    def save(self, df, epoch):
        """
        Save given DataFrame to the output directory.

        Args:
            df (DataFrame): represents formatted model output
            epoch (int): epoch in which the data was obtained

        Returns: None

        """
        # generate filename
        filename = f'{self.run_name}.epoch{epoch}.{datetime.now().strftime("%Y%m%d.%H.%M.%S")}.output.csv'
        path = os.path.join(self.output_dir, filename)

        logging.info(f'Saving test output to {path}...')

        # save output
        df.to_csv(path, header=True, index=False)
