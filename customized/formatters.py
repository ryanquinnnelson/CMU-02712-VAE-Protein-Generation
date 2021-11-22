"""
Contains all Formatter objects customized to the data.
"""
__author__ = 'ryanquinnnelson'

import pandas as pd
import logging


class OutputFormatter:
    """
    Defines an object to manage formatting of test output.
    """

    def __init__(self):
        """
        Initialize OutputFormatter.
        Args:
            data_dir (str): fully-qualified path to data directory
        """
        logging.info('Initializing output formatter...')

    def format_output(self, out):
        """
        Format given model output as desired.

        Args:
            out (np.array): Model output

        Returns: DataFrame after formatting

        """

        # convert string array to dataframe
        df = pd.DataFrame(out)

        return df
