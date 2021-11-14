"""
Common utilities.
"""
__author__ = 'ryanquinnnelson'

import logging
import os
import shutil

import torch
from pynvml import *


def create_directory(path):
    """
    Creates directory if it does not exist.
    Args:
        path (str): fully qualified path to the directory

    Returns: None

    """

    if os.path.isdir(path):
        logging.info(f'Directory already exists:{path}.')
    else:
        os.mkdir(path)
        logging.info(f'Created directory:{path}.')


def delete_directory(path):
    """
    Deletes directory if it exists.
    Args:
        path (str): fully qualified path to the directory

    Returns:None

    """

    if os.path.isdir(path):
        shutil.rmtree(path)
        logging.info(f'Deleted directory:{path}.')
    else:
        logging.info(f'Directory does not exist:{path}.')


def delete_file(path):
    """
    Deletes file if it exists.
    Args:
        path (str): fully qualified path to the file

    Returns:None

    """

    if os.path.isfile(path):
        os.remove(path)
        logging.info(f'Deleted file:{path}')
    else:
        logging.info(f'File does not exist:{path}')


def _to_int_list(s):
    """
    Builds an integer list from a comma-separated string of integers.
    Args:
        s (str): comma-separated string of integers

    Returns:List

    """

    return [int(a) for a in s.strip().split(',')]


def _to_string_list(s):
    """
    Builds a list of strings from a comma-separated string.
    Args:
        s (str): comma-separated string of strings

    Returns:List

    """

    return s.strip().split(',')


def _to_int_dict(s):
    """
    Builds a dictionary where each value is an integer, given a comma-separated string of key=value pairs. If a value
    cannot be converted to an integer, leaves the value as a string.
    Args:
        s (str): comma-separated string of key=value pairs (i.e. key1=1,key2=2)

    Returns:Dict

    """

    d = dict()

    pairs = s.split(',')
    for p in pairs:
        key, val = p.strip().split('=')

        # try converting the value to an int
        try:
            val = int(val)
        except ValueError:
            pass  # leave as string

        d[key] = val

    return d


def _to_float_dict(s):
    """
    Builds a dictionary where each value is a float, given a comma-separated string of key=value pairs. If a value
    cannot be converted to a float, leaves the value as a string.
    Args:
        s (str): comma-separated string of key=value pairs (i.e. key1=1,key2=2)

    Returns:Dict

    """

    d = dict()

    pairs = s.split(',')
    for p in pairs:
        key, val = p.strip().split('=')

        # try converting the value to a float
        try:
            val = float(val)
        except ValueError:
            pass  # leave as string

        d[key] = val

    return d
