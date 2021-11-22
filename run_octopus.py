"""
Wrapper script to run octopus.
"""
__author__ = 'ryanquinnnelson'

import configparser
import sys
import glob
import os

from octopus.octopus import Octopus


def main():
    # run octopus for each config file found in the path
    config_path = sys.argv[1]
    filenames = glob.glob(os.path.join(config_path, '*'))
    filenames.sort()
    for f in filenames:
        # parse configs
        config = configparser.ConfigParser()
        config.read(f)

        # run octopus
        octopus = Octopus(config, f)
        octopus.install_packages()
        octopus.setup_environment()
        octopus.initialize_pipeline_components()
        octopus.run_pipeline()
        octopus.cleanup()


if __name__ == "__main__":
    # execute only if run as a script
    main()
