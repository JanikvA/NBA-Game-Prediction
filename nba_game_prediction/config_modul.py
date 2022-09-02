import argparse
import os
import sys

import pandas as pd
import seaborn as sns
import yaml
from loguru import logger


def get_comandline_arguments(custom_args=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", default="data/config.yaml", help="Path to yaml config file."
    )
    if custom_args:
        cl_args = parser.parse_args(custom_args)
    else:
        cl_args = parser.parse_args()
        cl_args = vars(cl_args)
    return cl_args


def setup(config):
    for dir_path_keys in ["data_dir", "output_dir"]:
        if not os.path.isdir(config[dir_path_keys]):
            logger.info(f"Creating {config[dir_path_keys]}")
            os.makedirs(config[dir_path_keys])


def load_config(path_to_config):
    with open(path_to_config, "r") as config_file:
        config = yaml.safe_load(config_file)
    setup(config)
    logger.remove()
    logger.add(sys.stderr, level=config["logging_level"])
    from rich.traceback import install

    install(show_locals=True, suppress=[pd, sns])
    return config


config = load_config(get_comandline_arguments()["config"])
