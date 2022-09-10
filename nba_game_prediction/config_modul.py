import argparse
import os
import sys
from typing import Any, Dict

import pandas as pd
import seaborn as sns
import yaml
from loguru import logger


def get_comandline_arguments(custom_args: str = None) -> Dict[str, Any]:
    """Parse commandline arguments

    Args:
        custom_args (str, optional): Ignores the commandline arguments and
        uses these arguments if they are not None. Defaults to None.

    Returns:
        Dict[str, Any]: command line arguments
    """
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
    return vars(cl_args)


def setup(config: Dict[str, Any]) -> None:
    """Create directories defined in the yaml

    Args:
        config (Dict[str, Any]): config
    """
    for dir_path_keys in ["data_dir", "output_dir"]:
        if not os.path.isdir(config[dir_path_keys]):
            logger.info(f"Creating {config[dir_path_keys]}")
            os.makedirs(config[dir_path_keys])


def load_config(path_to_config: str) -> Dict[str, Any]:
    """Load config and do initial setup of logger
    and tracebacks from rich

    Args:
        path_to_config (str): path to the yaml config file

    Returns:
        Dict[str, Any]: config
    """
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Loading config from {path_to_config}")
    with open(path_to_config, "r") as config_file:
        config = yaml.safe_load(config_file)
    config["config_path"] = path_to_config
    setup(config)
    logger.remove()
    logger.add(sys.stderr, level=config["logging_level"])
    logger.add(config["logging_file"], level=config["logging_level"], rotation="2 MB")
    from rich.traceback import install

    install(show_locals=False, suppress=[pd, sns])
    return config
