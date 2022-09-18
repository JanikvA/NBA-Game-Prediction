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


def setup_config(config: Dict[str, Any]) -> None:
    """Create directories defined in the yaml

    Args:
        config (Dict[str, Any]): config
    """
    # for dir_path in [config["train_model"]["dir_name"], config["plot_train_data"]["dir_name"]]:
    for dir_path_key in ["plot_train_data", "train_model"]:
        full_dir_path = os.path.join(
            config["output_dir"], config[dir_path_key]["dir_name"]
        )
        if not os.path.isdir(full_dir_path):
            logger.info(f"Creating {full_dir_path}")
            os.makedirs(full_dir_path)
        config[f"{dir_path_key}_outdir"] = full_dir_path


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
        config_dic = yaml.safe_load(config_file)
    config_dic["config_path"] = path_to_config
    setup_config(config_dic)
    logger.remove()
    logger.add(sys.stderr, level=config_dic["logging_level"])
    config_dic["logging_file"] = os.path.join(
        config_dic["output_dir"], config_dic["logging_file_name"]
    )
    logger.add(
        config_dic["logging_file"], level=config_dic["logging_level"], rotation="2 MB"
    )
    from rich.traceback import install

    install(show_locals=False, suppress=[pd, sns])
    return config_dic
