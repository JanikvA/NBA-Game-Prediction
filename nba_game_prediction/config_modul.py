import argparse
import os

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

    # For convenience adding keys that are used multiple times
    config["raw_output_path"] = os.path.join(
        config["data_dir"], config["collect_game_data"]["raw_output_name"]
    )
    config["combined_output_path"] = os.path.join(
        config["data_dir"], config["collect_game_data"]["combined_output_name"]
    )
    config["train_data_output_path"] = os.path.join(
        config["data_dir"], config["create_train_data"]["train_data_output_name"]
    )


def load_config(path_to_config):
    with open(path_to_config, "r") as config_file:
        config = yaml.safe_load(config_file)
    setup(config)
    return config


config = load_config(get_comandline_arguments()["config"])
