import argparse

import yaml


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


def load_config(path_to_config):
    with open(path_to_config, "r") as config_file:
        config = yaml.safe_load(config_file)
        # config = yaml.load(config_file)
    return config


config = load_config(get_comandline_arguments()["config"])
