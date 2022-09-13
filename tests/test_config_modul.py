import os

from nba_game_prediction.config_modul import get_comandline_arguments, load_config


class TestConfig:
    test_config_path = "data/test_config.yaml"

    def test_load_config(self):
        config_dict = load_config(self.test_config_path)
        assert len(config_dict) > 0
        assert os.path.isdir(config_dict["data_dir"])
        assert os.path.isdir(config_dict["output_dir"])

    def test_get_commandline_arguments(self):
        cl_args = get_comandline_arguments(["--config", self.test_config_path])
        assert cl_args["config"] == self.test_config_path
