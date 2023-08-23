import logging
from argparse import ArgumentParser

from timewise.general import main_logger
from timewise.config_loader import TimewiseConfigLoader


logger = logging.getLogger(__name__)


def timewise_cli():
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Path to timewise config file")
    parser.add_argument("-l", "--logging-level", default="INFO", type=str)
    cfg = vars(parser.parse_args())

    main_logger.setLevel(cfg.pop("logging_level"))
    TimewiseConfigLoader.run_yaml(cfg["config"])
