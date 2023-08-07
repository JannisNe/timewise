import logging
from argparse import ArgumentParser
from timewise.config_loader import TimewiseConfig


logger = logging.getLogger(__name__)


def timewise_cli():
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Path to timewise config file")
    parser.add_argument("-l", "--logging-level", default="INFO", type=str)
    cfg = vars(parser.parse_args())

    logging.getLogger("timewise").setLevel(cfg.pop("logging_level"))
    TimewiseConfig.run_yaml(cfg["config"])
