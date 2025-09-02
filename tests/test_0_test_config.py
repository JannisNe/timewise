import unittest
import logging
import pandas as pd
from pathlib import Path

from timewise.general import get_directories, main_logger
from timewise.utils import get_mirong_sample, get_mirong_path
from timewise.config_loader import TimewiseConfigLoader


main_logger.setLevel("DEBUG")
logger = logging.getLogger("timewise.test")


def get_test_yaml_filename():
    return get_directories()["cache_dir"] / "test" / "test_config" / "test.yml"


def make_test_yaml():
    mirong_path = get_mirong_path()
    txt = (
        f"base_name: mirong_test \n"
        f"filename: {mirong_path} \n"
        f"default_keymap: \n"
        f" ra: RA \n"
        f" dec: DEC \n"
        f" id: Name \n"
        f"timewise_instructions: \n"
        f" - match_all_chunks: \n"
        f"    additional_columns: \n"
        f"     - w1mpro \n"
        f"     - w2mpro \n"
    )

    filename = get_test_yaml_filename()
    filename.parents[0].mkdir(parents=True, exist_ok=True)
    logger.debug(f"writing test yaml file to {filename}")
    with open(filename, "w") as f:
        f.write(txt)


class TestConfig(unittest.TestCase):

    def setUp(self) -> None:
        get_mirong_sample()
        make_test_yaml()

    def test_a_config(self):
        logger.info("\n\n Testing TimewiseConfig \n")
        TimewiseConfigLoader.run_yaml(get_test_yaml_filename())
        logger.info("checking result")
        fn = get_directories()["cache_dir"] / "mirong_test" / "sample.csv"
        df = pd.read_csv(fn)
        self.assertTrue("w1mpro" in df.columns)
        self.assertTrue("w2mpro" in df.columns)
