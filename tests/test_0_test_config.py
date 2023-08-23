import unittest
import logging
import pandas as pd
from pathlib import Path

from timewise.general import main_logger, cache_dir
from timewise.utils import get_mirong_sample, local_copy
from timewise.config_loader import TimewiseConfigLoader


main_logger.setLevel("DEBUG")
logger = logging.getLogger("timewise.test")


def get_test_yaml_filename():
    dir_path = Path(cache_dir)
    return dir_path / "test" / "test_config" / "test.yml"


def make_test_yaml():
    txt = (
        f"base_name: mirong_test \n"
        f"filename: {local_copy} \n"
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
        fn = Path(cache_dir) / "mirong_test" / "sample.csv"
        df = pd.read_csv(fn)
        self.assertTrue("w1mpro" in df.columns)
        self.assertTrue("w2mpro" in df.columns)
