# This script has to be executed with timewise v0.5.3.
# It will produce test data downloads for the test sample.
from timewise import __version__

assert __version__ == "0.5.3"
from timewise.wise_data_by_visit import WiseDataByVisit

import logging
from pathlib import Path
import pandas as pd
from timewise.parent_sample_base import ParentSampleBase

logger = logging.getLogger("timewise.tests.data.create_test_data_from_v0")

DATA_DIR = Path(__file__).parent


class TestParentSample(ParentSampleBase):
    default_keymap = {"ra": "ra", "dec": "dec", "id": "name"}
    base_name = "test_sample"

    def __init__(self):
        logger.info("initialising test ParentSample")
        super(TestParentSample, self).__init__(base_name=TestParentSample.base_name)
        self.df = pd.read_csv(DATA_DIR / "test_sample.csv")


if __name__ == "__main__":
    logging.getLogger("timewise").setLevel("DEBUG")
    wise_data = WiseDataByVisit(
        base_name=TestParentSample.base_name,
        parent_sample_class=TestParentSample,
        min_sep_arcsec=6,
        n_chunks=2,
    )
    wise_data.get_photometric_data(service="tap", nthreads=2, chunks=[0, 1])
