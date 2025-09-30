# This script has to be executed with timewise v0.5.3.
# It will produce test data downloads for the test sample.
from timewise import __version__

assert __version__ == "0.5.3"
from timewise.wise_data_by_visit import WiseDataByVisit

import logging
from pathlib import Path
import pandas as pd
import shutil
import json
from timewise.parent_sample_base import ParentSampleBase

logger = logging.getLogger("timewise.tests.data.create_test_data_from_v0")

DATA_DIR = Path(__file__).parent
PHOT_DIR = DATA_DIR / "photometry"


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
    wise_data.get_photometric_data(
        service="tap", nthreads=2, chunks=[0, 1], skip_download=True
    )

    wise_data_masked = WiseDataByVisit(
        base_name=TestParentSample.base_name + "_masked",
        parent_sample_class=TestParentSample,
        min_sep_arcsec=6,
        n_chunks=2,
    )
    wise_data_masked.get_photometric_data(
        service="tap",
        nthreads=2,
        chunks=[0, 1],
        mask_by_position=True,
        skip_download=True,
    )

    raw_phot = wise_data_masked._cache_photometry_dir.glob("raw_photometry_*.csv")
    for f in raw_phot:
        shutil.copy(wise_data_masked._cache_photometry_dir / f, PHOT_DIR / f.name)

    unmasked_stacked = wise_data.lightcurve_dir / "timewise_data_product_tap.json"
    shutil.copy(unmasked_stacked, PHOT_DIR / f"{unmasked_stacked.stem}_unmasked.json")

    masked_stacked = wise_data_masked.lightcurve_dir / "timewise_data_product_tap.json"
    shutil.copy(masked_stacked, PHOT_DIR / f"{masked_stacked.stem}_masked.json")

    for i in range(2):
        fn = DATA_DIR / "masks" / f"position_mask_c{i}.json"
        fn.parent.mkdir(parents=True, exist_ok=True)
        with open(fn, "w") as f:
            json.dump(wise_data_masked.get_position_mask("tap", i), f)

    wise_data_masked.plot_diagnostic_binning("tap", 5)
