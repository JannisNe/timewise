import json
import logging

from astropy.table import Table, vstack
from pymongo import MongoClient
import pandas as pd
import pytest
import numpy as np

from timewise.config import TimewiseConfig


from tests.constants import AMPEL_CONFIG_PATH, DATA_DIR


logger = logging.getLogger(__name__)


def test_ingest(ampel_prepper, timewise_config_path):
    # switch out the template so only ingestion is run
    ingestion_only_template = DATA_DIR / "template_ingest_only.yml"
    ampel_prepper.template_path = ingestion_only_template
    ampel_prepper.run(timewise_config_path, AMPEL_CONFIG_PATH)

    client = MongoClient()

    # ----------------------------
    # check t0 collection
    # ----------------------------
    t0 = client["test_ampel"].get_collection("t0")
    n_in_db = t0.count_documents({})

    bp = TimewiseConfig.from_yaml(timewise_config_path).download.backend.base_path
    file_contents = vstack(
        [Table.read(f, format="fits") for f in bp.glob("download_chunk*.fits")]
    )

    missing = []
    duplicates = []
    for r in file_contents:
        f = {
            "body.ra": r["ra"],
            "body.dec": r["dec"],
            "body.mjd": r["mjd"],
        }
        n_doc = t0.count_documents(f)
        if n_doc == 0:
            missing.append(r)
        if n_doc > 1:
            duplicates.append(list(t0.find(f)))

    assert len(missing) == 0, f"Missing {len(missing)} documents in DB!\n{missing}"
    assert len(duplicates) == 0, f"Duplicate documents!\n{duplicates}"
    assert n_in_db == len(file_contents)


@pytest.mark.parametrize("mode", ["masked", "unmasked"])
def test_stacking(ampel_prepper, timewise_config_path, mode):
    if mode == "unmasked":
        ampel_prepper.template_path = DATA_DIR / "template_stack_all.yml"

    ampel_prepper.run(timewise_config_path, AMPEL_CONFIG_PATH)
    # ----------------------------
    # check t1 collection
    # ----------------------------
    client = MongoClient()
    t1 = client["test_ampel"].get_collection("t1")
    assert t1.count_documents({}) > 0

    cfg = TimewiseConfig.from_yaml(timewise_config_path)
    extractor = cfg.ampel.build_extractor()

    input_data = pd.read_csv(cfg.download.input_csv)

    reference_path = DATA_DIR / "photometry" / f"timewise_data_product_tap_{mode}.json"
    with reference_path.open("r") as f:
        reference_data = json.load(f)

    records = []
    index = []
    for i in input_data.orig_id.astype(int):
        stacked_lc = extractor.extract_stacked_lightcurve(i)

        if "timewise_lightcurve" not in reference_data[str(i)]:
            # in this case all datapoints were masked so we just have to make sure that the
            # stacked lightcurve also contains no data
            assert len(stacked_lc) == 0
            continue
        reference_lc = pd.DataFrame(reference_data[str(i)]["timewise_lightcurve"])
        reference_lc.set_index(reference_lc.index.astype(int), inplace=True)

        n_epochs_diff = len(reference_lc) - len(stacked_lc)

        diff = reference_lc.astype(float) - stacked_lc.astype(float)
        m = diff > 1e-10
        n_bad_epochs = (m.any(axis=1) | m.isna().any(axis=1)).sum()

        try:
            datapoints_diff = min(
                [
                    (
                        stacked_lc[f"{b}_flux_density_Npoints"]
                        - reference_lc[f"{b}_flux_density_Npoints"]
                    ).sum()
                    for b in ["W1", "W2"]
                ]
            )
        except KeyError:
            datapoints_diff = np.nan

        records.append(
            {
                "n_epochs_diff": n_epochs_diff,
                "n_bad_epochs": n_bad_epochs,
                "datapoints_diff": datapoints_diff,
            }
        )
        index.append(i)

    res = pd.DataFrame(records, index=index)
    logger.info("\n" + res.to_string())
