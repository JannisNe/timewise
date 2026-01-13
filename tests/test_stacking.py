import logging

import numpy as np
import pandas as pd
import pytest
from astropy.table import Table, vstack
from numpy import typing as npt
from tests.constants import AMPEL_CONFIG_PATH, DATA_DIR
from tests.util import get_raw_reference_photometry, get_stacked_reference_photometry
from timewise.config import TimewiseConfig

logger = logging.getLogger(__name__)


def test_ingest(ampel_interface, timewise_config_path):
    # switch out the template so only ingestion is run
    ingestion_only_template = DATA_DIR / "template_ingest_only.yml"
    ampel_interface.template_path = ingestion_only_template
    ampel_interface.run(timewise_config_path, AMPEL_CONFIG_PATH)

    # ----------------------------
    # check t0 collection
    # ----------------------------
    t0 = ampel_interface.t0
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
def test_stacking(
    ampel_interface,
    timewise_config_path,
    mode,
):
    if mode == "unmasked":
        ampel_interface.template_path = DATA_DIR / "template_stack_all.yml"
    if mode == "masked":
        ampel_interface.template_path = DATA_DIR / "template_stack.yml"

    mongo_db_name = ampel_interface.mongo_db_name + "_" + mode
    ampel_interface.mongo_db_name = mongo_db_name
    ampel_interface.run(timewise_config_path, AMPEL_CONFIG_PATH)

    # ----------------------------
    # check t1 collection
    # ----------------------------
    t1 = ampel_interface.t1
    assert t1.count_documents({}) > 0

    cfg = TimewiseConfig.from_yaml(timewise_config_path)
    input_csv_path = cfg.download.input_csv
    input_data = pd.read_csv(input_csv_path)

    records = []
    index = []
    for i in input_data.orig_id.astype(int):
        if mode == "masked":
            # ----------------------------
            # check masking
            # ----------------------------

            # load reference
            reference_photometry = get_raw_reference_photometry(i)

            # load result
            t1_result = t1.find_one({"stock": i})

            if t1_result is None:
                # In this case, all datapoints were masked. We just have to check that
                # this is also the case in the reference
                assert len(reference_photometry) == 0
                continue

            dp_ids = t1_result["dps"]
            selected_photometry = ampel_interface.extract_datapoints(i).loc[dp_ids]

            index_map = []
            for j, r in reference_photometry.iterrows():
                m = (
                    (r.ra == selected_photometry.ra)
                    & (r.dec == selected_photometry.dec)
                    & (r.mjd == selected_photometry.mjd)
                )
                assert m.sum() == 1, (
                    f"None or more than one match found for datapoint {j}, stock {i}"
                )
                index_map.append((j, selected_photometry.index[m][0]))
            index_map = np.array(index_map)

            cols = ["ra", "dec", "mjd"]
            diff = (
                reference_photometry.loc[index_map[:, 0]].set_index(index_map[:, 1])[
                    cols
                ]
                - selected_photometry[cols]
            )
            assert all(diff.sum() == 0)

        # ----------------------------
        # check stacking result
        # ----------------------------

        stacked_lc = ampel_interface.extract_stacked_lightcurve(i)
        reference_lc = get_stacked_reference_photometry(i, mode)

        if reference_lc is None:
            # in this case all datapoints were masked so we just have to make sure that the
            # stacked lightcurve also contains no data
            assert len(stacked_lc) == 0, f"Found too many datapoints for {i}"
            continue

        n_epochs_diff = len(reference_lc) - len(stacked_lc)
        diff = reference_lc.astype(float) - stacked_lc.astype(float)

        # changed to > 9 because scipy v1.17.0 introduced some numerical difference in
        # calculation of stats.t.interval, introducing a difference O(1e-9) in the RMS
        # of the stacked fluxes
        m = diff > 1e-8

        n_bad_epochs = (m.any(axis=1) | m.isna().any(axis=1)).sum()

        try:
            datapoints_diff = min(
                [
                    (
                        stacked_lc[f"{b}fluxdensitynpoints"]
                        - reference_lc[f"{b}fluxdensitynpoints"]
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

    all_zero: npt.ArrayLike = res.sum() == 0

    assert all(all_zero)
