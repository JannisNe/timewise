import pytest
import sys
from hashlib import blake2b

import pandas as pd
from bson import encode
from astropy.table import Table

from ampel.timewise.ingest.TiMongoMuxer import TiMongoMuxer
from ampel.types import DataPointId, StockId
from ampel.content.DataPoint import DataPoint
from ampel.content.MetaRecord import MetaRecord
from ampel.log.AmpelLogger import DEBUG, AmpelLogger

from tests.constants import DATA_DIR
from tests.dummy_tap import DummyTAPService
from tests.util import dataframe_to_dps


DUPLICATE_FILENAME = DATA_DIR / "duplicate_mep_data.csv"
STOCK_ID = 666


def _populate_photo_col_from_dps(muxer: TiMongoMuxer, dps: list[DataPoint]) -> None:
    """Insert datapoints into the mock photo collection.
    ponytail: ultra - keep it minimal: insert only the fields TiMongoMuxer.find() needs.
    """
    docs = []
    for dp in dps:
        # dp supports dict-like access in tests
        docs.append(
            {
                "id": dp["id"],
                "tag": dp["tag"],
                "channel": dp["channel"],
                "stock": dp["stock"],
                "body": dp["body"],
            }
        )
    # insert into the mocked collection provided by mock_context
    muxer._photo_col.insert_many(docs)


def load_duplicate_data() -> pd.DataFrame:
    return pd.read_csv(DUPLICATE_FILENAME)


def test_muxer_fails_with_duplicates(mock_context):
    data = load_duplicate_data()
    data["table_name"] = "neowiser_p1bs_psd"
    logger = AmpelLogger.get_logger(console=dict(level=DEBUG))
    muxer = TiMongoMuxer(
        logger=logger,
        context=mock_context,
    )
    with pytest.raises(RuntimeError, match="Duplicate photopoints"):
        muxer.process(
            dataframe_to_dps(data, "neowiser_p1bs_psd", STOCK_ID), stock_id=STOCK_ID
        )


def test_muxer_combines(mock_context):
    data = load_duplicate_data()
    unique_cntrs = data["cntr_mf"].unique()
    data = data[data.cntr_mf == unique_cntrs[0]].reset_index(drop=True)
    i_dps = len(data) // 2
    dps_in_db = dataframe_to_dps(data.iloc[:i_dps], "neowiser_p1bs_psd", STOCK_ID)
    alert_dps = dataframe_to_dps(data.iloc[i_dps:], "neowiser_p1bs_psd", STOCK_ID)
    logger = AmpelLogger.get_logger(console=dict(level=DEBUG))
    muxer = TiMongoMuxer(logger=logger, context=mock_context)
    _populate_photo_col_from_dps(muxer, dps_in_db)
    dps_to_insert, dps_to_combine = muxer.process(alert_dps, stock_id=STOCK_ID)

    ids_alert = [dp["id"] for dp in alert_dps]
    all_ids = [dp["id"] for dp in dps_in_db] + ids_alert
    ids_to_insert = [dp["id"] for dp in dps_to_insert]
    ids_to_combine = [dp["id"] for dp in dps_to_combine]
    assert ids_to_insert is not None
    assert sorted(ids_alert) == sorted(ids_to_insert)
    assert sorted(all_ids) == sorted(ids_to_combine)


@pytest.mark.parametrize("data_in_db", [False, True])
def test_muxer_skips_redundant_allwise_mep_data(mock_context, data_in_db: bool):
    data = load_duplicate_data()
    data["table_name"] = "allwise_p3as_mep"
    alert_dps = dataframe_to_dps(data, "allwise_p3as_mep", STOCK_ID)
    logger = AmpelLogger.get_logger(console=dict(level=DEBUG))
    muxer = TiMongoMuxer(logger=logger, context=mock_context)
    valid_cntr = data["cntr_mf"].unique()[0]
    if data_in_db:
        invalid_db_data = [
            dp for dp in alert_dps if dp["body"]["cntr_mf"] != valid_cntr
        ][:2]  # take first 2 invalid datapoints
        _populate_photo_col_from_dps(muxer, invalid_db_data)

    corresponding_valid_dp_id = alert_dps[0]["id"]
    sync_res = Table({"cntr": [valid_cntr], "orig_id": [corresponding_valid_dp_id]})
    muxer._tap_service = DummyTAPService(sync_res=sync_res, baseurl="", chunksize=1)
    dps_to_insert, dps_to_combine = muxer.process(alert_dps, stock_id=STOCK_ID)

    ref_dps = dataframe_to_dps(
        data[data.cntr_mf == valid_cntr], "allwise_p3as_mep", STOCK_ID
    )

    assert dps_to_combine == ref_dps
    assert dps_to_insert == ref_dps
