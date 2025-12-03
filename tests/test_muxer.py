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
from ampel.test.conftest import mock_context, _patch_mongo, testing_config

from tests.constants import DATA_DIR
from tests.dummy_tap import DummyTAPService


DUPLICATE_FILENAME = DATA_DIR / "duplicate_mep_data.csv"
STOCK_ID = 666


class TestMuxer(TiMongoMuxer):
    dps_in_db: list[DataPoint] = []

    def _get_dps(self, stock_id: None | StockId) -> list[DataPoint]:
        return self.dps_in_db


def dataframe_to_dps(df: pd.DataFrame, table_name: str) -> list[DataPoint]:
    dps = []
    for _, row in df.iterrows():
        pp = {k: None if pd.isna(v) else v for k, v in row.to_dict().items()}
        pp_hash = blake2b(encode(pp), digest_size=7).digest()
        dp_id = int.from_bytes(pp_hash, byteorder=sys.byteorder)
        meta = MetaRecord()
        dp = DataPoint(
            id=DataPointId(dp_id),
            stock=STOCK_ID,
            channel=["test_channel"],
            body=row.to_dict(),
            meta=[meta],
            tag=["TIMEWISE", table_name],
        )
        dps.append(dp)
    return dps


def load_duplicate_data() -> pd.DataFrame:
    return pd.read_csv(DUPLICATE_FILENAME)


def test_muxer_fails_with_duplicates(mock_context):
    data = load_duplicate_data()
    data["table_name"] = "neowiser_p1bs_psd"
    logger = AmpelLogger.get_logger(console=dict(level=DEBUG))
    muxer = TestMuxer(
        logger=logger,
        context=mock_context,
    )
    with pytest.raises(RuntimeError, match="Duplicate photopoints"):
        muxer.process(dataframe_to_dps(data, "neowiser_p1bs_psd"), stock_id=STOCK_ID)


def test_muxer_combines(mock_context):
    data = load_duplicate_data()
    unique_cntrs = data["cntr_mf"].unique()
    data = data[data.cntr_mf == unique_cntrs[0]].reset_index(drop=True)
    i_dps = len(data) // 2
    dps_in_db = dataframe_to_dps(data.iloc[:i_dps], "neowiser_p1bs_psd")
    alert_dps = dataframe_to_dps(data.iloc[i_dps:], "neowiser_p1bs_psd")
    logger = AmpelLogger.get_logger(console=dict(level=DEBUG))
    muxer = TestMuxer(dps_in_db=dps_in_db, logger=logger, context=mock_context)
    dps_to_insert, dps_to_combine = muxer.process(alert_dps, stock_id=STOCK_ID)

    ids_alert = [dp["id"] for dp in alert_dps]
    all_ids = [dp["id"] for dp in dps_in_db] + ids_alert
    ids_to_insert = [dp["id"] for dp in dps_to_insert]
    ids_to_combine = [dp["id"] for dp in dps_to_combine]
    assert ids_to_insert is not None
    assert sorted(ids_alert) == sorted(ids_to_insert)
    assert sorted(all_ids) == sorted(ids_to_combine)


def test_muxer_skips_redundant_allwise_mep_data(mock_context):
    data = load_duplicate_data()
    data["table_name"] = "allwise_p3as_mep"
    alert_dps = dataframe_to_dps(data, "allwise_p3as_mep")
    logger = AmpelLogger.get_logger(console=dict(level=DEBUG))
    muxer = TestMuxer(logger=logger, context=mock_context)
    valid_cntr = data["cntr_mf"].unique()[0]
    corresponding_valid_dp_id = alert_dps[0]["id"]
    sync_res = Table({"cntr": [valid_cntr], "orig_id": [corresponding_valid_dp_id]})
    muxer._tap_service = DummyTAPService(sync_res=sync_res, baseurl="", chunksize=1)
    dps_to_insert, dps_to_combine = muxer.process(alert_dps, stock_id=STOCK_ID)

    ref_dps = dataframe_to_dps(data[data.cntr_mf == valid_cntr], "allwise_p3as_mep")

    assert dps_to_combine == ref_dps
    assert dps_to_insert == ref_dps
