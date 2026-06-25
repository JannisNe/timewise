import pytest
import pandas as pd

from ampel.timewise.t1.T1HDBSCAN import T1HDBSCAN
from ampel.timewise.util.pdutil import datapoints_to_dataframe
from ampel.log.AmpelLogger import DEBUG, AmpelLogger

from tests.util import (
    dataframe_to_dps,
    get_single_exposure_photometry,
    get_raw_reference_photometry,
)


def test_hdbscan(tmp_db_name, mongomock_client, ampel_interface, mock_context):
    ampel_interface.import_input()
    t1 = T1HDBSCAN(
        input_mongo_db_name=ampel_interface.input_mongo_db_name,
        original_id_key="orig_id",
        mongo="mongodb://localhost:27017/",
        plot=False,
        logger=AmpelLogger.get_logger(console=dict(level=DEBUG)),
    )
    input = pd.read_csv(ampel_interface.input_csv)

    for i in input.orig_id.astype(int):
        # combine test data
        single_exposure_data = get_single_exposure_photometry(i)
        allwise_mask = single_exposure_data.cntr_mf.notna()
        allwise_dps = dataframe_to_dps(
            single_exposure_data[allwise_mask],
            stock_id=i,
            table_name="allwise_p3as_mep",
        )
        neowise_dps = dataframe_to_dps(
            single_exposure_data[~allwise_mask],
            stock_id=i,
            table_name="neowiser_p1bs_psd",
        )
        ids = t1.combine(allwise_dps + neowise_dps).dps

        # check result
        combined_dps = [dp for dp in allwise_dps + neowise_dps if dp["id"] in ids]
        ref_data = get_raw_reference_photometry(i)
        assert len(ref_data) == len(combined_dps)
        for dp in combined_dps:
            body = dp["body"]
            m = (
                (ref_data.ra == body["ra"])
                & (ref_data.dec == body["dec"])
                & (ref_data.mjd == body["mjd"])
            )
            assert sum(m) == 1
