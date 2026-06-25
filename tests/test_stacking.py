import pytest
import logging
from pathlib import Path
import pandas as pd
import numpy.typing as npt

from timewise.process.stacking import stack_visits
from timewise.config import TimewiseConfig
from tests.util import (
    get_raw_reference_photometry,
    get_stacked_reference_photometry,
    check_stacking_result,
    dataframe_to_dps,
)

from ampel.timewise.t2.T2StackVisits import T2StackVisits
from ampel.log.AmpelLogger import DEBUG, AmpelLogger
from ampel.view.LightCurve import LightCurve

logger = logging.getLogger(__name__)


def test_stack_visits(timewise_config_path: Path):
    cfg = TimewiseConfig.from_yaml(timewise_config_path)
    input_data = pd.read_csv(cfg.download.input_csv)
    records = []
    index = []
    t2 = T2StackVisits(
        logger=AmpelLogger.get_logger(console=dict(level=DEBUG)),
        outlier_threshold=20,
        outlier_quantile=0.7,
    )
    for i in input_data.orig_id.astype(int):
        # build lightcurve
        raw_photometry = get_raw_reference_photometry(i)

        if len(raw_photometry) == 0:
            continue

        allwise_mask = raw_photometry.cntr_mf.notna()
        dps = dataframe_to_dps(
            raw_photometry[allwise_mask], table_name="allwise_p3as_mep", stock_id=i
        ) + dataframe_to_dps(
            raw_photometry[~allwise_mask], table_name="neowiser_p1bs_psd", stock_id=i
        )

        result = t2.process(LightCurve(compound_id=i, stock_id=i, photopoints=dps))
        # ----------------------------
        # check stacking result
        # ----------------------------

        stacked_lc = pd.DataFrame.from_records(result)
        reference_lc = get_stacked_reference_photometry(i, "masked")

        if (check := check_stacking_result(stacked_lc, reference_lc)) is not None:
            records.append(check)
            index.append(i)

    res = pd.DataFrame(records, index=index)
    logger.info("\n" + res.to_string())

    all_zero: npt.ArrayLike = res.sum() == 0

    assert all(all_zero)
