from typing import Iterable, List
import pandas as pd
import numpy as np
from ampel.types import StockId


def datapoints_to_dataframe(
    datapoints: Iterable[dict], columns: list[str]
) -> tuple[pd.DataFrame, List[List[StockId]]]:
    """
    Convert a list of Ampel DataPoints into a pandas DataFrame.

    Parameters
    ----------
    datapoints : list of dict
        List of DataPoints (each must have a "body" dict).
    columns : list of str
        Keys from datapoint["body"] to include as DataFrame columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per DataPoint and one column per requested key.
    """

    records = []
    ids = []
    stock_ids = []
    for dp in datapoints:
        body = dp.get("body", {})
        # Build one row with only requested keys
        row = {col: body.get(col, None) for col in columns}
        # build the index from datapoint ids
        ids.append(dp["id"])
        records.append(row)
        stock_ids.append(np.atleast_1d(dp["stock"]).tolist())

    return pd.DataFrame.from_records(records, columns=columns, index=ids), stock_ids
