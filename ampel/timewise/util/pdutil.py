from typing import Iterable, List
import pandas as pd
import numpy as np
from ampel.types import StockId
from ampel.content.DataPoint import DataPoint


def datapoints_to_dataframe(
    datapoints: Iterable[DataPoint],
    columns: list[str],
    check_tables: list[str] | None = None,
) -> tuple[pd.DataFrame, List[List[StockId]]]:
    """
    Convert a list of Ampel DataPoints into a pandas DataFrame.

    Parameters
    ----------
    datapoints : list of dict
        List of DataPoints (each must have a "body" dict).
    columns : list of str
        Keys from datapoint["body"] to include as DataFrame columns.
    check_tables: list of str
        check if the tables are in the tags

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
        row = {col: body[col] for col in columns}
        # check if the tables are in tags
        if check_tables is not None:
            for table in check_tables:
                row[table] = table in dp["tag"]
        # build the index from datapoint ids
        ids.append(dp["id"])
        records.append(row)
        stock_ids.append(np.atleast_1d(dp["stock"]).tolist())

    colnames = columns if check_tables is None else columns + check_tables
    return pd.DataFrame.from_records(records, columns=colnames, index=ids), stock_ids
