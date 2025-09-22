#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                timewise/ampel/timewise/alert/load/TimewiseFileLoader.py
# License:             BSD-3-Clause
# Author:              Jannis Necker <jannis.necker@gmail.com>
# Date:                16.09.2025
# Last Modified Date:  16.09.2025
# Last Modified By:    Jannis Necker <jannis.necker@gmail.com>
from typing import Dict, List, get_args
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import _typing as pdtype
from ampel.abstract.AbsAlertLoader import AbsAlertLoader
from timewise.tables import TableType


class TimewiseFileLoader(AbsAlertLoader[Dict]):
    """
    Load alerts from one of more files.
    """

    #: paths to files to load
    file: str | list[str]

    # chunk size for reading files in number of lines
    chunk_size: int = 100_000

    # column name of id
    stock_id_column_name: str

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        if not self.file:
            raise ValueError("Parameter 'files' cannot be empty")

        if self.logger:
            self.logger.info(f"Registering {len(self.file)} file(s) to load")

        self._paths = [Path(file) for file in np.atleast_1d(self.file)]

        self._table_types = get_args(TableType.__origin__)

    @staticmethod
    def encode_result(res: List[pd.DataFrame]) -> Dict:
        return pd.concat(res).to_dict(orient="list")

    def find_table_from_path(self, p: Path) -> TableType:
        tables = [
            t for t in self._table_types if t.model_fields["name"].default in p.name
        ]
        assert len(tables) > 0, f"No matching table found for {p}!"
        assert len(tables) < 2, f"More than one matching table found for {p}!"
        self.logger.debug(
            f"{p.name} is from table {tables[0].model_fields['name'].default}"
        )
        return tables[0]

    def find_dtypes_from_path(self, p: Path) -> Dict[str, pdtype.Dtype]:
        table = self.find_table_from_path(p)
        mapping = table.columns_dtypes
        mapping[self.stock_id_column_name] = int
        return mapping

    def __iter__(self):
        return self

    def __next__(self) -> Dict:
        current_stock_id = None

        # emit all datapoints per file and stock id
        # This way ampel runs not per datapoint but per object
        for p in self._paths:
            for f in p.parent.glob(p.name):
                self.logger.debug(f"reading {f}")
                tablegen = pd.read_csv(
                    f,
                    header=0,
                    dtype=self.find_dtypes_from_path(f),
                    engine="c",
                    chunksize=self.chunk_size,
                )

                # set up result list
                res = []

                # iterate over every table chunk:
                for c in tablegen:  # type: pd.DataFrame
                    c.rename(
                        columns={self.stock_id_column_name: "stock_id"}, inplace=True
                    )
                    # iterate over all stock ids
                    for stock_id in np.unique(c["stock_id"]):
                        selection = c[c["stock_id"] == stock_id]

                        if (stock_id == current_stock_id) and len(selection):
                            res.append(selection)

                        # emit the previous stock id result if present
                        else:
                            if res:
                                return self.encode_result(res)

                            # set up next result list and update current stock id
                            res = [selection] if len(selection) else []
                            current_stock_id = stock_id

                # emit the result for the last stock id
                if res:
                    return self.encode_result(res)
