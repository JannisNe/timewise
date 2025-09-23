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
from ampel.abstract.AbsAlertLoader import AbsAlertLoader
from timewise.tables import TableType
from timewise.config import TimewiseConfig
from timewise.io.download import Downloader


class TimewiseFileLoader(AbsAlertLoader[Dict]):
    """
    Load alerts from one of more files.
    """

    # path to timewise download config file
    timewise_config_file: str

    # chunk size for reading files in number of lines
    chunk_size: int = 100_000

    # column name of id
    stock_id_column_name: str

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.logger.debug(f"loading timewise config file {self.timewise_config_file}")
        timewise_config = TimewiseConfig.from_yaml(self.timewise_config_file)
        dl = Downloader(timewise_config.download)

        self._paths = [dl.backend._data_path(task) for task in dl.iter_tasks()]
        self._files = np.array(
            [list(p.parent.glob(p.name)) for p in self._paths]
        ).flatten()

        if self.logger:
            self.logger.info(f"Registering {len(self._files)} file(s) to load")

        self._table_types = get_args(TableType.__origin__)

        self._gen = self.iter_stocks()

    @staticmethod
    def encode_result(res: List[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(res)

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

    def iter_stocks(self):
        current_stock_id = None

        # emit all datapoints per file and stock id
        # This way ampel runs not per datapoint but per object
        for p in self._paths:
            for f in p.parent.glob(p.name):
                self.logger.debug(f"reading {f}")

                # find which table the data comes from and use the corresponding dtype
                table = self.find_table_from_path(f)
                dtype_mapping = table.columns_dtypes
                dtype_mapping[self.stock_id_column_name] = int

                tablegen = pd.read_csv(
                    f,
                    header=0,
                    dtype=dtype_mapping,
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

                    c["table_name"] = table.model_fields["name"].default

                    # iterate over all stock ids
                    for stock_id in np.unique(c["stock_id"]):
                        selection = c[c["stock_id"] == stock_id]

                        if (stock_id == current_stock_id) and len(selection):
                            res.append(selection)

                        # emit the previous stock id result if present
                        else:
                            if res:
                                yield self.encode_result(res)

                            # set up next result list and update current stock id
                            res = [selection] if len(selection) else []
                            current_stock_id = stock_id

                # emit the result for the last stock id
                if res:
                    yield self.encode_result(res)

    def __iter__(self):
        return self

    def __next__(self) -> pd.DataFrame:
        return next(self._gen)
