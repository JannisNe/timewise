#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                timewise/ampel/timewise/alert/load/TimewiseFileLoader.py
# License:             BSD-3-Clause
# Author:              Jannis Necker <jannis.necker@gmail.com>
# Date:                16.09.2025
# Last Modified Date:  16.09.2025
# Last Modified By:    Jannis Necker <jannis.necker@gmail.com>
from typing import Dict, get_args

import numpy as np
import pandas as pd
from astropy.table import vstack
from ampel.abstract.AbsAlertLoader import AbsAlertLoader
from ampel.base.AmpelABC import AmpelABC
from timewise.tables import TableType
from timewise.config import TimewiseConfig
from timewise.types import TaskID
from timewise.util.path import expand


class TimewiseFileLoader(AbsAlertLoader[Dict], AmpelABC):
    """
    Load alerts from one of more files.
    """

    # path to timewise download config file
    timewise_config_file: str

    # column name of id
    stock_id_column_name: str

    chunks: list[int] | None = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        expanded_config_file = expand(self.timewise_config_file)

        self.logger.info(f"loading timewise config file {expanded_config_file}")
        timewise_config = TimewiseConfig.from_yaml(expanded_config_file)
        dl = timewise_config.download.build_downloader()
        self._timewise_backend = dl.backend

        # selecting tasks to run
        _tasks = [tasks for tasks in dl.iter_tasks_per_chunk()]
        if self.chunks is not None:
            self._tasks = [_tasks[i] for i in self.chunks]
        else:
            self._tasks = _tasks
        if self.logger:
            self.logger.info(
                f"Registering {len(self._tasks)} chunk(s) to load: {self._tasks}"
            )

        self._table_types = get_args(TableType.__origin__)  # type: ignore
        self._gen = self.iter_stocks()

    def find_table_from_task(self, task: TaskID) -> TableType:
        tables = [
            t for t in self._table_types if t.model_fields["name"].default in str(task)
        ]
        assert len(tables) > 0, f"No matching table found for {task}!"
        assert len(tables) < 2, f"More than one matching table found for {task}!"
        self.logger.debug(
            f"{task} is from table {tables[0].model_fields['name'].default}"
        )
        return tables[0]

    @staticmethod
    def merge_allwise_and_neowise_data(table: pd.DataFrame) -> pd.DataFrame:
        # remove the _ep at the end of AllWISE MEP data
        columns_to_rename = [c for c in table.columns if c.endswith("_ep")]
        if len(columns_to_rename):
            rename = {
                c: c.replace("_ep", "")
                for c in columns_to_rename
                if c.replace("_ep", "") not in table.columns
            }
            if rename:
                # in this case only the allwise column eith the _ep extension exists
                # and we can simply rename the columns
                table.rename(columns=rename, inplace=True)

            move = {
                c: c.replace("_ep", "")
                for c in columns_to_rename
                if c.replace("_ep", "") in table.columns
            }
            if move:
                # In this case, the columns already exists because the neowise data is present
                # we have to insert the values form the columns with the _ep extension into the
                # respective neowise columns
                for c, nc in move.items():
                    na_mask = table[nc].isna()
                    table.loc[na_mask, nc] = table[c][na_mask]
                pd.options.mode.chained_assignment = None
                table.drop(columns=[c for c in move], inplace=True)
                pd.options.mode.chained_assignment = "warn"

        return table

    def iter_stocks(self):
        # emit all datapoints per stock id
        # This way ampel runs not per datapoint but per object
        backend = self._timewise_backend
        for tasks in self._tasks:
            data = []
            for task in tasks:
                self.logger.debug(f"reading {task}")
                idata = backend.load_data(task)

                # add table name
                idata["table_name"] = (
                    self.find_table_from_task(task).model_fields["name"].default
                )

                data.append(idata)

            data = vstack(data).to_pandas()

            # rename stock id column
            data.rename(columns={self.stock_id_column_name: "stock_id"}, inplace=True)

            # Find the indices for each stock id. This is much faster than making a mask
            # each loop and accessing the table then. Shown below is a comparison.
            # The top example is the access provided by pandas which would be
            # again a factor 3 faster.
            #
            # In [45]: %timeit test_df()
            # 5.62 μs ± 47.2 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
            #
            # In [46]: %timeit test_index()
            # 14.6 μs ± 45 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
            #
            # In [47]: %timeit test_mask()
            # 2.61 ms ± 18 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)
            data.set_index(data.stock_id, inplace=True)

            # iterate over all stock ids
            for stock_id in np.unique(data["stock_id"]):
                selection = data.loc[stock_id]
                if isinstance(selection, pd.Series):
                    return pd.DataFrame([selection])
                yield self.merge_allwise_and_neowise_data(selection)

    def __iter__(self):
        return self

    def __next__(self) -> pd.DataFrame:  # type: ignore
        return next(self._gen)
