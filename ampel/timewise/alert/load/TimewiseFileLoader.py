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
from astropy.table import Table, vstack
from ampel.abstract.AbsAlertLoader import AbsAlertLoader
from timewise.tables import TableType
from timewise.config import TimewiseConfig
from timewise.types import TaskID


class TimewiseFileLoader(AbsAlertLoader[Dict]):
    """
    Load alerts from one of more files.
    """

    # path to timewise download config file
    timewise_config_file: str

    # column name of id
    stock_id_column_name: str

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.logger.debug(f"loading timewise config file {self.timewise_config_file}")
        timewise_config = TimewiseConfig.from_yaml(self.timewise_config_file)
        dl = timewise_config.download.build_downloader()
        self._timewise_backend = dl.backend

        self._tasks = [tasks for tasks in dl.iter_tasks_per_chunk()]

        if self.logger:
            self.logger.info(f"Registering {len(self._tasks)} task(s) to load")

        self._table_types = get_args(TableType.__origin__)

        self._gen = self.iter_stocks()

    @staticmethod
    def encode_result(res: Table) -> Table:
        return res

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

            data = vstack(data)

            # rename stock id column
            data.rename_column(self.stock_id_column_name, "stock_id")

            # iterate over all stock ids
            for stock_id in np.unique(data["stock_id"]):
                selection = data[data["stock_id"] == stock_id]
                yield self.encode_result(selection)

    def __iter__(self):
        return self

    def __next__(self) -> Table:
        return next(self._gen)
