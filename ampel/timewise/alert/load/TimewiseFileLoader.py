#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                timewise/ampel/timewise/alert/load/TimewiseFileLoader.py
# License:             BSD-3-Clause
# Author:              Jannis Necker <jannis.necker@gmail.com>
# Date:                16.09.2025
# Last Modified Date:  16.09.2025
# Last Modified By:    Jannis Necker <jannis.necker@gmail.com>

from typing import Dict, List
from pathlib import Path
from astropy.table import Table, vstack

import numpy as np
from ampel.abstract.AbsAlertLoader import AbsAlertLoader


class TimewiseFileLoader(AbsAlertLoader[Dict]):
    """
    Load alerts from one of more files.
    """

    #: paths to files to load
    file: str | list[str]

    # chunk size for reading files in MB
    chunk_size: int = 100

    # column name of id
    stock_id_column_name: str

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        if not self.file:
            raise ValueError("Parameter 'files' cannot be empty")

        if self.logger:
            self.logger.info(f"Registering {len(self.file)} file(s) to load")

        self._paths = [Path(file) for file in np.atleast_1d(self.file)]

    @staticmethod
    def encode_result(res: List[Table]) -> Dict:
        return vstack(res).to_pandas().to_dict()

    def __iter__(self):
        return self

    def __next__(self) -> Dict:
        current_stock_id = None

        # emit all datapoints per file and stock id
        # This way ampel runs not per datapoint but per object
        for p in self._paths:
            for f in p.parent.glob(p.name):
                table = Table.read(
                    f,
                    format="csv",
                    guess=False,
                    fast_reader={
                        "chunk_size": 100 * self.chunk_size,
                        "chunk_generator": True,
                    },
                )

                table.rename_colum(self.stock_id_column_name, "stock_id")

                # set up result list
                res = []

                # iterate over every table chunk:
                for c in table:
                    # iterate over all stock ids
                    for stock_id in np.unique(c["stock_id"]):
                        selection = c[c["stock_id"] == stock_id]

                        if (stock_id == current_stock_id) and selection:
                            res.append(selection)

                        # emit the previous stock id result if present
                        else:
                            if res:
                                return self.encode_result(res)

                            # set up next result list and update current stock id
                            res = [selection] if selection else []
                            current_stock_id = stock_id

                # emit the result for the last stock id
                if res:
                    return self.encode_result(res)
