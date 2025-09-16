#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                timewise/ampel/timewise/alert/load/TimewiseFileLoader.py
# License:             BSD-3-Clause
# Author:              Jannis Necker <jannis.necker@gmail.com>
# Date:                16.09.2025
# Last Modified Date:  16.09.2025
# Last Modified By:    Jannis Necker <jannis.necker@gmail.com>

from typing import Dict
from pathlib import Path
from astropy.table import Table

import numpy as np
from ampel.abstract.AbsAlertLoader import AbsAlertLoader


class WiseFileAlertLoader(AbsAlertLoader[Dict]):
    """
    Load alerts from one of more files.
    """

    #: paths to files to load
    file: str | list[str]

    # chunk size for reading files in MB
    chunk_size: int = 100

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        if not self.file:
            raise ValueError("Parameter 'files' cannot be empty")

        if self.logger:
            self.logger.info(f"Registering {len(self.file)} file(s) to load")

        self.paths = [Path(file) for file in np.atleast_1d(self.file)]

    def __iter__(self):
        return self

    def __next__(self) -> Dict:
        for p in self.paths:
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

                # iterate over every table chunk:
                for c in table:
                    # iterate over every row in the table
                    for row in c:
                        yield dict(row)
