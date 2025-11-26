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
from timewise.io.loader import Loader


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
        self._gen = Loader(
            timewise_config_file=self.timewise_config_file,
            chunks=self.chunks,
            stock_id_column_name=self.stock_id_column_name,
            logger=self.logger,
        ).gen

    def __iter__(self):
        return self

    def __next__(self) -> pd.DataFrame:  # type: ignore
        return next(self._gen)
