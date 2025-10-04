#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                ampel/timewise/t1/T1CombineWISEVisits.py
# License:             BSD-3-Clause
# Author:              Jannis Necker <jannis.necker@gmail.com>
# Date:                24.09.2025
# Last Modified Date:  24.09.2025
# Last Modified By:    Jannis Necker <jannis.necker@gmail.com>
from typing import Dict

import numpy as np

from ampel.abstract.AbsT1ComputeUnit import AbsT1ComputeUnit
from ampel.content.DataPoint import DataPoint
from ampel.struct.UnitResult import UnitResult
from ampel.types import StockId, UBson
from ampel.model.PlotProperties import PlotProperties

from ampel.timewise.util.pdutil import datapoints_to_dataframe

from timewise.process import keys
from timewise.process.stacking import stack_visits


class T1StackVisits(AbsT1ComputeUnit):
    clean_outliers_when_stacking: bool = True

    plot_properties: PlotProperties | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute(
        self, datapoints: list[DataPoint]
    ) -> tuple[UBson | UnitResult, StockId]:
        """
        :param datapoints: list of datapoints to combine
        :return: tuple of UBson or UnitResult and StockId
        """

        columns = [
            "ra",
            "dec",
            "mjd",
        ]
        for i in range(1, 3):
            for key in [keys.MAG_EXT, keys.FLUX_EXT]:
                columns.extend([f"w{i}{key}", f"w{i}{keys.ERROR_EXT}{key}"])
        raw_lightcurve, stock_ids = datapoints_to_dataframe(datapoints, columns)
        stacked_lightcurve = stack_visits(
            raw_lightcurve, clean_outliers=self.clean_outliers_when_stacking
        )

        # make sure that the is one stock id that fits all dps
        # this is a redundant check, the muxer should take care of it
        unique_stocks = np.unique(np.array(stock_ids).flatten())
        stock_in_all_dps = [
            all([s in sids for sids in stock_ids]) for s in unique_stocks
        ]
        # make sure only one stock is in all datapoints
        assert sum(stock_in_all_dps) == 1
        stock_id = unique_stocks[stock_in_all_dps][0].item()

        return stacked_lightcurve.to_dict(orient="records"), stock_id
