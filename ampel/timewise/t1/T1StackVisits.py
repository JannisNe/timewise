#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                ampel/timewise/t1/T1CombineWISEVisits.py
# License:             BSD-3-Clause
# Author:              Jannis Necker <jannis.necker@gmail.com>
# Date:                24.09.2025
# Last Modified Date:  24.09.2025
# Last Modified By:    Jannis Necker <jannis.necker@gmail.com>

from ampel.abstract.AbsT1ComputeUnit import AbsT1ComputeUnit
from ampel.base.AuxUnitRegister import AuxUnitRegister
from ampel.content.DataPoint import DataPoint
from ampel.struct.UnitResult import UnitResult
from ampel.model.UnitModel import UnitModel
from ampel.types import StockId, UBson

from ampel.timewise.base.BaseDatapointSelector import BaseDatapointSelector


class T1StackVisits(AbsT1ComputeUnit):
    select: UnitModel

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._combine: BaseDatapointSelector = AuxUnitRegister.new_unit(
            model=self.select, sub_type=BaseDatapointSelector
        )

    def compute(
        self, datapoints: list[DataPoint]
    ) -> tuple[UBson | UnitResult, StockId]:
        """
        :param datapoints: list of datapoints to combine
        :return: tuple of UBson or UnitResult and StockId
        """
        self.logger.info("Computing something!")
        return {"computed": True}, datapoints[0]["stock"]
