#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                ampel/timewise/t1/T1CombineWISEVisits.py
# License:             BSD-3-Clause
# Author:              Jannis Necker <jannis.necker@gmail.com>
# Date:                24.09.2025
# Last Modified Date:  24.09.2025
# Last Modified By:    Jannis Necker <jannis.necker@gmail.com>
from typing import Iterable, Sequence

from ampel.abstract.AbsT1ComputeUnit import AbsT1ComputeUnit
from ampel.abstract.AbsT1CombineUnit import AbsT1CombineUnit
from ampel.base.AuxUnitRegister import AuxUnitRegister
from ampel.content.DataPoint import DataPoint
from ampel.struct.T1CombineResult import T1CombineResult
from ampel.struct.UnitResult import UnitResult
from ampel.model.UnitModel import UnitModel
from ampel.types import DataPointId, StockId, UBson

from ampel.timewise.base.BaseDatapointSelector import BaseDatapointSelector


class T1CombineWISEVisits(AbsT1ComputeUnit, AbsT1CombineUnit):
    select: UnitModel

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._combine: BaseDatapointSelector = AuxUnitRegister.new_unit(
            model=self.select, sub_type=BaseDatapointSelector
        )

    def combine(
        self, datapoints: Iterable[DataPoint]
    ) -> Sequence[DataPointId] | T1CombineResult:
        return self._combine.select(datapoints)

    def compute(
        self, datapoints: list[DataPoint]
    ) -> tuple[UBson | UnitResult, StockId]:
        """
        :param datapoints: list of datapoints to combine
        :return: tuple of UBson or UnitResult and StockId
        """
        pass
