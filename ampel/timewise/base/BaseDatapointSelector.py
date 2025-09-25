#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                ampel/timewise/base/BaseDatapointSelector.py
# License:             BSD-3-Clause
# Author:              Jannis Necker <jannis.necker@gmail.com>
# Date:                26.09.2025
# Last Modified Date:  26.09.2025
# Last Modified By:    Jannis Necker <jannis.necker@gmail.com>

from typing import Iterable, Sequence

from ampel.content.DataPoint import DataPoint
from ampel.struct.T1CombineResult import T1CombineResult
from ampel.types import DataPointId
from ampel.base.AmpelBaseModel import AmpelBaseModel
from ampel.base.AmpelABC import AmpelABC
from ampel.base.decorator import abstractmethod


class BaseDatapointSelector(AmpelBaseModel, AmpelABC, abstract=True):
    @abstractmethod
    def select(
        self, datapoints: Iterable[DataPoint]
    ) -> Sequence[DataPointId] | T1CombineResult: ...
