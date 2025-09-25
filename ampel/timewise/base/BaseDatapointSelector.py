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
from ampel.log.AmpelLogger import AmpelLogger
from ampel.struct.Resource import Resource


class BaseDatapointSelector(AmpelBaseModel, AmpelABC, abstract=True):
    @property
    def logger(self) -> AmpelLogger:
        return self._logger

    @property
    def resources(self) -> dict[str, Resource]:
        return self._resources

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._logger: AmpelLogger = AmpelLogger.get_logger()
        self._resources: dict[str, Resource] = {}

    def set_logger(self, logger: AmpelLogger) -> None:
        self._logger = logger

    def add_resource(self, name: str, value: Resource) -> None:
        self._resources[name] = value

    @abstractmethod
    def select(
        self, datapoints: Iterable[DataPoint]
    ) -> Sequence[DataPointId] | T1CombineResult: ...
