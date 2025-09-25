#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : ampel/timewise/t2/T2StackVisits.py
# License           : BSD-3-Clause
# Author            : Jannis Necker <jannis.necker@gmail.com>
# Date              : 26.09.2025
# Last Modified Date: 26.09.2025
# Last Modified By  : Jannis Necker <jannis.necker@gmail.com>


from ampel.abstract.AbsLightCurveT2Unit import AbsLightCurveT2Unit
from ampel.struct.UnitResult import UnitResult
from ampel.types import UBson
from ampel.view.LightCurve import LightCurve

# ruff: noqa: E712


class T2StackVisits(AbsLightCurveT2Unit):
    def process(self, light_curve: LightCurve) -> UBson | UnitResult:
        return {"stock_id": light_curve.stock_id}
