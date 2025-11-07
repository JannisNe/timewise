#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                timewise/ampel/timewise/t1/TimewiseFilter.py
# License:             BSD-3-Clause
# Author:              Jannis Necker <jannis.necker@gmail.com>
# Date:                16.09.2025
# Last Modified Date:  16.09.2025
# Last Modified By:    Jannis Necker <jannis.necker@gmail.com>
import numpy as np

from ampel.abstract.AbsAlertFilter import AbsAlertFilter
from ampel.protocol.AmpelAlertProtocol import AmpelAlertProtocol

from timewise.util.visits import get_visit_map
from timewise.process import keys


class TimewiseFilter(AbsAlertFilter):
    det_per_visit: int = 8
    n_visits = 10

    def process(self, alert: AmpelAlertProtocol) -> None | bool | int:
        columns = [
            "ra",
            "dec",
            "mjd",
        ]
        for i in range(1, 3):
            for key in [keys.MAG_EXT, keys.FLUX_EXT]:
                columns.extend([f"w{i}{key}", f"w{i}{keys.ERROR_EXT}{key}"])

        mjd = np.array([dp["mjd"] for dp in alert.datapoints])

        # enough single detections per visit
        visit_map = get_visit_map(mjd)
        visits, counts = np.unique(visit_map, return_counts=True)
        visit_passed = counts >= self.det_per_visit
        if not all(visit_passed):
            self.logger.debug(None, extra={"min_det_per_visit": min(counts).item()})
            return None

        # enough visits
        if not len(visits) >= self.n_visits:
            self.logger.debug(None, extra={"n_visits": len(visits)})
            return None

        return True
